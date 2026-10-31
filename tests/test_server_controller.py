"""The controller's decision step over HTTP and WebSocket (fake mode, no network, no browser)."""
import time

import pytest
from starlette.testclient import TestClient

from negotiator.fake_llm import FakeLLM
from negotiator.log import ConsensusLog, LogEntry
from negotiator.parsing import ParseResult
from negotiator.server import create_app

from .test_server import cfg, client, collect, drain_snapshot  # noqa: F401  (shared fixtures and helpers)


def start(client, **extra):
    res = client.post("/api/run", json={"controller": "manual", "step_delay_s": 0} | extra)
    assert res.status_code == 202, res.text
    return res.json()["run_id"]


def until(ws, kinds, limit=300):
    """Read events until one of `kinds` arrives; returns (everything read, the event that matched)."""
    seen = []
    for _ in range(limit):
        ev = ws.receive_json()
        seen.append(ev)
        if ev["type"] in kinds:
            return seen, ev
    raise AssertionError(f"never saw {kinds}; last: {[e['type'] for e in seen][-5:]}")


def answer(client, ev, **body):
    return client.post("/api/decision", json={"prompt_id": ev["prompt_id"]} | body)


def finished(ws):
    return until(ws, ("run_finished", "run_error"))


def settle_open_questions(client):
    """Answer whatever is still open so the run ends now instead of waiting out the decision timeout."""
    manager = client.app.state.manager
    for _ in range(10):
        pending = manager.pending
        if manager.status != "running" or pending is None:
            return
        client.post("/api/decision", json={"prompt_id": pending.prompt_id, "action": "approve", "option": 1})
        time.sleep(0.2)


@pytest.fixture
def session(client):
    with client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        start(client)
        _, options = until(ws, ("options",))
        yield client, ws, options
        settle_open_questions(client)


# --- approve ---------------------------------------------------------------------------------------------------


def test_the_agents_finish_and_the_controller_is_asked(session):
    client, ws, ev = session
    assert [o["rank"] for o in ev["options"]] == [1, 2] and ev["attempt"] == 1 and ev["rejections_left"] == 2
    assert ev["timeout_s"] == 60 and 0 < ev["remaining_s"] <= 60 and ev["problem"] is None and ev["rules"] == []
    assert "re-checked by the validator" in ev["rank_rule"]
    assert ev["options"][0]["title"].startswith("Recommended") and ev["options"][0]["chips"][0].startswith("Ground delay")
    assert client.get("/api/state").json()["status"] == "running"  # the run waits for the human


def test_approving_option_one_finishes_the_run(session):
    client, ws, ev = session
    assert answer(client, ev, action="approve", option=1).status_code == 202
    seen, fin = finished(ws)
    assert fin["type"] == "run_finished" and fin["outcome"] == "APPROVED" and fin["option"] == 1 and fin["controller"] == "manual"
    assert fin["negotiations"] == 1 and fin["rejections"] == 0 and fin["rules"] == [] and fin["log_valid"] is True
    assert fin["proposal"]["proposal_id"] == ev["options"][0]["id"] and fin["proposal_chips"][0].startswith("Ground delay")
    kinds = [e["type"] for e in seen]
    assert "prompt_closed" in kinds and kinds.index("prompt_closed") < kinds.index("run_finished")
    events = [e["entry"]["event"] for e in seen if e["type"] == "entry"]
    assert events[-2:] == ["DECISION", "FINAL"]


def test_approving_option_two(session):
    client, ws, ev = session
    answer(client, ev, action="approve", option=2)
    _, fin = finished(ws)
    assert fin["outcome"] == "APPROVED" and fin["option"] == 2 and fin["proposal"]["proposal_id"] == ev["options"][1]["id"]


# --- reject ----------------------------------------------------------------------------------------------------------


def test_a_rejection_renegotiates_and_asks_again_with_the_rule(session):
    client, ws, ev = session
    gate = ev["options"][0]["proposal"]["gate"]
    assert answer(client, ev, action="reject", text=f"avoid gate {gate}").status_code == 202
    seen, again = until(ws, ("options", "run_finished"))
    assert again["type"] == "options" and again["attempt"] == 2 and again["rejections_left"] == 1 and again["prompt_id"] != ev["prompt_id"]
    assert again["rules"] == [f"gate ≠ {gate}"] and all(o["proposal"]["gate"] != gate for o in again["options"])
    notes = [e["entry"]["note"] for e in seen if e["type"] == "entry"]
    assert any(f"gate ≠ {gate}" in n for n in notes) and any("Renegotiation 1 of 2" in n for n in notes)
    summaries = [e for e in seen if e["type"] == "round_summary"]
    assert summaries and all(s["attempt"] == 2 for s in summaries if s["text"].startswith("Renegotiation"))
    answer(client, again, action="approve", option=1)
    _, fin = finished(ws)
    assert fin["outcome"] == "APPROVED" and fin["negotiations"] == 2 and fin["rejections"] == 1 and fin["rules"] == [f"gate ≠ {gate}"]


def test_text_that_is_not_a_rule_asks_again_without_using_a_rejection(session):
    client, ws, ev = session
    answer(client, ev, action="reject", text="make it better")
    _, again = until(ws, ("options",))
    assert again["attempt"] == 1 and again["rejections_left"] == 2 and "I did not understand" in again["problem"]
    assert again["prompt_id"] != ev["prompt_id"] and [o["id"] for o in again["options"]] == [o["id"] for o in ev["options"]]


def test_a_quick_reason_sent_over_http_renegotiates(session):
    client, ws, ev = session
    assert "plain" in ev["options"][0] and ev["options"][0]["plain"].endswith("behind the aircraft ahead.")
    sep = ev["options"][0]["proposal"]["separation_sec"]
    assert answer(client, ev, action="reject", text=f"no hold; separation at least {sep + 60}").status_code == 202
    _, again = until(ws, ("options",))
    assert again["rules"] == ["air hold ≤ 0 min", f"separation ≥ {sep + 60} s"] and again["attempt"] == 2
    assert all(o["proposal"]["hold_minutes"] == 0 and o["proposal"]["separation_sec"] >= sep + 60 for o in again["options"][:1])


def test_the_third_rejection_ends_in_the_fallback(session):
    client, ws, ev = session
    for text in ("avoid gate A1", "no hold", "avoid runway 27L"):
        answer(client, ev, action="reject", text=text)
        seen, ev = until(ws, ("options", "run_finished"))
        if ev["type"] == "run_finished":
            break
    assert ev["type"] == "run_finished" and ev["outcome"] in ("REJECTED_FALLBACK", "ESCALATED") and ev["rejections"] == 3
    assert ev["negotiations"] == 3 and len(ev["rules"]) == 3


# --- bad answers --------------------------------------------------------------------------------------------------------


def test_answers_are_validated(session):
    client, ws, ev = session
    assert answer(client, ev, action="approve", option=9).status_code == 400
    assert answer(client, ev, action="approve").status_code == 400
    assert answer(client, ev, action="reject", text="   ").status_code == 400
    assert answer(client, ev, action="reject").status_code == 400
    assert answer(client, ev, action="reject", text="x" * 501).status_code == 422
    assert answer(client, ev, action="maybe").status_code == 422
    assert client.post("/api/decision", json={"prompt_id": ev["prompt_id"], "action": "approve", "option": 1, "extra": 1}).status_code == 422
    assert client.get("/api/state").json()["status"] == "running"  # none of that disturbed the run
    assert answer(client, ev, action="approve", option=1).status_code == 202  # and the real answer still works


def test_a_stale_or_unknown_question_is_refused(session):
    client, ws, ev = session
    assert client.post("/api/decision", json={"prompt_id": "nope", "action": "approve", "option": 1}).status_code == 409
    answer(client, ev, action="reject", text="avoid gate A1")
    _, again = until(ws, ("options",))
    assert answer(client, ev, action="approve", option=1).status_code == 409  # the first question is closed
    assert answer(client, again, action="approve", option=1).status_code == 202


def test_a_question_can_only_be_answered_once(session):
    client, ws, ev = session
    assert answer(client, ev, action="approve", option=1).status_code == 202
    second = answer(client, ev, action="approve", option=2)
    assert second.status_code == 409  # whether the worker has moved on yet or not
    _, fin = finished(ws)
    assert fin["option"] == 1


def test_no_question_open_means_409(client):
    assert client.post("/api/decision", json={"prompt_id": "x", "action": "approve", "option": 1}).status_code == 409


def test_a_run_without_a_controller_never_asks(client):
    with client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        assert client.post("/api/run", json={"step_delay_s": 0}).status_code == 202
        seen, fin = finished(ws)
    assert fin["outcome"] == "CONSENSUS" and fin["controller"] == "off" and "option" not in fin
    assert not [e for e in seen if e["type"] == "options"]


# --- time and late joiners -----------------------------------------------------------------------------------------------------


def test_no_answer_in_time_applies_the_fallback(cfg, tmp_path):
    cfg.limits.decision_timeout_s = 1
    with TestClient(create_app(cfg, scenario_path=tmp_path / "none.json")) as client, client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        start(client)
        _, ev = until(ws, ("options",))
        began = time.monotonic()
        seen, fin = finished(ws)
        waited = time.monotonic() - began
    assert ev["timeout_s"] == 1 and 0.5 <= waited < 6
    assert fin["outcome"] == "TIMEOUT_FALLBACK" and "did not decide within 1 s" in fin["reason"] and fin["proposal"] is not None
    assert [e["type"] for e in seen].count("prompt_closed") == 1


def test_a_browser_that_joins_mid_question_sees_it_with_the_time_left(cfg, tmp_path):
    cfg.limits.decision_timeout_s = 30
    with TestClient(create_app(cfg, scenario_path=tmp_path / "none.json")) as client:
        with client.websocket_connect("/ws") as first:
            drain_snapshot(first)
            start(client)
            _, ev = until(first, ("options",))
            time.sleep(0.6)
            with client.websocket_connect("/ws") as late:
                seen, joined = until(late, ("options",))
            assert joined["prompt_id"] == ev["prompt_id"] and joined["remaining_s"] < ev["remaining_s"]
            assert [e["type"] for e in seen][0] == "run_started" and any(e["type"] == "entry" for e in seen)
            assert [e["type"] for e in seen].count("options") == 1  # shown once, not once stale and once fresh
            answer(client, ev, action="approve", option=1)
            finished(first)
        with client.websocket_connect("/ws") as after:  # once it is over, a replay shows the question closed and the result
            seen, fin = finished(after)
        kinds = [e["type"] for e in seen]
        assert kinds.count("options") == 1 and kinds.index("prompt_closed") < kinds.index("run_finished")


# --- the log and the live-mode parser ----------------------------------------------------------------------------------------------------


def test_the_downloaded_log_has_the_controller_steps_and_still_verifies(session):
    client, ws, ev = session
    answer(client, ev, action="reject", text="avoid gate A1")
    _, again = until(ws, ("options",))
    answer(client, again, action="approve", option=1)
    _, fin = finished(ws)
    log = ConsensusLog([LogEntry.model_validate(d) for d in client.get("/api/log").json()])
    names = [e.event for e in log.entries]
    assert log.verify() and log.entries[-1].hash == fin["head_hash"]
    assert names.count("OPTIONS") == 2 and names.count("DECISION") == 2 and "CONSTRAINTS" in names and "RENEGOTIATE" in names and names[-1] == "FINAL"
    assert {e.attempt for e in log.entries} == {1, 2}


def test_live_mode_uses_the_parser_the_factory_provides(cfg, tmp_path):
    cfg.server.allow_live = True
    cfg.limits.decision_timeout_s = 2  # nobody answers the second question, so let it time out quickly
    seen = {}

    class Recording:
        name = "recording"

        def parse(self, text, known):
            seen["text"], seen["gates"] = text, known.gates
            return ParseResult((), "I only record.", "recording")

    def parser_factory(mode, llm, c):
        seen["mode"] = mode
        return Recording()

    app = create_app(cfg, llm_factory=lambda mode, personas, c: FakeLLM(), scenario_path=tmp_path / "none.json", parser_factory=parser_factory)
    with TestClient(app) as client, client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        start(client, mode="live")
        _, ev = until(ws, ("options",))
        answer(client, ev, action="reject", text="hello there")
        _, again = until(ws, ("options",))
    assert seen["mode"] == "live" and seen["text"] == "hello there" and seen["gates"]
    assert again["problem"] == "I only record." and again["attempt"] == 1


def test_state_reports_the_controller_limits(client):
    s = client.get("/api/state").json()
    assert s["max_human_rejections"] == 2 and s["decision_timeout_s"] == 60
