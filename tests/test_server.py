"""Stage 5 tests: REST + WebSocket dashboard, all in fake mode (no network, no browser)."""
import logging
import re

import pytest
from starlette.testclient import TestClient

import negotiator.server as server
from negotiator.config import ServerConfig, load_config
from negotiator.fake_llm import FakeLLM
from negotiator.log import ConsensusLog
from negotiator.server import STATIC, create_app


@pytest.fixture
def cfg():
    c = load_config().model_copy(deep=True)
    c.data.flights_csv = "does-not-exist.csv"  # "csv" source falls back to the bundled fixture, never the 560 MB file
    return c


@pytest.fixture
def client(cfg, tmp_path):
    app = create_app(cfg, scenario_path=tmp_path / "none.json")  # no saved scenario -> fixture weather preview
    with TestClient(app) as c:  # keeps one event loop alive so background runs complete between requests
        yield c


def collect(ws, limit=200):
    """Read events until the run ends."""
    events = []
    for _ in range(limit):
        ev = ws.receive_json()
        events.append(ev)
        if ev["type"] in ("run_finished", "run_error"):
            return events
    raise AssertionError("run never finished")


def drain_snapshot(ws):
    """Consume the replay a new connection gets (idle preview, or a whole earlier run) so only new events remain."""
    ev = ws.receive_json()
    while ev["type"] not in ("idle", "run_finished", "run_error"):
        ev = ws.receive_json()


def run_and_collect(client, body=None):
    body = {"step_delay_s": 0} | (body or {})
    with client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        res = client.post("/api/run", json=body)
        assert res.status_code == 202, res.text
        return res.json()["run_id"], collect(ws)


# --- static page ---------------------------------------------------------------


def test_index_page_is_self_contained_and_xss_safe(client):
    html = client.get("/").text
    assert "ATC disruption negotiator" in html and "SIMULATED" in html.upper()
    assert "innerHTML" not in html  # agent/LLM text must only ever enter the page via textContent
    assert not re.search(r"(src|href)=[\"']https?://", html) and "<script src" not in html  # works offline
    assert (STATIC / "index.html").read_text(encoding="utf-8") == html


def test_page_marks_status_with_icon_and_label_not_colour_alone():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for label in ("rejected by validator", "vetoed by validator", "within limit", "over limit", "low confidence"):
        assert label in html


# --- state + snapshot ------------------------------------------------------------


def test_state_endpoint(client):
    s = client.get("/api/state").json()
    assert s["status"] == "idle" and s["allow_live"] is False and s["max_rounds"] == 3
    assert set(s["roles"]) == {"ATC", "PILOT", "GROUND"} and "stubborn" in s["personas"]
    assert s["scenario"]["scenario_id"] and s["scenario"]["enrichment_note"]


def test_websocket_starts_with_idle_preview(client):
    with client.websocket_connect("/ws") as ws:
        ev = ws.receive_json()
    assert ev["type"] == "idle" and ev["scenario"]["chain"]


# --- runs ---------------------------------------------------------------------------


def test_fake_run_streams_every_log_entry_in_order(client):
    run_id, events = run_and_collect(client)
    types = [e["type"] for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    entries = [e["entry"] for e in events if e["type"] == "entry"]
    assert [e["seq"] for e in entries] == list(range(1, len(entries) + 1))
    assert all(e["run_id"] == run_id for e in events[1:])
    fin = events[-1]
    assert fin["outcome"] == "CONSENSUS" and fin["log_valid"] is True and fin["entries"] == len(entries)
    assert fin["head_hash"] == entries[-1]["hash"]
    assert fin["proposal"]["proposal_id"] and fin["llm_usage"] is None
    assert events[0]["mode"] == "fake" and events[0]["scenario"]["context"]["aircraft"]["enrichment_label"].startswith("SIMULATED")


def test_late_joiner_gets_full_replay(client):
    _, live = run_and_collect(client)
    with client.websocket_connect("/ws") as ws:
        replay = collect(ws)
    assert [e["type"] for e in replay] == [e["type"] for e in live]
    assert [e.get("entry", {}).get("hash") for e in replay] == [e.get("entry", {}).get("hash") for e in live]


def test_persona_forces_fallback_and_is_reported(client):
    _, events = run_and_collect(client, {"personas": {"pilot": "stubborn"}})
    fin = events[-1]
    assert fin["outcome"] == "FALLBACK" and "our conservative default procedure" in fin["reason"]
    assert any(e["type"] == "entry" and e["entry"]["event"] == "FALLBACK" for e in events)


def test_csv_source_builds_scenario_from_request(client):
    _, events = run_and_collect(client, {"source": "csv", "month": 1, "day": 1, "kind": "weather"})
    assert events[0]["scenario"]["disruption"]["kind"] == "WEATHER"
    assert "N222BB" in events[0]["scenario"]["scenario_id"]


def test_second_run_while_running_is_409_then_allowed(client):
    with client.websocket_connect("/ws") as ws:
        ws.receive_json()
        assert client.post("/api/run", json={"step_delay_s": 0.05}).status_code == 202
        assert client.post("/api/run", json={"step_delay_s": 0}).status_code == 409
        collect(ws)
    assert client.get("/api/state").json()["status"] == "done"
    assert client.post("/api/run", json={"step_delay_s": 0}).status_code == 202  # free again


@pytest.mark.parametrize("body, code", [
    ({"personas": {"PILOT": "sleepy"}}, 400),
    ({"personas": {"CAPTAIN": "stubborn"}}, 400),
    ({"source": "csv", "month": 1, "day": 3}, 400),       # no flights that day in the fixture
    ({"month": 13}, 422), ({"step_delay_s": 99}, 422), ({"surprise": 1}, 422),
])
def test_bad_requests_are_rejected(client, body, code):
    res = client.post("/api/run", json=body)
    assert res.status_code == code
    assert client.get("/api/state").json()["status"] == "idle"  # a rejected request never starts a run


def test_no_flights_error_names_the_problem(client):
    assert "no flights" in client.post("/api/run", json={"source": "csv", "month": 1, "day": 3}).json()["detail"]


# --- live mode guard ---------------------------------------------------------------------


def test_live_runs_are_blocked_by_default(client):
    res = client.post("/api/run", json={"mode": "live"})
    assert res.status_code == 403 and "allow_live" in res.json()["detail"]
    assert client.get("/api/state").json()["status"] == "idle"


def test_live_run_when_enabled_reports_usage(cfg, tmp_path):
    cfg.server.allow_live = True

    class UsageLLM(FakeLLM):
        def usage_summary(self):
            return {"calls": 3, "repairs": 0, "input_tokens": 10, "output_tokens": 5, "models": ["fake"]}

    seen = {}

    def factory(mode, personas, c):
        seen["mode"] = mode
        return UsageLLM()

    with TestClient(create_app(cfg, llm_factory=factory, scenario_path=tmp_path / "none.json")) as client:
        _, events = run_and_collect(client, {"mode": "live"})
    assert seen["mode"] == "live" and events[0]["mode"] == "live"
    assert events[-1]["llm_usage"]["calls"] == 3


def test_live_with_personas_is_rejected(cfg, tmp_path):
    cfg.server.allow_live = True
    with TestClient(create_app(cfg, scenario_path=tmp_path / "none.json")) as client:
        assert client.post("/api/run", json={"mode": "live", "personas": {"PILOT": "stubborn"}}).status_code == 400


def test_live_start_failure_is_a_clean_500(cfg, tmp_path):
    cfg.server.allow_live = True

    def factory(mode, personas, c):
        raise RuntimeError("Could not resolve authentication method")

    with TestClient(create_app(cfg, llm_factory=factory, scenario_path=tmp_path / "none.json")) as client:
        res = client.post("/api/run", json={"mode": "live"})
        assert res.status_code == 500 and "could not start the agents" in res.json()["detail"]
        assert client.get("/api/state").json()["status"] == "idle"


# --- failure inside a run ------------------------------------------------------------------------


def test_crash_inside_a_run_is_reported_and_server_stays_usable(client, monkeypatch):
    class Boom:
        def __init__(self, *a, **k): pass
        def run(self): raise RuntimeError("kaboom")

    monkeypatch.setattr(server, "Negotiation", Boom)
    _, events = run_and_collect(client)
    assert events[-1]["type"] == "run_error" and "kaboom" in events[-1]["message"]
    monkeypatch.undo()
    _, events = run_and_collect(client)
    assert events[-1]["type"] == "run_finished"


# --- supporting pieces -------------------------------------------------------------------------------


def test_log_subscribers_see_entries_and_cannot_break_the_log(caplog):
    log, seen = ConsensusLog(), []
    log.subscribe(seen.append)
    log.subscribe(lambda e: 1 / 0)
    with caplog.at_level(logging.ERROR):
        log.add(1, "ROUND_END", note="x")
        log.add(1, "CONSENSUS")
    assert [e.seq for e in seen] == [1, 2] and log.verify() and len(log.entries) == 2
    assert "subscriber failed" in caplog.text


def test_server_config_defaults_are_local_and_safe():
    s = ServerConfig()
    assert (s.host, s.port, s.allow_live) == ("127.0.0.1", 8000, False)
    assert load_config().server.allow_live is False


def test_main_warns_when_exposed_to_the_network(monkeypatch, capsys):
    import uvicorn

    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: None)
    monkeypatch.setattr(server, "create_app", lambda cfg: None)
    assert server.main(["--host", "0.0.0.0"]) == 0
    assert "exposes this unauthenticated dashboard" in capsys.readouterr().out
    server.main(["--host", "127.0.0.1"])
    assert "exposes" not in capsys.readouterr().out
