"""Dashboard additions: round reasons on the wire, scenario preview, log download, page wording."""
import json

import pytest
from starlette.testclient import TestClient

import negotiator.server as server
from negotiator.config import load_config
from negotiator.log import ConsensusLog, LogEntry
from negotiator.server import STATIC, create_app

from .test_server import cfg, client, collect, drain_snapshot, run_and_collect  # noqa: F401  (shared fixtures and helpers)


# --- round reasons on the wire ---------------------------------------------------------------------


def test_entries_carry_display_text(client):
    _, events = run_and_collect(client)
    entries = [e for e in events if e["type"] == "entry"]
    assert entries and all({"actor_label", "label", "chips", "checks"} <= set(e["view"]) for e in entries)
    first = entries[0]
    assert first["view"]["actor_label"] == "ATC" and first["view"]["label"] == "Propose"
    assert first["view"]["chips"][0].startswith("Ground delay ") and all(set(c) == {"passed", "text"} for c in first["view"]["checks"])


def test_round_summaries_arrive_in_place_and_once_per_round(client):
    _, events = run_and_collect(client)
    entries = [e["entry"] for e in events if e["type"] == "entry"]
    summaries = [e for e in events if e["type"] == "round_summary"]
    assert [s["round"] for s in summaries] == sorted({e["round"] for e in entries})
    for i, ev in enumerate(events):
        if ev["type"] == "round_summary":
            nxt = next(e for e in events[i + 1:] if e["type"] in ("entry", "run_finished"))
            if nxt["type"] == "entry":  # the summary comes right before the first entry that follows its round
                assert nxt["entry"]["round"] > ev["round"] or nxt["entry"]["event"] in ("FALLBACK", "ESCALATE")
    assert [e["type"] for e in events[-2:]] == ["round_summary", "run_finished"]  # consensus: the last round is announced at the end
    fin = events[-1]
    assert [s["round"] for s in fin["round_summaries"]] == [s["round"] for s in summaries]
    assert fin["proposal_chips"][0].startswith("Ground delay ") and summaries[-1]["kind"] == "consensus"


def test_fallback_explains_each_round_and_the_banner_has_the_reasons(client):
    _, events = run_and_collect(client, {"personas": {"PILOT": "stubborn"}})
    fallback_at = next(i for i, e in enumerate(events) if e["type"] == "entry" and e["entry"]["event"] == "FALLBACK")
    assert events[fallback_at - 1]["type"] == "round_summary"  # the last round's reason sits right above the fallback
    fin = events[-1]
    assert fin["outcome"] == "FALLBACK" and len(fin["round_summaries"]) >= 2
    assert all(s["text"].startswith(f"Round {s['round']} ended: ") for s in fin["round_summaries"])
    assert any("Pilot rejected" in s["text"] for s in fin["round_summaries"])


def test_late_joiner_replay_includes_the_round_summaries(client):
    _, live = run_and_collect(client)
    with client.websocket_connect("/ws") as ws:
        replay = collect(ws)
    assert [e["type"] for e in replay] == [e["type"] for e in live]
    assert [e.get("text") for e in replay if e["type"] == "round_summary"] == [e["text"] for e in live if e["type"] == "round_summary"]


# --- preview ------------------------------------------------------------------------------------------------


def test_preview_returns_the_scenario_without_running_anything(client):
    res = client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "weather"})
    assert res.status_code == 200 and "N222BB" in res.json()["scenario_id"] and res.json()["disruption"]["kind"] == "WEATHER"
    other = client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "late_aircraft"}).json()
    assert "N111AA" in other["scenario_id"] and other["disruption"]["kind"] == "LATE_AIRCRAFT"
    assert client.get("/api/state").json()["status"] == "idle"  # looking never starts a run


def test_preview_of_the_saved_scenario(client):
    res = client.get("/api/preview")
    assert res.status_code == 200 and res.json()["scenario_id"] == client.get("/api/state").json()["scenario"]["scenario_id"]


@pytest.mark.parametrize("params, code", [
    ({"source": "csv", "month": 1, "day": 3}, 400),   # no flights that day in the sample file
    ({"source": "csv", "month": 13, "day": 1}, 422), ({"source": "csv", "month": 1, "day": 32}, 422),
    ({"kind": "fog"}, 422), ({"source": "web"}, 422),
])
def test_preview_rejects_bad_requests(client, params, code):
    res = client.get("/api/preview", params=params)
    assert res.status_code == code
    if code == 400:
        assert "no flights" in res.json()["detail"]


def test_preview_and_run_share_one_cached_build(cfg, tmp_path, monkeypatch):
    calls = []
    real = server.build_scenario

    def counting(csv_path, month, day, *a, **k):
        calls.append((month, day))
        return real(csv_path, month, day, *a, **k)

    monkeypatch.setattr(server, "build_scenario", counting)
    with TestClient(create_app(cfg, scenario_path=tmp_path / "none.json")) as client:
        startup = calls.count((1, 2))
        for _ in range(2):
            assert client.get("/api/preview", params={"source": "csv", "month": 1, "day": 2}).status_code == 200
        run_and_collect(client, {"source": "csv", "month": 1, "day": 2})
    assert calls.count((1, 2)) - startup == 1  # built once, then reused by the second preview and by the run


def test_preview_cache_is_bounded(cfg, tmp_path, monkeypatch):
    monkeypatch.setattr(server, "PREVIEW_CACHE_SIZE", 2)
    builds = []
    real = server.build_scenario

    def counting(csv_path, month, day, *a, **k):
        builds.append((month, day))
        return real(csv_path, month, day, *a, **k)

    monkeypatch.setattr(server, "build_scenario", counting)
    with TestClient(create_app(cfg, scenario_path=tmp_path / "none.json")) as client:
        for kind in ("weather", "late_aircraft"):
            client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": kind})
        client.get("/api/preview", params={"source": "csv", "month": 1, "day": 2})  # evicts the oldest
        before = len(builds)
        client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "weather"})  # was evicted: rebuilt
        client.get("/api/preview", params={"source": "csv", "month": 1, "day": 2})                      # still cached
    assert len(builds) - before == 1


# --- log download ----------------------------------------------------------------------------------------------


def test_log_download_is_404_before_any_run(client):
    assert client.get("/api/log").status_code == 404


def test_log_download_matches_the_run_and_still_verifies(client):
    run_id, events = run_and_collect(client)
    res = client.get("/api/log")
    assert res.status_code == 200 and res.headers["content-type"].startswith("application/json")
    assert f"consensus_log_{run_id}.json" in res.headers["content-disposition"]
    log = ConsensusLog([LogEntry.model_validate(d) for d in res.json()])
    assert log.verify() and log.entries[-1].hash == events[-1]["head_hash"] and len(log.entries) == events[-1]["entries"]


# --- page wording ------------------------------------------------------------------------------------------------


def test_page_has_the_requested_wording_and_no_connection_banner():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "live feed connected" not in html.lower()
    assert 'id="liveBadge"' in html and "Live: real Claude agents" in html
    for text in ("Weather delay", "Late-arriving aircraft", "Either type", "Audit log intact", "What is this?", "Download log (JSON)",
                 "Why the fallback applied", "Why a human must decide", "Spare for a hold", "Room for delay and hold"):
        assert text in html, text
    assert "innerHTML" not in html


def test_tile_labels_do_not_share_a_class_with_the_legend_swatches():
    """Regression: a label and a 10px swatch both called 'k' made the tile labels wrap one word per line."""
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "class: 'lab'" in html and ".tile .lab" in html and ".tile .k " not in html


def test_meters_do_not_reuse_the_role_colours():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    roles = {"--atc", "--pilot", "--ground"}
    for line in html.splitlines():
        if ".s.a" in line or ".s.b" in line or ".k.a" in line:
            assert not any(f"var({r})" in line for r in roles), line
    assert "--seg-a: #6250d6" in html and "--seg-b: #008300" in html  # the documented, adjacent slot-7 and slot-6 pair
