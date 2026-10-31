"""Pilot phone: turbulence readings in, live conditions out (fake mode, no phone needed)."""
import time

import pytest
from starlette.testclient import TestClient

import negotiator.server as server
from negotiator.server import STATIC, create_app

from .test_server import cfg, client, collect, drain_snapshot, run_and_collect  # noqa: F401  (shared fixtures and helpers)


def post(client, **over):
    body = {"area": "NORTH", "index": 0.9, "confidence": 0.95} | over
    return client.post("/api/observation/turbulence", json=body)


def turb(scenario, area):
    return next(o for o in scenario["context"]["turbulence"] if o["area"] == area)


def test_a_reading_is_accepted_and_shows_up_as_sensor_data_in_the_preview(client):
    assert post(client).status_code == 202
    s = client.get("/api/preview").json()
    north, south = turb(s, "NORTH"), turb(s, "SOUTH")
    assert north["index"] == 0.9 and north["provenance"]["source"] == "sensor" and north["provenance"]["confidence"] == 0.95
    assert "phone" in north["provenance"]["note"]
    assert south["provenance"]["source"] == "simulated"  # only the area the phone reported is replaced


def test_state_lists_the_areas_and_the_live_readings(client):
    assert client.get("/api/state").json()["areas"] == ["NORTH", "SOUTH"] and client.get("/api/state").json()["live"] == {}
    post(client, area="south", index=0.2, confidence=0.7)  # case does not matter
    live = client.get("/api/state").json()["live"]
    assert set(live) == {"SOUTH"} and live["SOUTH"]["index"] == 0.2 and live["SOUTH"]["age_s"] < 5


def test_the_next_run_uses_the_phone_reading(client):
    post(client, index=0.9, confidence=0.95)
    _, events = run_and_collect(client)
    north = turb(events[0]["scenario"], "NORTH")
    assert north["index"] == 0.9 and north["provenance"]["source"] == "sensor"
    assert events[-1]["type"] == "run_finished"  # and the negotiation still completes with it


def test_a_rough_phone_reading_changes_what_the_validator_allows(client):
    post(client, index=0.95, confidence=0.95)  # NORTH is now too rough to hold in
    _, events = run_and_collect(client)
    entries = [e["entry"] for e in events if e["type"] == "entry"]
    checks = [c["text"] for e in events if e["type"] == "entry" for c in e["view"]["checks"] if "Turbulence" in c["text"] and not c.get("passed", True)]
    assert checks and any("0.95" in t for t in checks)  # the validator used the phone's number
    final = [e for e in events if e["type"] == "run_finished"][0]["proposal"]
    assert final["hold_minutes"] == 0 or final["holding_area"] != "NORTH"


def test_a_low_confidence_reading_is_not_trusted(client):
    post(client, index=0.1, confidence=0.2)  # calm but doubtful: below the 0.6 confidence the validator needs
    s = client.get("/api/preview").json()
    north = turb(s, "NORTH")
    assert north["provenance"]["confidence"] == 0.2 and north["provenance"]["source"] == "sensor"


def test_stale_readings_expire(client, monkeypatch):
    monkeypatch.setattr(server, "LIVE_TTL_S", 0.2)
    post(client)
    assert turb(client.get("/api/preview").json(), "NORTH")["provenance"]["source"] == "sensor"
    time.sleep(0.35)
    assert turb(client.get("/api/preview").json(), "NORTH")["provenance"]["source"] == "simulated"
    assert client.get("/api/state").json()["live"] == {}


def test_a_newer_reading_replaces_the_older_one(client):
    post(client, index=0.9)
    post(client, index=0.1)
    assert turb(client.get("/api/preview").json(), "NORTH")["index"] == 0.1


@pytest.mark.parametrize("over, code", [({"area": "WEST"}, 400), ({"area": ""}, 422), ({"index": 1.1}, 422), ({"index": -0.1}, 422),
                                        ({"confidence": 2}, 422), ({"confidence": -1}, 422), ({"area": "N" * 21}, 422)])
def test_bad_readings_are_refused_and_change_nothing(client, over, code):
    assert post(client, **over).status_code == code
    assert client.get("/api/state").json()["live"] == {}


def test_missing_or_extra_fields_are_refused(client):
    assert client.post("/api/observation/turbulence", json={"area": "NORTH", "index": 0.5}).status_code == 422
    assert client.post("/api/observation/turbulence", json={"area": "NORTH", "index": 0.5, "confidence": 0.9, "lat": 1.0}).status_code == 422  # no GPS
    assert client.post("/api/observation/turbulence", json={"area": "NORTH", "index": "x", "confidence": 0.9}).status_code == 422


def test_browsers_are_told_but_the_run_history_is_not_polluted(client):
    with client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        post(client, index=0.4, confidence=0.9)
        ev = ws.receive_json()
        assert ev == {"type": "observation", "kind": "turbulence", "area": "NORTH", "index": 0.4, "confidence": 0.9, "source": "sensor"}
    with client.websocket_connect("/ws") as late:  # a late joiner does not replay phone chatter
        assert late.receive_json()["type"] == "idle"


def test_readings_never_change_a_run_that_is_already_going(client):
    with client.websocket_connect("/ws") as ws:
        drain_snapshot(ws)
        assert client.post("/api/run", json={"step_delay_s": 0.05}).status_code == 202
        started = ws.receive_json()
        post(client, index=0.99)  # arrives mid-run
        events = collect(ws)
        assert turb(started["scenario"], "NORTH")["provenance"]["source"] == "simulated"  # the scenario this run was started with
        assert events[-1]["type"] == "run_finished"
        assert turb(client.get("/api/preview").json(), "NORTH")["index"] == 0.99  # but the next run will use it


# --- the pages ---------------------------------------------------------------------------------------------


def test_the_pilot_page_is_served(client):
    res = client.get("/pilot")
    assert res.status_code == 200 and "Start sensing" in res.text and "text/html" in res.headers["content-type"]
    assert res.text == (STATIC / "pilot.html").read_text(encoding="utf-8")


def test_the_pilot_page_documents_its_mapping_and_asks_for_the_right_things():
    html = (STATIC / "pilot.html").read_text(encoding="utf-8")
    for needed in ("devicemotion", "requestPermission", "FULL_SCALE = 3.0", "WINDOW_MS = 3000", "min(1, RMS ÷ 3)", "isSecureContext",
                   "/api/observation/turbulence", "NORTH", "SOUTH", "Not started."):
        assert needed in html, needed
    for banned in ("geolocation", "getUserMedia", "innerHTML", "navigator.mediaDevices"):  # no GPS, no camera, no microphone, no HTML injection
        assert banned not in html, banned


def test_the_dashboard_shows_a_phone_badge_and_has_the_rerun_button():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert "'phone'" in html and "Re-run with current conditions" in html and "ev.type === 'observation'" in html
    assert "never wipe a finished run" in html


def test_a_phone_reading_survives_across_the_cached_scenario(client):
    post(client, index=0.8)
    a = client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "weather"}).json()
    b = client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "weather"}).json()
    assert turb(a, "NORTH")["index"] == turb(b, "NORTH")["index"] == 0.8  # the cache holds the base scenario; the reading is laid on top
    post(client, index=0.3)
    assert turb(client.get("/api/preview", params={"source": "csv", "month": 1, "day": 1, "kind": "weather"}).json(), "NORTH")["index"] == 0.3
