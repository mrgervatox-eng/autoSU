"""Fake mode: a deterministic mock LLM, one behaviour per role. No network.

It returns JSON *strings*, exactly like a real model would, so parsing, validation and the
state machine are exercised for real. The FAKE preferences below (pilot fuel comfort buffer,
ground's drier-runway preference, ATC's naive opening bid) are invented for the demo.

Personas let tests make an agent misbehave:
  yes_man   - accepts anything (correct proposal_id)
  reckless  - accepts without reading: wrong proposal_id
  stubborn  - rejects everything (ATC: keeps proposing an unsafe separation)
  garbage   - replies with prose instead of JSON
  impostor  - responds but claims to be a different role
"""
from __future__ import annotations

import json
from typing import Any

from .schemas import Role

PERSONAS = ("reasonable", "yes_man", "reckless", "stubborn", "garbage", "impostor")

PILOT_OWNS = ("delay_min", "hold_minutes", "holding_area")
GROUND_OWNS = ("runway", "gate")
PILOT_MAX_HOLD_MIN = 15
PILOT_FUEL_BUFFER_KG = 300.0
PILOT_MAX_TURBULENCE = 0.5
GROUND_MAX_WATER_MM = 2.5


def _msg(role: Role, view: dict, type_: str, proposal: dict | None = None, proposal_id: str | None = None,
         why: str = "", sender: Role | None = None) -> dict[str, Any]:
    d: dict[str, Any] = {
        "sender": (sender or role).value,
        "round": view["round"],
        "type": type_,
        "provenance": {"source": "simulated", "confidence": 0.9, "note": "fake LLM"},
        "rationale": why,
    }
    if proposal is not None:
        d["proposal"] = proposal
    if proposal_id is not None:
        d["proposal_id"] = proposal_id
    return d


def _table_id(view: dict) -> str:
    return view["proposal_id"]


# --- ATC ------------------------------------------------------------------


def _atc(view: dict) -> dict[str, Any]:
    m, af = view["margins"], view["airfield"]
    runways = sorted(af["runways"], key=lambda r: r["runway"])
    areas = sorted(af["areas"], key=lambda a: a["area"])
    gates = af["gates"]
    driest = min(runways, key=lambda r: (r["standing_water_mm"], r["runway"]))["runway"]
    calmest = min(areas, key=lambda a: (a["index"], a["area"]))["area"]
    last = view["own_last_proposal"]
    counters = [c for c in view["counters"] if c["valid"]]

    if last is None:
        p = dict(delay_min=10, hold_minutes=min(20, m["max_hold_min"]), separation_sec=m["min_separation_sec"] + 30,
                 holding_area=areas[0]["area"], gate=gates[0], runway=runways[0]["runway"])
        why = "Opening bid: 10 min delay, 20 min hold, first-listed area/runway/gate."
    elif counters:
        p = dict(last)
        for c in counters:
            for f in (PILOT_OWNS if c["sender"] == "PILOT" else GROUND_OWNS):
                p[f] = c["proposal"][f]
        why = "Merged counters: pilot's hold/delay, ground's runway/gate."
    else:
        names = {f["name"] for f in view["feedback"]}
        p = dict(last)
        if "min_separation_sec" in names:
            p["separation_sec"] = m["min_separation_sec"]
        if "runway_standing_water" in names:
            p["runway"] = driest
        if "gate_available" in names:
            p["gate"] = gates[0]
        if names & {"fuel_reserve_kg", "max_hold_min", "crew_duty_min"}:
            p["hold_minutes"], p["delay_min"] = min(p["hold_minutes"], 5), 0
        if "turbulence_at_holding_area" in names:
            p["holding_area"] = calmest
            if min(a["index"] for a in areas) > m.get("max_turbulence_index", 1.0):
                p["hold_minutes"] = 0  # no holding area is calm enough: do not hold at all
        if not names:  # rejected with no usable counter: back off
            p["hold_minutes"] = p["hold_minutes"] // 2
        why = "Revised after objections/validator feedback: " + (", ".join(sorted(names)) or "halved the hold") + "."
    rules = view.get("hard_constraints") or []
    if rules:
        _satisfy(p, rules, runways, areas, gates)
        why += " Respecting the controller's rules."
    if p["hold_minutes"] == 0:
        p["holding_area"] = None
    elif p["holding_area"] is None:
        p["holding_area"] = calmest
    return _msg(Role.ATC, view, "PROPOSE", proposal=p, why=why)


def _holds(x, op: str, v) -> bool:
    return {"==": lambda: x == v, "!=": lambda: x != v, "in": lambda: x in v, "not_in": lambda: x not in v,
            "<=": lambda: x <= v, ">=": lambda: x >= v}[op]()


def _satisfy(p: dict, rules: list[dict], runways: list[dict], areas: list[dict], gates: list[str]) -> None:
    """Bend the proposal to meet the controller's rules where it can. The validator still has the last word."""
    pools = {"gate": list(gates), "runway": [r["runway"] for r in sorted(runways, key=lambda r: r["standing_water_mm"])],
             "holding_area": [a["area"] for a in sorted(areas, key=lambda a: a["index"])]}
    for r in rules:
        f, op, v = r["field"], r["op"], r["value"]
        if f in pools:
            if p[f] is None or not _holds(p[f], op, v):
                pick = next((x for x in pools[f] if _holds(x, op, v)), None)
                if pick is not None:
                    p[f] = pick
        elif not _holds(p[f], op, v):
            p[f] = v if op in ("<=", ">=", "==") else v + 1


# --- PILOT ----------------------------------------------------------------


def _pilot(view: dict) -> dict[str, Any]:
    t, a, m = view["table"], view["aircraft"], view["margins"]
    spare = a["fuel_on_board_kg"] - a["trip_fuel_kg"] - m["fuel_reserve_kg"] - PILOT_FUEL_BUFFER_KG
    max_hold = int(min(PILOT_MAX_HOLD_MIN, max(0, spare // a["hold_burn_kg_per_min"])))
    index = {x["area"]: x["index"] for x in view["turbulence"]}
    calmest = min(view["turbulence"], key=lambda x: (x["index"], x["area"]))["area"]

    c, why = dict(t), []
    if c["hold_minutes"] > max_hold:
        c["hold_minutes"] = max_hold
        why.append(f"hold capped at {max_hold} min to keep a {PILOT_FUEL_BUFFER_KG:.0f} kg fuel buffer")
        if max_hold == 0:
            c["holding_area"] = None
    if c["hold_minutes"] > 0 and index.get(c["holding_area"], 1.0) > PILOT_MAX_TURBULENCE:
        c["holding_area"] = calmest
        why.append(f"moved hold to calmer area {calmest}")
    if c == t:
        return _msg(Role.PILOT, view, "ACCEPT", proposal_id=_table_id(view), why="Fuel and turbulence acceptable to the crew.")
    return _msg(Role.PILOT, view, "COUNTER", proposal=c, why="; ".join(why).capitalize() + ".")


# --- GROUND ---------------------------------------------------------------


def _ground(view: dict) -> dict[str, Any]:
    t = view["table"]
    water = {r["runway"]: r["standing_water_mm"] for r in view["runways"]}
    driest = min(view["runways"], key=lambda r: (r["standing_water_mm"], r["runway"]))["runway"]
    c, why = dict(t), []
    if c["runway"] is not None and water.get(c["runway"], 99.0) > GROUND_MAX_WATER_MM:
        c["runway"] = driest
        why.append(f"runway {t['runway']} too wet, switched to {driest}")
    if c["gate"] is not None and c["gate"] not in view["gates"]:
        c["gate"] = sorted(view["gates"])[0]
        why.append(f"gate {t['gate']} unavailable, reassigned to {c['gate']}")
    if c == t:
        return _msg(Role.GROUND, view, "ACCEPT", proposal_id=_table_id(view), why="Runway and gate are workable.")
    return _msg(Role.GROUND, view, "COUNTER", proposal=c, why="; ".join(why).capitalize() + ".")


# --- the mock client ------------------------------------------------------


class FakeLLM:
    """Implements the LLM protocol: complete(role, view) -> JSON string."""

    def __init__(self, personas: dict[Role | str, str] | None = None) -> None:
        self.personas = {Role(k): v for k, v in (personas or {}).items()}
        for v in self.personas.values():
            if v not in PERSONAS:
                raise ValueError(f"unknown persona {v!r}; choose from {PERSONAS}")
        self.calls: list[Role] = []

    def complete(self, role: Role, view: dict) -> str:
        self.calls.append(role)
        persona = self.personas.get(role, "reasonable")
        if persona == "garbage":
            return "Sure! I think a short hold sounds fine to me, thanks."
        if role == Role.ATC:
            if persona == "stubborn":
                p = dict(delay_min=0, hold_minutes=0, separation_sec=10, holding_area=None, gate=None, runway=None)
                return json.dumps(_msg(role, view, "PROPOSE", proposal=p, why="Separation is fine, go."))
            return json.dumps(_atc(view))
        if persona == "yes_man":
            out = _msg(role, view, "ACCEPT", proposal_id=_table_id(view), why="Sounds good.")
        elif persona == "reckless":
            out = _msg(role, view, "ACCEPT", proposal_id="00000000", why="Accepted without reading.")
        elif persona == "stubborn":
            out = _msg(role, view, "REJECT", proposal_id=_table_id(view), why="No.")
        elif persona == "impostor":
            other = Role.GROUND if role == Role.PILOT else Role.PILOT
            out = _msg(role, view, "ACCEPT", proposal_id=_table_id(view), why="Trust me.", sender=other)
        else:
            out = _pilot(view) if role == Role.PILOT else _ground(view)
        return json.dumps(out)
