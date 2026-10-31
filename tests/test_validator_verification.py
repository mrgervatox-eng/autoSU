"""Verifying the validator itself (not just exercising it).

Four independent lines of evidence:
  1. ORACLE      a second, separately written statement of the rules, compared on thousands of random cases
  2. BOUNDARIES  every limit is exact: at the limit passes, one unit beyond fails
  3. PROPERTIES  tightening a margin never turns a failure into a pass (and relaxing the reverse),
                 a strictly safer proposal never regresses, missing data always fails closed
  4. SYSTEM      random scenarios x misbehaving agents: no accepted result ever violates a margin

All randomness is seeded. Set FUZZ_SCALE=0.2 for a quick run (the mutation checker does).
"""
import os
import random
from collections import Counter

import pytest
from pydantic import ValidationError

from negotiator.config import load_config
from negotiator.constraints import (
    AircraftState, ConstraintContext, HardConstraint, Limits, Margins, RunwayObservation, TurbulenceObservation,
)
from negotiator.engine import Negotiation, Outcome
from negotiator.fake_llm import PERSONAS, FakeLLM
from negotiator.scenario import FIXTURE_CSV, build_scenario
from negotiator.schemas import Proposal, Provenance, Role, Source
from negotiator.validator import check_proposal

from .conftest import prov

SCALE = float(os.environ.get("FUZZ_SCALE", "1"))
AREAS, RUNWAYS, GATES = ["NORTH", "SOUTH", "WEST"], ["27L", "09R", "18C"], ["A1", "A2", "B1", "B2", "C1"]


def n(base: int) -> int:
    return max(25, int(base * SCALE))


# ---------------------------------------------------------------------------
# random generators
# ---------------------------------------------------------------------------


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def random_margins(rng: random.Random) -> Margins:
    return Margins(
        min_separation_sec=rng.randint(30, 300), fuel_reserve_kg=float(rng.randint(0, 3000)),
        max_hold_min=rng.randint(0, 60), max_duty_min=rng.randint(300, 900),
        max_turbulence_index=rng.choice([0.2, 0.35, 0.5, 0.6, 0.8, 1.0]),
        max_standing_water_mm=rng.choice([0.0, 1.0, 2.5, 3.0, 5.0]),
        min_observation_confidence=rng.choice([0.0, 0.3, 0.6, 0.9, 1.0]),
        require_gate_and_runway=rng.random() < 0.7,
    )


def random_provenance(rng, m: Margins) -> Provenance:
    conf = rng.choice([m.min_observation_confidence, clamp(m.min_observation_confidence - 0.01, 0, 1),
                       clamp(m.min_observation_confidence + 0.01, 0, 1), rng.random(), 1.0])
    return Provenance(source=Source.SIMULATED, confidence=conf)


def random_hard(rng) -> HardConstraint:
    field = rng.choice(["delay_min", "hold_minutes", "separation_sec", "holding_area", "gate", "runway"])
    pool = {"holding_area": AREAS, "gate": GATES, "runway": RUNWAYS}.get(field)
    if pool and rng.random() < 0.85:
        op = rng.choice(["==", "!=", "in", "not_in"])
        value = rng.sample(pool, rng.randint(1, 2)) if op in ("in", "not_in") else rng.choice(pool)
    else:
        op = rng.choice(["<=", ">=", "==", "!="])
        value = rng.randint(0, 150)
    return HardConstraint(field=field, op=op, value=value)


def random_context(rng: random.Random, max_rounds: int | None = None, benign: bool = False) -> ConstraintContext:
    """Hostile by default (missing data, doubtful readings, limits sitting right on the numbers).
    benign=True makes a mostly workable situation, so negotiations can actually reach consensus."""
    m = random_margins(rng)
    trip = float(rng.randint(500, 9000))
    burn = float(rng.randint(10, 80))
    flight = rng.randint(30, 200 if benign else 500)

    if benign:
        fuel = trip + m.fuel_reserve_kg + rng.randint(800, 3000)
        duty_used = rng.randint(0, max(0, m.max_duty_min - flight - 150))
        gates = rng.sample(GATES, rng.randint(1, 4))
        turb = [TurbulenceObservation(area=a, index=rng.uniform(0, m.max_turbulence_index * 0.9), provenance=Provenance(
            source=Source.SIMULATED, confidence=1.0)) for a in AREAS]
        runs = [RunwayObservation(runway=r, standing_water_mm=rng.uniform(0, m.max_standing_water_mm * 0.9), provenance=Provenance(
            source=Source.SIMULATED, confidence=1.0)) for r in RUNWAYS]
        hard = []
    else:
        fuel = float(max(0, trip + m.fuel_reserve_kg + rng.randint(-1500, 3000)))
        duty_used = rng.randint(0, 700)
        gates = [] if rng.random() < 0.15 else rng.sample(GATES, rng.randint(1, 4))

        def readings(keys, make):
            return [make(key) for key in keys for _ in range(rng.choice([0, 1, 1, 1, 2]))]  # 0 = no data, 2 = duplicate readings

        def idx_near_limit():
            return clamp(rng.choice([m.max_turbulence_index, m.max_turbulence_index - 0.01, m.max_turbulence_index + 0.01, rng.random()]), 0, 1)

        def water_near_limit():
            return max(0.0, rng.choice([m.max_standing_water_mm, m.max_standing_water_mm - 0.1, m.max_standing_water_mm + 0.1, rng.uniform(0, 8)]))

        turb = readings(AREAS, lambda a: TurbulenceObservation(
            area=a, index=idx_near_limit(), provenance=random_provenance(rng, m), human_confirmed=rng.random() < 0.15))
        runs = readings(RUNWAYS, lambda r: RunwayObservation(
            runway=r, standing_water_mm=water_near_limit(), provenance=random_provenance(rng, m), human_confirmed=rng.random() < 0.15))
        hard = [random_hard(rng) for _ in range(rng.choice([0, 0, 1, 2]))]

    return ConstraintContext(
        margins=m,
        limits=Limits(max_rounds=max_rounds or rng.randint(1, 4)),
        aircraft=AircraftState(
            tail_number="NFUZZ", fuel_on_board_kg=fuel, trip_fuel_kg=trip, hold_burn_kg_per_min=burn,
            crew_duty_used_min=duty_used, flight_time_remaining_min=flight, available_gates=gates),
        turbulence=turb, runways=runs, hard_constraints=hard,
    )


def random_proposal(rng: random.Random, ctx: ConstraintContext) -> Proposal:
    m, a = ctx.margins, ctx.aircraft
    spare_hold = int((a.fuel_on_board_kg - a.trip_fuel_kg - m.fuel_reserve_kg) // a.hold_burn_kg_per_min)  # fuel boundary
    hold = clamp(rng.choice([0, m.max_hold_min, m.max_hold_min + 1, spare_hold, spare_hold + 1, rng.randint(0, 60)]), 0, 1440)
    duty_delay = m.max_duty_min - a.crew_duty_used_min - hold - a.flight_time_remaining_min  # crew-duty boundary
    delay = clamp(rng.choice([duty_delay, duty_delay + 1, duty_delay - 1, rng.randint(0, 200)]), 0, 1440)
    sep = clamp(rng.choice([m.min_separation_sec, m.min_separation_sec - 1, m.min_separation_sec + 1, rng.randint(0, 400)]), 0, 3600)
    return Proposal(
        delay_min=delay, hold_minutes=hold, separation_sec=sep,
        holding_area=rng.choice([None] + AREAS), gate=rng.choice([None] + GATES), runway=rng.choice([None] + RUNWAYS),
    )


# ---------------------------------------------------------------------------
# 1. ORACLE: the rules restated independently (from docs/VALIDATOR.md, not from validator.py)
# ---------------------------------------------------------------------------


def oracle(p: Proposal, ctx: ConstraintContext) -> dict[str, bool]:
    m, a = ctx.margins, ctx.aircraft

    def usable(obs):
        return obs.human_confirmed or obs.provenance.confidence >= m.min_observation_confidence

    def readings_ok(readings, value, limit):
        if not readings:
            return False
        if max(value(r) for r in readings) > limit:
            return False
        return any(usable(r) for r in readings)

    out = {
        "min_separation_sec": p.separation_sec >= m.min_separation_sec,
        "max_hold_min": p.hold_minutes <= m.max_hold_min,
        "fuel_reserve_kg": a.fuel_on_board_kg - a.hold_burn_kg_per_min * p.hold_minutes >= m.fuel_reserve_kg + a.trip_fuel_kg,
        "crew_duty_min": a.crew_duty_used_min + p.delay_min + p.hold_minutes + a.flight_time_remaining_min <= m.max_duty_min,
        "gate_available": p.gate is None or p.gate in a.available_gates,
        "assignment_complete": (not m.require_gate_and_runway) or (p.gate is not None and p.runway is not None),
        "turbulence_at_holding_area": p.hold_minutes == 0 or (p.holding_area is not None and readings_ok(
            [o for o in ctx.turbulence if o.area == p.holding_area], lambda o: o.index, m.max_turbulence_index)),
        "runway_standing_water": p.runway is None or readings_ok(
            [o for o in ctx.runways if o.runway == p.runway], lambda o: o.standing_water_mm, m.max_standing_water_mm),
    }
    for hc in ctx.hard_constraints:
        actual = getattr(p, hc.field)
        if hc.op in ("<=", ">="):
            ok = isinstance(actual, int) and (actual <= hc.value if hc.op == "<=" else actual >= hc.value)
        elif hc.op == "==":
            ok = actual == hc.value
        elif hc.op == "!=":
            ok = actual != hc.value
        elif hc.op == "in":
            ok = actual in hc.value
        else:
            ok = actual not in hc.value
        out[f"hard:{hc.field}{hc.op}{hc.value}"] = ok
    return out


def results(p, ctx):
    return {r.name: r.passed for r in check_proposal(p, ctx).results}


def test_validator_agrees_with_independent_oracle_on_random_cases():
    rng = random.Random(20260101)
    seen = Counter()
    for i in range(n(4000)):
        ctx = random_context(rng)
        p = random_proposal(rng, ctx)
        got, want = results(p, ctx), oracle(p, ctx)
        assert got == want, f"case {i}: validator {got} != oracle {want}\n{p}\n{ctx}"
        for name, ok in got.items():
            if not name.startswith("hard:"):
                seen[(name, ok)] += 1
    total = n(4000)
    for name in ["min_separation_sec", "max_hold_min", "fuel_reserve_kg", "crew_duty_min", "gate_available",
                 "assignment_complete", "turbulence_at_holding_area", "runway_standing_water"]:
        for ok in (True, False):  # non-vacuous: every check both passes and fails often
            assert seen[(name, ok)] > 0.03 * total, f"fuzz rarely produces {name}={ok}: {seen[(name, ok)]}/{total}"


# ---------------------------------------------------------------------------
# 2. BOUNDARIES: exact at the limit, fails one unit beyond
# ---------------------------------------------------------------------------
# Base (conftest): good = delay 20, hold 10, sep 150, NORTH, gate A1, runway 27L;
# fuel 6000, trip 3000, burn 40/min, duty used 600, flight 90, NORTH index 0.3 conf 0.9, 27L water 1.0 conf 0.9.


def _set(path, value):
    def apply(ctx):
        obj = ctx
        *parents, leaf = path.split(".")
        for part in parents:
            obj = getattr(obj, part)
        setattr(obj, leaf, value)
    return apply


def _trust_the_other_sensor(kind):
    """Raise the confidence of the sensor a confidence-boundary test is NOT about, so only one check can flip."""
    def apply(ctx):
        for o in (ctx.runways if kind == "runway" else ctx.turbulence):
            o.provenance.confidence = 1.0
    return apply


def _both(*fns):
    def apply(ctx):
        for fn in fns:
            fn(ctx)
    return apply


BOUNDARY = [
    ("min_separation_sec", _set("margins.min_separation_sec", 150), _set("margins.min_separation_sec", 151)),
    ("max_hold_min", _set("margins.max_hold_min", 10), _set("margins.max_hold_min", 9)),
    ("fuel_reserve_kg", _set("aircraft.fuel_on_board_kg", 5200.0), _set("aircraft.fuel_on_board_kg", 5199.0)),  # 5200 - 40*10 = 4800 = 3000 + 1800
    ("crew_duty_min", _set("margins.max_duty_min", 720), _set("margins.max_duty_min", 719)),                      # 600 + 20 + 10 + 90
    ("turbulence_at_holding_area", _set("margins.max_turbulence_index", 0.3), _set("margins.max_turbulence_index", 0.29)),
    ("runway_standing_water", _set("margins.max_standing_water_mm", 1.0), _set("margins.max_standing_water_mm", 0.99)),
    ("turbulence_at_holding_area", _set("margins.min_observation_confidence", 0.9),
     _both(_set("margins.min_observation_confidence", 0.91), _trust_the_other_sensor("runway"))),
    ("runway_standing_water", _set("margins.min_observation_confidence", 0.9),
     _both(_set("margins.min_observation_confidence", 0.91), _trust_the_other_sensor("turbulence"))),
    ("gate_available", _set("aircraft.available_gates", ["A1"]), _set("aircraft.available_gates", ["A2"])),
]


@pytest.mark.parametrize("name, at_limit, beyond", BOUNDARY, ids=[f"{b[0]}#{i}" for i, b in enumerate(BOUNDARY)])
def test_limits_are_exact(ctx, good, name, at_limit, beyond):
    over = ctx.model_copy(deep=True)
    at_limit(ctx)
    beyond(over)
    at = results(good, ctx)
    assert at[name] and all(at.values()), f"at the limit only {name} should matter, and it must pass: {at}"
    past = results(good, over)
    assert not past[name]
    assert [k for k, v in past.items() if not v] == [name], "exactly one check should flip"


def test_base_case_really_sits_on_the_documented_numbers(ctx, good):
    a = ctx.aircraft
    assert (a.fuel_on_board_kg, a.trip_fuel_kg, a.hold_burn_kg_per_min, a.crew_duty_used_min, a.flight_time_remaining_min) == (6000, 3000, 40, 600, 90)
    assert (ctx.margins.fuel_reserve_kg, good.hold_minutes, good.delay_min) == (1800.0, 10, 20)


# ---------------------------------------------------------------------------
# 3. PROPERTIES
# ---------------------------------------------------------------------------

# margin -> (direction that makes it stricter, step)
TIGHTER = {
    "min_separation_sec": (+1, lambda r: r.randint(1, 120)), "fuel_reserve_kg": (+1, lambda r: float(r.randint(1, 2000))),
    "max_hold_min": (-1, lambda r: r.randint(1, 30)), "max_duty_min": (-1, lambda r: r.randint(1, 300)),
    "max_turbulence_index": (-1, lambda r: r.uniform(0.001, 0.3)), "max_standing_water_mm": (-1, lambda r: r.uniform(0.0, 2.0)),
    "min_observation_confidence": (+1, lambda r: r.uniform(0.0, 0.5)),
}
LOWER_BOUND = {"min_separation_sec": 1, "fuel_reserve_kg": 0.0, "max_hold_min": 0, "max_duty_min": 1,
               "max_turbulence_index": 0.001, "max_standing_water_mm": 0.0, "min_observation_confidence": 0.0}
UPPER_BOUND = {"min_separation_sec": 3600, "max_hold_min": 1440, "max_duty_min": 1440, "max_turbulence_index": 1.0, "min_observation_confidence": 1.0}


def moved(ctx, field, sign, step):
    """A copy of ctx with one margin moved stricter (sign=+1) or looser (sign=-1) by `step`."""
    c = ctx.model_copy(deep=True)
    direction, _ = TIGHTER[field]
    value = getattr(c.margins, field) + direction * sign * step
    setattr(c.margins, field, clamp(value, LOWER_BOUND[field], UPPER_BOUND.get(field, float("inf"))))
    return c


def test_tightening_a_margin_never_turns_a_failure_into_a_pass():
    rng = random.Random(7)
    for _ in range(n(2500)):
        ctx, field = random_context(rng), rng.choice(list(TIGHTER))
        p = random_proposal(rng, ctx)
        before = results(p, ctx)
        after = results(p, moved(ctx, field, +1, TIGHTER[field][1](rng)))
        assert all(not after[k] for k, v in before.items() if not v), (field, before, after)


def test_relaxing_a_margin_never_turns_a_pass_into_a_failure():
    rng = random.Random(8)
    for _ in range(n(2500)):
        ctx, field = random_context(rng), rng.choice(list(TIGHTER))
        p = random_proposal(rng, ctx)
        before = results(p, ctx)
        after = results(p, moved(ctx, field, -1, TIGHTER[field][1](rng)))
        assert all(after[k] for k, v in before.items() if v), (field, before, after)


def test_requiring_assignments_is_strictly_tighter():
    rng = random.Random(9)
    for _ in range(n(500)):
        ctx = random_context(rng)
        p = random_proposal(rng, ctx)
        ctx.margins.require_gate_and_runway = False
        loose = results(p, ctx)
        ctx.margins.require_gate_and_runway = True
        strict = results(p, ctx)
        assert all(not strict[k] for k, v in loose.items() if not v)


def test_a_strictly_safer_proposal_never_regresses():
    rng = random.Random(10)
    monotone = ["min_separation_sec", "max_hold_min", "fuel_reserve_kg", "crew_duty_min", "turbulence_at_holding_area"]
    for _ in range(n(2500)):
        ctx = random_context(rng)
        p = random_proposal(rng, ctx)
        safer = p.model_copy(update={
            "separation_sec": min(3600, p.separation_sec + rng.randint(0, 100)),
            "hold_minutes": max(0, p.hold_minutes - rng.randint(0, p.hold_minutes)),
            "delay_min": max(0, p.delay_min - rng.randint(0, p.delay_min)),
        })
        before, after = results(p, ctx), results(safer, ctx)
        assert all(after[k] for k in monotone if before[k]), (p, safer, before, after)


def test_missing_data_always_fails_closed():
    rng = random.Random(11)
    for _ in range(n(1000)):
        ctx = random_context(rng)
        ctx.turbulence, ctx.runways = [], []  # sensors silent
        p = random_proposal(rng, ctx)
        r = results(p, ctx)
        if p.hold_minutes > 0:
            assert not r["turbulence_at_holding_area"]
        if p.runway is not None:
            assert not r["runway_standing_water"]


def test_a_reading_for_the_wrong_area_never_vouches_for_this_one():
    ctx = random_context(random.Random(3))
    ctx.turbulence = [TurbulenceObservation(area="SOUTH", index=0.0, provenance=prov(1.0), human_confirmed=True)]
    p = Proposal(delay_min=0, hold_minutes=5, separation_sec=3600, holding_area="NORTH", gate=None, runway=None)
    assert not results(p, ctx)["turbulence_at_holding_area"]


# --- duplicate / conflicting readings: worst case wins ------------------------------------------


@pytest.mark.parametrize("readings, expect_pass", [
    ([(0.1, 0.9, False), (0.9, 0.9, False)], False),   # a benign reading cannot hide a bad one ...
    ([(0.9, 0.9, False), (0.1, 0.9, False)], False),   # ... in either order
    ([(0.9, 0.2, False), (0.1, 0.9, False)], False),   # a doubtful BAD reading still blocks
    ([(0.1, 0.2, False), (0.2, 0.9, False)], True),    # a doubtful good reading beside a trusted good one is fine
    ([(0.1, 0.2, False), (0.2, 0.3, False)], False),   # nothing trustworthy: needs a human confirm
    ([(0.1, 0.2, True), (0.2, 0.3, False)], True),     # human confirmed one
    ([(0.9, 0.2, True)], False),                       # confirming a bad reading does not make it safe
])
def test_duplicate_turbulence_readings_use_the_worst_case(ctx, good, readings, expect_pass):
    ctx.turbulence = [TurbulenceObservation(area="NORTH", index=i, provenance=prov(conf=c), human_confirmed=h) for i, c, h in readings]
    ctx.margins.min_observation_confidence = 0.6
    assert results(good, ctx)["turbulence_at_holding_area"] is expect_pass


def test_duplicate_runway_readings_use_the_worst_case(ctx, good):
    ctx.runways = [RunwayObservation(runway="27L", standing_water_mm=w, provenance=prov(0.9)) for w in (0.5, 7.0)]
    assert not results(good, ctx)["runway_standing_water"]
    ctx.runways = [RunwayObservation(runway="27L", standing_water_mm=w, provenance=prov(0.9)) for w in (0.5, 1.0)]
    assert results(good, ctx)["runway_standing_water"]


# --- completeness ------------------------------------------------------------------------------------------


@pytest.mark.parametrize("update, ok", [({}, True), ({"gate": None}, False), ({"runway": None}, False), ({"gate": None, "runway": None}, False)])
def test_proposal_must_assign_gate_and_runway(ctx, good, update, ok):
    assert results(good.model_copy(update=update), ctx)["assignment_complete"] is ok


def test_assignment_requirement_can_be_switched_off_by_policy(ctx, good):
    ctx.margins.require_gate_and_runway = False
    assert results(good.model_copy(update={"gate": None, "runway": None}), ctx)["assignment_complete"]


# ---------------------------------------------------------------------------
# parameters are validated, documented, and typo-proof
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls, bad", [
    (Margins, {"min_separation_sec": 0}), (Margins, {"min_separation_sec": 4000}), (Margins, {"fuel_reserve_kg": -1}),
    (Margins, {"max_hold_min": -1}), (Margins, {"max_duty_min": 0}), (Margins, {"max_duty_min": 2000}),
    (Margins, {"max_turbulence_index": 0}), (Margins, {"max_turbulence_index": 1.5}), (Margins, {"max_standing_water_mm": -0.1}),
    (Margins, {"min_observation_confidence": 1.1}), (Margins, {"min_observation_confidence": -0.1}),
    (Margins, {"min_seperation_sec": 120}),                      # typo in a key must not be silently ignored
    (Limits, {"max_rounds": 0}), (Limits, {"max_rounds": 11}), (Limits, {"max_human_rejections": -1}),
    (Limits, {"max_human_rejections": 6}), (Limits, {"decision_timeout_s": 0}),
])
def test_nonsense_parameters_are_rejected(cls, bad):
    with pytest.raises(ValidationError):
        cls(**bad)


def test_extreme_but_legal_parameters_are_accepted():
    Margins(min_separation_sec=1, fuel_reserve_kg=0, max_hold_min=0, max_duty_min=1, max_turbulence_index=1.0,
            max_standing_water_mm=0, min_observation_confidence=0)
    Limits(max_rounds=1, max_human_rejections=0, decision_timeout_s=1)


def test_bad_config_file_fails_loudly(tmp_path):
    bad = tmp_path / "c.toml"
    bad.write_text("[margins]\nmax_hold_min = -5\n")
    with pytest.raises(ValidationError):
        load_config(bad)
    bad.write_text("[limits]\nmax_rounds = 0\n")
    with pytest.raises(ValidationError):
        load_config(bad)
    bad.write_text("[margins]\nmin_seperation_sec = 100\n")
    with pytest.raises(ValidationError):
        load_config(bad)


def test_every_parameter_is_described_and_documented():
    docs = (os.path.join(os.path.dirname(__file__), "..", "docs", "VALIDATOR.md"))
    text = open(docs, encoding="utf-8").read()
    for model in (Margins, Limits):
        for name, field in model.model_fields.items():
            assert field.description, f"{model.__name__}.{name} has no description"
            assert name in text, f"{name} is not documented in docs/VALIDATOR.md"


def test_default_parameters_pass_their_own_sanity_checks():
    m, l = Margins(), Limits()
    assert m.max_hold_min >= 10  # the fallback's 10 minute hold fits
    assert m.min_separation_sec + 60 <= 3600  # the fallback's extra separation fits the Proposal range
    assert l.max_rounds >= 2  # at least one revision is possible


# ---------------------------------------------------------------------------
# 4. SYSTEM invariants: random scenarios x misbehaving agents
# ---------------------------------------------------------------------------


def test_no_accepted_result_ever_violates_a_margin_under_fuzz():
    rng = random.Random(424242)
    base = build_scenario(FIXTURE_CSV, 1, 1, kind="weather")
    outcomes = Counter()
    total = n(400)
    for i in range(total):
        scen = base.model_copy(deep=True)
        scen.context = random_context(rng, benign=rng.random() < 0.5)
        personas = {role.value: rng.choice(PERSONAS) for role in Role if rng.random() < 0.6}
        llm = FakeLLM(personas)
        res = Negotiation(scen, llm).run()
        ctx, max_rounds = scen.context, scen.context.limits.max_rounds
        outcomes[res.outcome] += 1
        tag = f"run {i} personas={personas} outcome={res.outcome}"

        # nothing unsafe is ever adopted, whatever the agents said
        if res.proposal is not None:
            assert check_proposal(res.proposal, ctx).passed, tag
        if res.outcome == Outcome.ESCALATED:
            assert res.proposal is None, tag
        else:
            assert res.proposal is not None, tag

        # bounded
        assert res.rounds_used <= max_rounds, tag
        assert llm.calls.count(Role.ATC) <= max_rounds and len(llm.calls) <= 3 * max_rounds, tag

        # traceable
        assert res.log.verify(), tag

        # consensus is only declared when both responders really accepted the same proposal, unvetoed
        if res.outcome == Outcome.CONSENSUS:
            r = res.log.events("CONSENSUS")[0].round
            accepts = {e.actor for e in res.log.events("RESPONSE")
                       if e.round == r and e.message.type.value == "ACCEPT" and e.ok and not e.vetoed
                       and e.message.proposal_id == res.proposal.proposal_id}
            assert accepts == {"GROUND", "PILOT"}, tag
            assert not [e for e in res.log.events("VALIDATOR_VETO") if e.round == r], tag
    # non-vacuous: every outcome is reached, and consensus (the path with the veto logic) is common
    for outcome in Outcome:
        assert outcomes[outcome] > 0, f"fuzz never produced {outcome}: {outcomes}"
    assert outcomes[Outcome.CONSENSUS] >= 0.10 * total, f"too few consensus runs to trust the invariants: {outcomes}"
