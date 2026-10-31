"""Deterministic validator. No LLM involved; it can veto any agent's message.

Agents' self-reported `constraint_check` is never trusted: review_message() overwrites it
with the validator's own results.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .constraints import ConstraintContext, Margins, RunwayObservation, TurbulenceObservation
from .schemas import ConstraintCheck, Message, MessageType, Proposal


@dataclass
class CheckReport:
    results: list[ConstraintCheck]

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.results)

    @property
    def violations(self) -> list[ConstraintCheck]:
        return [r for r in self.results if not r.passed]


@dataclass
class Verdict:
    ok: bool
    vetoed: bool  # True when an ACCEPT was overruled
    reasons: list[str] = field(default_factory=list)
    message: Message | None = None  # copy with validator-authored constraint_check


def observation_usable(obs: TurbulenceObservation | RunwayObservation, margins: Margins) -> bool:
    return obs.human_confirmed or obs.provenance.confidence >= margins.min_observation_confidence


def needs_human_confirm(obs: TurbulenceObservation | RunwayObservation, margins: Margins) -> bool:
    return not observation_usable(obs, margins)


def _check_readings(name: str, target: str, readings: list, value_of, limit: float, margins: Margins) -> ConstraintCheck:
    """Fail closed, worst case over every reading of the target.

    - no reading at all                      -> fail
    - ANY reading over the limit             -> fail (even a low-confidence one: a doubtful bad reading blocks)
    - no reading is usable (all unconfirmed) -> fail, needs a human confirm
    """
    if not readings:
        return ConstraintCheck(name=name, passed=False, limit=limit, detail=f"no observation for {target} (fail closed)")
    worst = max(value_of(o) for o in readings)
    if worst > limit:
        return ConstraintCheck(name=name, passed=False, actual=worst, limit=limit)
    if not any(observation_usable(o, margins) for o in readings):
        return ConstraintCheck(name=name, passed=False, actual=worst, limit=limit, detail="low-confidence observation, needs human confirm")
    return ConstraintCheck(name=name, passed=True, actual=worst, limit=limit)


def _check_turbulence(p: Proposal, ctx: ConstraintContext) -> ConstraintCheck:
    limit = ctx.margins.max_turbulence_index
    name = "turbulence_at_holding_area"
    if p.hold_minutes == 0:
        return ConstraintCheck(name=name, passed=True, limit=limit, detail="no hold requested")
    if p.holding_area is None:
        return ConstraintCheck(name=name, passed=False, limit=limit, detail="holding_area required when hold_minutes > 0")
    readings = [o for o in ctx.turbulence if o.area == p.holding_area]
    return _check_readings(name, f"area {p.holding_area}", readings, lambda o: o.index, limit, ctx.margins)


def _check_runway(p: Proposal, ctx: ConstraintContext) -> ConstraintCheck:
    limit = ctx.margins.max_standing_water_mm
    name = "runway_standing_water"
    if p.runway is None:
        return ConstraintCheck(name=name, passed=True, limit=limit, detail="no runway requested")
    readings = [o for o in ctx.runways if o.runway == p.runway]
    return _check_readings(name, f"runway {p.runway}", readings, lambda o: o.standing_water_mm, limit, ctx.margins)


def _check_assignment(p: Proposal, margins: Margins) -> ConstraintCheck:
    """A plan that assigns no gate or runway is not a resolution, so it cannot reach consensus."""
    name = "assignment_complete"
    if not margins.require_gate_and_runway:
        return ConstraintCheck(name=name, passed=True, detail="not required by policy")
    missing = [f for f in ("gate", "runway") if getattr(p, f) is None]
    return ConstraintCheck(name=name, passed=not missing, actual=("missing " + ", ".join(missing)) if missing else "gate and runway assigned",
                           limit="gate and runway required")


def check_proposal(p: Proposal, ctx: ConstraintContext) -> CheckReport:
    m, a = ctx.margins, ctx.aircraft

    fuel_left = a.fuel_on_board_kg - a.hold_burn_kg_per_min * p.hold_minutes
    fuel_needed = m.fuel_reserve_kg + a.trip_fuel_kg
    duty_total = a.crew_duty_used_min + p.delay_min + p.hold_minutes + a.flight_time_remaining_min

    if p.gate is None:
        gate = ConstraintCheck(name="gate_available", passed=True, detail="no gate requested")
    else:
        gate = ConstraintCheck(name="gate_available", passed=p.gate in a.available_gates, actual=p.gate, limit=",".join(a.available_gates))

    results = [
        ConstraintCheck(name="min_separation_sec", passed=p.separation_sec >= m.min_separation_sec, actual=p.separation_sec, limit=m.min_separation_sec),
        ConstraintCheck(name="max_hold_min", passed=p.hold_minutes <= m.max_hold_min, actual=p.hold_minutes, limit=m.max_hold_min),
        ConstraintCheck(name="fuel_reserve_kg", passed=fuel_left >= fuel_needed, actual=round(fuel_left, 1), limit=round(fuel_needed, 1), detail="fuel after hold must cover trip fuel + reserve"),
        ConstraintCheck(name="crew_duty_min", passed=duty_total <= m.max_duty_min, actual=duty_total, limit=m.max_duty_min),
        gate,
        _check_assignment(p, m),
        _check_turbulence(p, ctx),
        _check_runway(p, ctx),
    ]
    results += [hc.check(p) for hc in ctx.hard_constraints]
    return CheckReport(results)


def review_message(msg: Message, ctx: ConstraintContext, current: Proposal | None = None) -> Verdict:
    """Judge one message. `current` is the proposal on the table (needed for ACCEPT/REJECT)."""
    reasons: list[str] = []

    if msg.round > ctx.limits.max_rounds:
        return Verdict(ok=False, vetoed=msg.type == MessageType.ACCEPT,
                       reasons=[f"round {msg.round} exceeds max_rounds {ctx.limits.max_rounds}"], message=msg)

    if msg.type == MessageType.REJECT:
        if current is not None and msg.proposal_id != current.proposal_id:
            return Verdict(ok=False, vetoed=False, reasons=["REJECT references a proposal that is not on the table"], message=msg)
        return Verdict(ok=True, vetoed=False, message=msg)

    if msg.type == MessageType.ACCEPT:
        if current is None:
            return Verdict(ok=False, vetoed=True, reasons=["ACCEPT with no proposal on the table"], message=msg)
        if msg.proposal_id != current.proposal_id:
            return Verdict(ok=False, vetoed=True,
                           reasons=[f"ACCEPT references {msg.proposal_id}, table has {current.proposal_id}"], message=msg)
        target = current
    else:  # PROPOSE / COUNTER (Message schema guarantees a proposal)
        target = msg.proposal

    report = check_proposal(target, ctx)
    checked = msg.model_copy(update={"constraint_check": report.results})
    if not report.passed:
        reasons = [f"{v.name}: actual={v.actual} limit={v.limit} {v.detail}".strip() for v in report.violations]
    return Verdict(ok=report.passed, vetoed=(not report.passed and msg.type == MessageType.ACCEPT), reasons=reasons, message=checked)
