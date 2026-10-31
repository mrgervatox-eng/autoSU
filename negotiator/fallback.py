"""Deterministic non-AI fallback: "our conservative default procedure".

Conservative hold + reassignment, built only from the validator's own context and re-checked by it.
This is our own simple rule, not a real-world procedure.
"""
from __future__ import annotations

from itertools import product

from .constraints import ConstraintContext
from .schemas import Message, MessageType, Proposal, Provenance, Role, Source
from .validator import CheckReport, check_proposal

FALLBACK_LABEL = "our conservative default procedure"
FALLBACK_HOLD_MIN = 10
FALLBACK_FUEL_BUFFER_KG = 300.0
FALLBACK_EXTRA_SEPARATION_SEC = 60


def build_fallback(ctx: ConstraintContext) -> tuple[Proposal, CheckReport] | None:
    """Return a validator-approved conservative proposal, or None if no safe resolution exists."""
    m, a = ctx.margins, ctx.aircraft

    # Largest hold up to FALLBACK_HOLD_MIN that keeps an extra fuel buffer on top of the validator's reserve.
    spare_kg = a.fuel_on_board_kg - a.trip_fuel_kg - m.fuel_reserve_kg - FALLBACK_FUEL_BUFFER_KG
    hold = max(0, min(FALLBACK_HOLD_MIN, m.max_hold_min, int(spare_kg // a.hold_burn_kg_per_min)))

    # The preferred choice comes first in every list (the hold, no ground delay, extra separation, the driest runway,
    # the first gate, the calmest area). Values the controller's rules point at come after them, so a rule such as
    # "avoid gate A1" or "delay at least 30" is met by searching, not by luck. There is no filter of our own here: the
    # validator judges every combination (unconfirmed readings, wet runways, rules...) and the first one it passes wins.
    holds = _candidates(ctx, "hold_minutes", [hold, 0], 1440)
    delays = _candidates(ctx, "delay_min", [0], 1440)
    seps = _candidates(ctx, "separation_sec", [min(m.min_separation_sec + FALLBACK_EXTRA_SEPARATION_SEC, 3600), m.min_separation_sec], 3600)
    gates = sorted(a.available_gates)
    runways = _by_worst(ctx.runways, lambda o: o.runway, lambda o: o.standing_water_mm)  # driest first
    areas = _by_worst(ctx.turbulence, lambda o: o.area, lambda o: o.index)               # calmest first
    for h, d, s, rw, g in product(holds, delays, seps, runways, gates):
        for area in (areas if h else [None]):
            proposal = Proposal(delay_min=d, hold_minutes=h, separation_sec=s, holding_area=area, gate=g, runway=rw)
            report = check_proposal(proposal, ctx)
            if report.passed:
                return proposal, report
    return None  # nothing safe to reassign to: a human must decide


def _by_worst(readings, name, value) -> list[str]:
    """Names ordered best first, judging each by its worst reading (the same worst case the validator uses)."""
    worst: dict[str, float] = {}
    for o in readings:
        worst[name(o)] = max(worst.get(name(o), value(o)), value(o))
    return sorted(worst, key=lambda k: (worst[k], k))


def _candidates(ctx: ConstraintContext, field: str, preferred: list[int], hi: int) -> list[int]:
    """The preferred values, then the numbers the controller's rules mention (and one above, for 'not equal to')."""
    values = list(preferred)
    for rule in ctx.hard_constraints:
        if rule.field == field and isinstance(rule.value, (int, float)):
            values += [rule.value, rule.value + 1]
    return [v for v in dict.fromkeys(int(v) for v in values) if 0 <= v <= hi]


def fallback_message(proposal: Proposal, report: CheckReport, round_: int) -> Message:
    return Message(
        sender=Role.ATC,
        round=round_,
        type=MessageType.PROPOSE,
        proposal=proposal,
        constraint_check=report.results,
        provenance=Provenance(source=Source.FALLBACK, confidence=1.0, note=FALLBACK_LABEL),
        rationale=f"Negotiation did not converge; applying {FALLBACK_LABEL}.",
    )
