"""Deterministic non-AI fallback: "our conservative default procedure".

Conservative hold + reassignment, built only from the validator's own context and re-checked by it.
This is our own simple rule, not a real-world procedure.
"""
from __future__ import annotations

from .constraints import ConstraintContext
from .schemas import Message, MessageType, Proposal, Provenance, Role, Source
from .validator import CheckReport, check_proposal, observation_usable

FALLBACK_LABEL = "our conservative default procedure"
FALLBACK_HOLD_MIN = 10
FALLBACK_FUEL_BUFFER_KG = 300.0
FALLBACK_EXTRA_SEPARATION_SEC = 60


def build_fallback(ctx: ConstraintContext) -> tuple[Proposal, CheckReport] | None:
    """Return a validator-approved conservative proposal, or None if no safe resolution exists."""
    m, a = ctx.margins, ctx.aircraft

    runways = sorted((o for o in ctx.runways if observation_usable(o, m) and o.standing_water_mm <= m.max_standing_water_mm),
                     key=lambda o: (o.standing_water_mm, o.runway))
    areas = sorted((o for o in ctx.turbulence if observation_usable(o, m) and o.index <= m.max_turbulence_index),
                   key=lambda o: (o.index, o.area))

    if not runways or not a.available_gates:
        return None  # nothing safe to reassign to: a human must decide

    # Largest hold up to FALLBACK_HOLD_MIN that keeps an extra fuel buffer on top of the validator's reserve.
    spare_kg = a.fuel_on_board_kg - a.trip_fuel_kg - m.fuel_reserve_kg - FALLBACK_FUEL_BUFFER_KG
    hold = max(0, min(FALLBACK_HOLD_MIN, m.max_hold_min, int(spare_kg // a.hold_burn_kg_per_min)))
    if not areas:
        hold = 0

    base = dict(
        delay_min=0,
        separation_sec=min(m.min_separation_sec + FALLBACK_EXTRA_SEPARATION_SEC, 3600),
        gate=sorted(a.available_gates)[0],
        runway=runways[0].runway,
    )
    for h in dict.fromkeys([hold, 0]):  # try the hold, then no hold
        proposal = Proposal(hold_minutes=h, holding_area=areas[0].area if h and areas else None, **base)
        report = check_proposal(proposal, ctx)
        if report.passed:
            return proposal, report
    return None


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
