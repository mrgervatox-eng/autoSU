"""The ranked options shown to the controller after the agents finish.

Rule for the ranking, shown to the controller too:
  Option 1 is what the agents agreed (or the conservative default procedure, if they did not agree).
  Option 2 is the best valid alternative: first "take the hold on the ground instead", otherwise the
  conservative default procedure.
Every option is re-checked by the validator before it is shown; an option that fails is dropped.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .constraints import ConstraintContext
from .engine import NegotiationResult, Outcome
from .fallback import build_fallback
from .schemas import Proposal
from .summary import area_name, diff_proposal, num, proposal_chips
from .validator import check_proposal


@dataclass(frozen=True)
class Option:
    rank: int
    id: str  # the proposal's id
    title: str
    reason: str
    proposal: Proposal
    metrics: dict[str, float | int | None]

    def as_dict(self) -> dict[str, Any]:
        return {"rank": self.rank, "id": self.id, "title": self.title, "reason": self.reason,
                "proposal": self.proposal.model_dump(), "plain": plain_plan(self.proposal), "chips": proposal_chips(self.proposal),
                "metrics": self.metrics, "metric_lines": metric_lines(self.metrics)}


def plain_plan(p: Proposal) -> str:
    """The plan as one sentence a controller can read at a glance."""
    area = area_name(p.holding_area) or "holding"
    if p.delay_min and p.hold_minutes:
        timing = f"Keep the flight on the ground for {p.delay_min} minutes, then hold for {p.hold_minutes} minutes in the {area} area"
    elif p.delay_min:
        timing = f"Keep the flight on the ground for {p.delay_min} minutes, with no airborne hold"
    elif p.hold_minutes:
        timing = f"No ground delay; hold for {p.hold_minutes} minutes in the {area} area"
    else:
        timing = "No ground delay and no airborne hold"
    landing = f"land on runway {p.runway}" if p.runway else "no runway assigned"
    parking = f"park at gate {p.gate}" if p.gate else "no gate assigned"
    return f"{timing}; {landing} and {parking}, keeping {p.separation_sec} seconds behind the aircraft ahead."


def metrics_for(p: Proposal, ctx: ConstraintContext) -> dict[str, float | int | None]:
    """How much room a proposal leaves, using the same worst-case readings the validator uses."""
    a, m = ctx.aircraft, ctx.margins
    water = [o.standing_water_mm for o in ctx.runways if o.runway == p.runway]
    rough = [o.index for o in ctx.turbulence if o.area == p.holding_area]
    fuel_left = a.fuel_on_board_kg - a.hold_burn_kg_per_min * p.hold_minutes
    return {
        "total_delay_min": p.delay_min + p.hold_minutes,
        "fuel_margin_kg": round(fuel_left - (a.trip_fuel_kg + m.fuel_reserve_kg), 1),
        "duty_margin_min": m.max_duty_min - (a.crew_duty_used_min + p.delay_min + p.hold_minutes + a.flight_time_remaining_min),
        "runway_margin_mm": round(m.max_standing_water_mm - max(water), 2) if water else None,
        "turbulence_margin": round(m.max_turbulence_index - max(rough), 2) if p.hold_minutes > 0 and rough else None,
    }


def metric_lines(metrics: dict[str, float | int | None]) -> list[str]:
    lines = [f"Total delay absorbed: {metrics['total_delay_min']} min",
             f"Fuel above the required minimum: {num(metrics['fuel_margin_kg'])} kg",
             f"Crew-duty room left: {metrics['duty_margin_min']} min"]
    if metrics["runway_margin_mm"] is not None:
        lines.append(f"Runway water under the limit by: {num(metrics['runway_margin_mm'])} mm")
    lines.append("No airborne hold" if metrics["turbulence_margin"] is None else f"Turbulence under the limit by: {num(metrics['turbulence_margin'])}")
    return lines


def _compare(base: Proposal, alt: Proposal, base_m: dict, alt_m: dict) -> str:
    parts = diff_proposal(base, alt)
    fuel = alt_m["fuel_margin_kg"] - base_m["fuel_margin_kg"]
    duty = alt_m["duty_margin_min"] - base_m["duty_margin_min"]
    if fuel:
        parts.append(f"{fuel:+g} kg fuel margin")
    if duty:
        parts.append(f"{duty:+d} min crew-duty room")
    return "Compared with option 1: " + ("; ".join(parts) if parts else "no numeric difference") + "."


def _alternatives(p: Proposal, ctx: ConstraintContext, fallback_is_first: bool) -> list[tuple[Proposal, str]]:
    out: list[tuple[Proposal, str]] = []
    if p.hold_minutes:  # same total delay, but all of it on the ground: no airborne hold, no turbulence exposure, less fuel burned
        out.append((p.model_copy(update={"hold_minutes": 0, "holding_area": None, "delay_min": p.delay_min + p.hold_minutes}),
                    "Take the hold on the ground instead"))
    if not fallback_is_first:
        fb = build_fallback(ctx)
        if fb is not None:
            out.append((fb[0], "Our conservative default procedure"))
    return out


def build_options(result: NegotiationResult, ctx: ConstraintContext) -> list[Option]:
    """One or two validator-approved options, ranked. `result` must carry a proposal (consensus or fallback)."""
    p = result.proposal
    if p is None:
        raise ValueError("nothing to recommend: the negotiation produced no proposal")
    agreed = result.outcome == Outcome.CONSENSUS
    base_m = metrics_for(p, ctx)
    options = [Option(
        1, p.proposal_id,
        "Recommended: what the agents agreed" if agreed else "Recommended: our conservative default procedure",
        "Ground and Pilot accepted it, and the validator confirmed it." if agreed
        else "The agents did not agree within the round limit, so the fixed conservative procedure applies. It passed every validator check.",
        p, base_m)]
    for cand, title in _alternatives(p, ctx, fallback_is_first=not agreed):
        if cand.proposal_id == p.proposal_id or not check_proposal(cand, ctx).passed:
            continue
        cand_m = metrics_for(cand, ctx)
        options.append(Option(2, cand.proposal_id, f"Alternative: {title.lower()}", _compare(p, cand, base_m, cand_m), cand, cand_m))
        break
    return options
