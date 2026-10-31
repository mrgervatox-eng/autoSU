"""The negotiation state machine. Python owns the loop; LLMs are only actors inside it.

Per round (at most limits.max_rounds, hard cap):
  1. ATC PROPOSEs. The validator reviews it. An invalid proposal never reaches the others;
     ATC gets the violations as feedback next round.
  2. GROUND and PILOT each ACCEPT / REJECT / COUNTER. Every ACCEPT is re-checked by the
     validator and can be vetoed.
  3. Both accepted (and not vetoed) -> CONSENSUS. Otherwise the valid counters go to ATC.
No consensus after the last round -> deterministic fallback.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from .constraints import ConstraintContext
from .fallback import FALLBACK_LABEL, build_fallback, fallback_message
from .log import ConsensusLog
from .scenario import Scenario
from .schemas import Message, MessageType, Proposal, Role, parse_message
from .validator import check_proposal, review_message

RESPONDERS = (Role.GROUND, Role.PILOT)  # order of replies within a round
ALLOWED = {
    Role.ATC: {MessageType.PROPOSE},
    Role.PILOT: {MessageType.ACCEPT, MessageType.REJECT, MessageType.COUNTER},
    Role.GROUND: {MessageType.ACCEPT, MessageType.REJECT, MessageType.COUNTER},
}


class LLM(Protocol):
    def complete(self, role: Role, view: dict[str, Any]) -> str:
        """Return the agent's JSON reply as a string. One call per role, never combined."""


class Outcome(str, Enum):
    CONSENSUS = "CONSENSUS"
    FALLBACK = "FALLBACK"
    ESCALATED = "ESCALATED"  # not even the fallback is safe: a human must decide


@dataclass
class NegotiationResult:
    outcome: Outcome
    proposal: Proposal | None
    rounds_used: int
    reason: str
    log: ConsensusLog


# --- what each role is allowed to see ------------------------------------


def _runways(ctx: ConstraintContext) -> list[dict]:
    return [{"runway": o.runway, "standing_water_mm": o.standing_water_mm, "confidence": o.provenance.confidence,
             "human_confirmed": o.human_confirmed} for o in ctx.runways]


def _areas(ctx: ConstraintContext) -> list[dict]:
    return [{"area": o.area, "index": o.index, "confidence": o.provenance.confidence,
             "human_confirmed": o.human_confirmed} for o in ctx.turbulence]


def build_view(role: Role, scenario: Scenario, round_: int, table: Proposal | None, own_last: Proposal | None,
               counters: list[dict], objections: list[dict], feedback: list[dict]) -> dict[str, Any]:
    ctx, m, seed = scenario.context, scenario.context.margins, scenario.seed
    base: dict[str, Any] = {
        "role": role.value, "round": round_, "max_rounds": ctx.limits.max_rounds,
        # the flight is inbound: ground delay is absorbed at the origin, holds and runway/gate/separation are at the destination
        "flight": {"flight": seed.label, "origin": seed.origin, "destination": seed.destination,
                   "direction": f"inbound to {seed.destination}"},
    }
    if role == Role.ATC:
        return base | {
            "disruption": scenario.disruption.description,
            "margins": {"min_separation_sec": m.min_separation_sec, "max_hold_min": m.max_hold_min,
                        "max_turbulence_index": m.max_turbulence_index},
            "airfield": {"runways": _runways(ctx), "areas": _areas(ctx), "gates": list(ctx.aircraft.available_gates)},
            "own_last_proposal": own_last.model_dump() if own_last else None,
            "counters": counters,
            "objections": objections,
            "feedback": feedback,
            "hard_constraints": [h.model_dump() for h in ctx.hard_constraints],
        }
    base |= {"table": table.model_dump(), "proposal_id": table.proposal_id}
    if role == Role.PILOT:
        a = ctx.aircraft
        return base | {
            "aircraft": {"fuel_on_board_kg": a.fuel_on_board_kg, "trip_fuel_kg": a.trip_fuel_kg,
                         "hold_burn_kg_per_min": a.hold_burn_kg_per_min, "crew_duty_used_min": a.crew_duty_used_min,
                         "flight_time_remaining_min": a.flight_time_remaining_min},
            "margins": {"fuel_reserve_kg": m.fuel_reserve_kg, "max_hold_min": m.max_hold_min,
                        "max_duty_min": m.max_duty_min, "max_turbulence_index": m.max_turbulence_index},
            "turbulence": _areas(ctx),
        }
    return base | {"runways": _runways(ctx), "gates": list(ctx.aircraft.available_gates),
                   "margins": {"max_standing_water_mm": m.max_standing_water_mm}}


# --- the loop -------------------------------------------------------------


class Negotiation:
    def __init__(self, scenario: Scenario, llm: LLM, log: ConsensusLog | None = None) -> None:
        self.scenario = scenario
        self.ctx = scenario.context
        self.llm = llm
        self.log = log if log is not None else ConsensusLog()

    # one agent call -> a parsed, role/round-checked Message, or None (already logged)
    def _ask(self, role: Role, round_: int, view: dict[str, Any]) -> Message | None:
        try:
            raw = self.llm.complete(role, view)
        except Exception as exc:  # noqa: BLE001 - any agent failure must not crash the loop
            self.log.add(round_, "AGENT_ERROR", role.value, note=f"{type(exc).__name__}: {exc}")
            return None
        msg, err = parse_message(raw)
        if msg is None:
            self.log.add(round_, "INVALID_OUTPUT", role.value, reasons=[err[:300]], note=f"raw: {str(raw)[:120]!r}")
            return None
        problem = None
        if msg.sender != role:
            problem = f"sender {msg.sender.value} does not match role {role.value}"
        elif msg.round != round_:
            problem = f"message round {msg.round} does not match current round {round_}"
        elif msg.type not in ALLOWED[role]:
            problem = f"{role.value} may not send {msg.type.value}"
        if problem:
            self.log.add(round_, "INVALID_OUTPUT", role.value, message=msg, reasons=[problem])
            return None
        return msg

    def run(self) -> NegotiationResult:
        ctx, max_rounds = self.ctx, self.ctx.limits.max_rounds
        own_last: Proposal | None = None
        counters: list[dict] = []
        objections: list[dict] = []
        feedback: list[dict] = []

        for r in range(1, max_rounds + 1):  # bounded: no other loop can outlive this one
            # 1. ATC proposes
            msg = self._ask(Role.ATC, r, build_view(Role.ATC, self.scenario, r, None, own_last, counters, objections, feedback))
            counters, objections, feedback = [], [], []
            if msg is None:  # already logged as INVALID_OUTPUT / AGENT_ERROR
                continue
            verdict = review_message(msg, ctx)
            self.log.add(r, "PROPOSAL", "ATC", verdict.message, ok=verdict.ok, reasons=verdict.reasons)
            table = msg.proposal
            own_last = table
            if not verdict.ok:
                feedback = [{"name": v.name, "actual": v.actual, "limit": v.limit, "detail": v.detail}
                            for v in check_proposal(table, ctx).violations]
                continue

            # 2. responders
            accepted: dict[Role, bool] = {}
            for role in RESPONDERS:
                reply = self._ask(role, r, build_view(role, self.scenario, r, table, own_last, [], [], []))
                if reply is None:
                    accepted[role] = False
                    continue
                v = review_message(reply, ctx, current=table)
                self.log.add(r, "RESPONSE", role.value, v.message, ok=v.ok, vetoed=v.vetoed, reasons=v.reasons)
                if v.vetoed:
                    self.log.add(r, "VALIDATOR_VETO", role.value, v.message, ok=False, vetoed=True, reasons=v.reasons,
                                 note="validator overruled this ACCEPT")
                accepted[role] = reply.type == MessageType.ACCEPT and v.ok
                if reply.type == MessageType.COUNTER:
                    counters.append({"sender": role.value, "valid": v.ok, "proposal": reply.proposal.model_dump()})
                if reply.type in (MessageType.COUNTER, MessageType.REJECT):
                    objections.append({"sender": role.value, "rationale": reply.rationale})

            # 3. consensus?
            if all(accepted.get(role) for role in RESPONDERS):
                final = check_proposal(table, ctx)  # defence in depth: re-check what we are about to adopt
                if final.passed:
                    self.log.add(r, "CONSENSUS", note=f"ATC proposed, GROUND and PILOT accepted, validator confirmed ({table.proposal_id})")
                    return NegotiationResult(Outcome.CONSENSUS, table, r, f"consensus in round {r}", self.log)
                self.log.add(r, "VALIDATOR_VETO", ok=False, vetoed=True, reasons=[v.name for v in final.violations],
                             note="final check failed; consensus refused")

        return self._fallback(max_rounds)

    def _fallback(self, rounds_used: int) -> NegotiationResult:
        result = build_fallback(self.ctx)
        if result is None:
            self.log.add(rounds_used, "ESCALATE", note="no safe fallback exists; human decision required")
            return NegotiationResult(Outcome.ESCALATED, None, rounds_used, "no safe resolution; escalate to controller", self.log)
        proposal, report = result
        self.log.add(rounds_used, "FALLBACK", "ATC", fallback_message(proposal, report, rounds_used), ok=True,
                     note=f"no consensus in {rounds_used} rounds; {FALLBACK_LABEL}")
        return NegotiationResult(Outcome.FALLBACK, proposal, rounds_used,
                                 f"no consensus in {rounds_used} rounds; applied {FALLBACK_LABEL}", self.log)
