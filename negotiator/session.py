"""Stage 6: the human in the loop.

The negotiation engine is unchanged. A Session wraps it:

  negotiate -> show the controller ranked options -> the controller
      approves one option                       -> done (the validator re-checks it first)
      rejects with free text                    -> the text becomes hard rules, the agents negotiate again
      does not answer in time                   -> our conservative default procedure
  After more than `max_human_rejections` rejections                -> our conservative default procedure
  After MAX_UNCLEAR answers that could not be understood           -> our conservative default procedure

Everything is bounded: at most 1 + max_human_rejections negotiations, and a few prompts each. A reply
that cannot be turned into a rule does not use up one of the controller's rejections. The controller's
rules only ever make the validator stricter.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from .constraints import HardConstraint
from .engine import LLM, Negotiation, Outcome
from .fallback import FALLBACK_LABEL, build_fallback, fallback_message
from .log import ConsensusLog
from .options import Option, build_options
from .parsing import ConstraintParser, Known, ParseResult
from .scenario import Scenario
from .schemas import Proposal
from .summary import describe_proposal, quote
from .validator import check_proposal

MAX_UNCLEAR = 3  # answers that cannot be understood, per prompt, before the fallback applies
OUTCOMES = ("APPROVED", "TIMEOUT_FALLBACK", "UNCLEAR_FALLBACK", "REJECTED_FALLBACK", "VETOED_FALLBACK", "ESCALATED")
RANK_RULE = ("Option 1 is what the agents agreed (or the conservative default if they did not agree). "
             "Option 2 is the best valid alternative: the hold taken on the ground instead, otherwise the conservative default. "
             "Every option is re-checked by the validator.")


@dataclass(frozen=True)
class Decision:
    action: Literal["approve", "reject", "timeout"]
    option: int | None = None   # the rank, for approve
    text: str | None = None     # the free text, for reject


class DecisionSource(Protocol):
    def decide(self, options: Sequence[Option], *, attempt: int, rejections_left: int, timeout_s: float,
               rules: Sequence[str], problem: str | None) -> Decision:
        """Block until the controller answers (or time runs out, in which case return Decision('timeout'))."""


@dataclass
class SessionResult:
    outcome: str  # one of OUTCOMES
    proposal: Proposal | None
    option: int | None
    negotiations: int
    rejections: int
    constraints: list[HardConstraint]
    reason: str
    log: ConsensusLog


class Session:
    def __init__(self, scenario: Scenario, llm: LLM, parser: ConstraintParser, decider: DecisionSource,
                 log: ConsensusLog | None = None) -> None:
        self.scenario, self.llm, self.parser, self.decider = scenario, llm, parser, decider
        self.ctx = scenario.context
        self.log = log if log is not None else ConsensusLog()
        self._round = 1  # the last round the agents used; controller entries are stamped with it
        self._rules_added: list[HardConstraint] = []

    # --- the loop ---------------------------------------------------------------------------------

    def run(self) -> SessionResult:
        max_rejections = self.ctx.limits.max_human_rejections
        rejections = 0
        while True:  # bounded: it only continues after a rejection, and rejections are capped just below
            self.log.attempt = rejections + 1
            result = Negotiation(self.scenario, self.llm, self.log).run()
            self._round = max(1, result.rounds_used)
            if result.outcome == Outcome.ESCALATED:
                return self._finish("ESCALATED", None, None, rejections, "No safe resolution exists, so a human must decide.")

            options = build_options(result, self.ctx)
            self._log_options(options, result)
            action, payload = self._ask(options, rejections)

            if action == "approve":
                return self._approve(payload, rejections)
            if action == "timeout":
                return self._fallback("TIMEOUT_FALLBACK", rejections, payload)
            if action == "unclear":
                return self._fallback("UNCLEAR_FALLBACK", rejections, payload)

            rejections += 1  # action == "reject": payload is the understood ParseResult
            self._add_rules(payload.constraints)
            if rejections > max_rejections:
                return self._fallback("REJECTED_FALLBACK", rejections,
                                      f"The controller has rejected {rejections} times and only {max_rejections} renegotiations are allowed.")
            self.log.add(self._round, "RENEGOTIATE", "SYSTEM",
                         note=f"Renegotiation {rejections} of {max_rejections}: the agents start again with the controller's rules added.",
                         data={"rules": [c.describe() for c in self.ctx.hard_constraints], "renegotiation": rejections})

    # --- asking the controller -------------------------------------------------------------------------

    def _ask(self, options: list[Option], rejections: int) -> tuple[str, object]:
        limits = self.ctx.limits
        known = Known.from_context(self.ctx, [o.proposal for o in options])
        problem: str | None = None
        unclear = 0
        while True:  # bounded by MAX_UNCLEAR
            decision = self.decider.decide(
                options, attempt=self.log.attempt, rejections_left=limits.max_human_rejections - rejections,
                timeout_s=float(limits.decision_timeout_s), rules=[c.describe() for c in self.ctx.hard_constraints], problem=problem)

            if decision.action == "timeout":
                self.log.add(self._round, "DECISION", "CONTROLLER", ok=False,
                             note=f"The controller did not decide within {limits.decision_timeout_s} s.", data={"action": "timeout"})
                return "timeout", f"The controller did not decide within {limits.decision_timeout_s} s."

            if decision.action == "approve":
                chosen = next((o for o in options if o.rank == decision.option), None)
                if chosen is not None:
                    self.log.add(self._round, "DECISION", "CONTROLLER", ok=True, note=f"The controller approved option {chosen.rank}.",
                                 data={"action": "approve", "option": chosen.rank, "proposal_id": chosen.id})
                    return "approve", chosen
                problem = f"There is no option {decision.option}. Choose {' or '.join(str(o.rank) for o in options)}."
                self.log.add(self._round, "DECISION", "CONTROLLER", ok=False, note=problem,
                             data={"action": "approve", "option": decision.option})
            elif decision.action == "reject":
                text = (decision.text or "").strip()
                self.log.add(self._round, "DECISION", "CONTROLLER", ok=True, note=f'The controller rejected: "{quote(text)}"',
                             data={"action": "reject", "text": text[:500]})
                parsed: ParseResult = self.parser.parse(text, known)
                self.log.add(self._round, "CONSTRAINTS", "CONTROLLER", ok=parsed.ok, note=parsed.note,
                             data={"parser": parsed.parser, "text": text[:500], "rules": [c.describe() for c in parsed.constraints],
                                   "constraints": [c.model_dump() for c in parsed.constraints]})
                if parsed.ok:
                    return "reject", parsed
                problem = parsed.note  # not understood: this does not use up one of the controller's rejections
            else:
                problem = "Choose an option to approve, or reject with a reason."
                self.log.add(self._round, "DECISION", "CONTROLLER", ok=False, note=problem, data={"action": str(decision.action)})

            unclear += 1
            if unclear >= MAX_UNCLEAR:
                return "unclear", f"The controller's answers could not be understood {MAX_UNCLEAR} times in a row."

    # --- endings -------------------------------------------------------------------------------------------

    def _add_rules(self, rules: Sequence[HardConstraint]) -> None:
        have = {c.label for c in self.ctx.hard_constraints}
        for rule in rules:
            if rule.label not in have:
                self.ctx.hard_constraints.append(rule)
                self._rules_added.append(rule)
                have.add(rule.label)

    def _approve(self, option: Option, rejections: int) -> SessionResult:
        report = check_proposal(option.proposal, self.ctx)  # defence in depth: the world may have changed since the options were built
        if not report.passed:
            names = [v.name for v in report.violations]
            self.log.add(self._round, "VALIDATOR_VETO", "SYSTEM", ok=False, vetoed=True, reasons=names,
                         note="the approved option no longer passes the validator; it will not be used")
            return self._fallback("VETOED_FALLBACK", rejections, f"The approved option failed the validator's re-check ({', '.join(names)}).")
        return self._finish("APPROVED", option.proposal, option.rank, rejections,
                            f"The controller approved option {option.rank}: {describe_proposal(option.proposal)}.")

    def _fallback(self, outcome: str, rejections: int, reason: str) -> SessionResult:
        built = build_fallback(self.ctx)  # checked by the validator, including every rule the controller added
        if built is None:
            self.log.add(self._round, "ESCALATE", note="no safe fallback exists; human decision required")
            return self._finish("ESCALATED", None, None, rejections, f"{reason} No safe fallback exists, so a human must decide.")
        proposal, report = built
        self.log.add(self._round, "FALLBACK", "ATC", fallback_message(proposal, report, self._round), ok=True,
                     note=f"{reason} Applying {FALLBACK_LABEL}.")
        return self._finish(outcome, proposal, None, rejections, f"{reason} Applied {FALLBACK_LABEL}.")

    def _finish(self, outcome: str, proposal: Proposal | None, option: int | None, rejections: int, reason: str) -> SessionResult:
        rules = [c.describe() for c in self.ctx.hard_constraints]
        self.log.add(self._round, "FINAL", "SYSTEM", ok=outcome == "APPROVED", note=reason,
                     data={"outcome": outcome, "option": option, "proposal_id": proposal.proposal_id if proposal else None,
                           "proposal": proposal.model_dump() if proposal else None, "rejections": rejections,
                           "negotiations": self.log.attempt, "rules": rules})
        return SessionResult(outcome, proposal, option, self.log.attempt, rejections, list(self.ctx.hard_constraints), reason, self.log)

    def _log_options(self, options: list[Option], result) -> None:
        shown = "; ".join(f"{o.rank}) {o.title}" for o in options)
        self.log.add(self._round, "OPTIONS", "ATC", note=f"Options for the controller: {shown}.",
                     data={"options": [o.as_dict() for o in options], "rank_rule": RANK_RULE, "outcome": result.outcome.value})


# --- simple decision sources (tests and the terminal) --------------------------------------------------------


class ScriptedDecider:
    """Plays a fixed list of decisions; when the list runs out the controller 'times out'."""

    def __init__(self, decisions: Sequence[Decision]) -> None:
        self._queue = list(decisions)
        self.prompts: list[dict] = []

    def decide(self, options, *, attempt, rejections_left, timeout_s, rules, problem) -> Decision:
        self.prompts.append({"options": [o.rank for o in options], "attempt": attempt, "rejections_left": rejections_left,
                             "timeout_s": timeout_s, "rules": list(rules), "problem": problem})
        return self._queue.pop(0) if self._queue else Decision("timeout")


class AutoApprove:
    """Approves option 1 every time."""

    def decide(self, options, *, attempt, rejections_left, timeout_s, rules, problem) -> Decision:
        return Decision("approve", option=1)


def parse_script(text: str) -> list[Decision]:
    """'reject: avoid gate A1 | approve 2 | timeout'  ->  [Decision, Decision, Decision]."""
    out: list[Decision] = []
    for raw in filter(None, (t.strip() for t in text.split("|"))):
        low = raw.lower()
        try:
            if low.startswith("reject"):
                out.append(Decision("reject", text=raw[len("reject"):].lstrip(" :").strip()))
            elif low.startswith("approve"):
                rest = raw[len("approve"):].lstrip(" :").strip()
                out.append(Decision("approve", option=int(rest) if rest else 1))
            elif low == "timeout":
                out.append(Decision("timeout"))
            else:
                raise ValueError
        except ValueError:
            raise ValueError(f"cannot read decision {raw!r}; use 'approve 1', 'reject: <text>' or 'timeout'") from None
    return out
