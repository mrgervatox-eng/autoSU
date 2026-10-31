"""Plain-language text for the dashboard and the terminal: why each round ended, what each check
means, how a proposal reads. One place for the wording, so the page, the terminal and the tests agree.

Everything an agent wrote is quoted and length-capped here; the page shows it as plain text.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from .log import ConsensusLog, LogEntry
from .schemas import ConstraintCheck, MessageType, Proposal

ROLE_LABEL = {"ATC": "ATC", "PILOT": "Pilot", "GROUND": "Ground", "SYSTEM": "System", "CONTROLLER": "Controller"}
EVENT_LABEL = {"PROPOSAL": "Propose", "VALIDATOR_VETO": "Validator veto", "INVALID_OUTPUT": "Invalid output",
               "AGENT_ERROR": "Agent error", "CONSENSUS": "Consensus", "FALLBACK": "Fallback", "ESCALATE": "Escalate",
               "OPTIONS": "Options for the controller", "DECISION": "Controller decision", "CONSTRAINTS": "Controller rules",
               "RENEGOTIATE": "Renegotiate", "FINAL": "Final decision"}
QUOTE_LIMIT = 160
MORE_AFTER = 2  # failed checks named in a sentence before "and N more"
RESPONDERS = ("GROUND", "PILOT")


# --- small helpers -------------------------------------------------------------------------


def num(x: float | int) -> str:
    """7199.0 -> '7199', 0.3 -> '0.3', 8.70001 -> '8.7'."""
    return f"{round(float(x), 2):.2f}".rstrip("0").rstrip(".")


def label_of(actor: str) -> str:
    return ROLE_LABEL.get(actor, actor.title())


def area_name(name: str | None) -> str | None:
    return name.title() if name else name  # display only: "NORTH" -> "North"


def quote(text: str, limit: int = QUOTE_LIMIT) -> str:
    """Untrusted agent text: collapse whitespace, cap the length, drop the closing full stop."""
    flat = re.sub(r"\s+", " ", text or "").strip().rstrip(".")
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


# --- proposals ------------------------------------------------------------------------------


def proposal_chips(p: Proposal) -> list[str]:
    hold = f"Air hold {p.hold_minutes} min" + (f" @ {area_name(p.holding_area)}" if p.holding_area else "")
    return [f"Ground delay {p.delay_min} min", hold, f"Separation {p.separation_sec} s", f"Gate {p.gate or '–'}", f"Runway {p.runway or '–'}"]


def describe_proposal(p: Proposal) -> str:
    return ", ".join(c[0].lower() + c[1:] for c in proposal_chips(p))


def diff_proposal(old: Proposal, new: Proposal) -> list[str]:
    out = []
    if old.delay_min != new.delay_min:
        out.append(f"ground delay {old.delay_min} → {new.delay_min} min")
    if old.hold_minutes != new.hold_minutes:
        out.append(f"air hold {old.hold_minutes} → {new.hold_minutes} min")
    if old.holding_area != new.holding_area:
        out.append(f"hold area {area_name(old.holding_area) or '–'} → {area_name(new.holding_area) or '–'}")
    if old.separation_sec != new.separation_sec:
        out.append(f"separation {old.separation_sec} → {new.separation_sec} s")
    if old.gate != new.gate:
        out.append(f"gate {old.gate or '–'} → {new.gate or '–'}")
    if old.runway != new.runway:
        out.append(f"runway {old.runway or '–'} → {new.runway or '–'}")
    return out


# --- validator checks, in words ---------------------------------------------------------------

# name -> (label, unit, how the number must compare with the limit, extra explanation when it fails)
_NUMERIC = {
    "min_separation_sec": ("Separation", " s", "at least", ""),
    "max_hold_min": ("Air hold", " min", "at most", ""),
    "fuel_reserve_kg": ("Fuel left after the hold", " kg", "at least", "must cover the trip fuel plus the reserve"),
    "crew_duty_min": ("Crew duty", " min", "at most", ""),
    "turbulence_at_holding_area": ("Turbulence at the holding area", "", "at most", ""),
    "runway_standing_water": ("Runway standing water", " mm", "at most", ""),
}


def check_phrase(c: ConstraintCheck) -> str:
    if c.name == "assignment_complete":
        return "Gate and runway are both assigned" if c.passed else f"Gate and runway must both be assigned ({c.actual})"
    if c.name == "gate_available":
        if c.actual is None:
            return "No gate requested"
        return f"Gate {c.actual} is available" if c.passed else f"Gate {c.actual} is not available (free gates: {c.limit or 'none'})"
    info = _NUMERIC.get(c.name)
    if info is None:  # controller constraint (hard:...) or anything unknown
        label = f"Controller constraint ({c.name[5:]})" if c.name.startswith("hard:") else c.name
        bits = [b for b in (None if c.actual is None else str(c.actual), f"allowed: {c.limit}" if c.limit else None, c.detail or None) if b]
        return label + (": " + ", ".join(bits) if bits else "")
    label, unit, rel, why = info
    if c.actual is None:
        return f"{label}: {c.detail or 'no data'} (need {rel} {num(c.limit)}{unit})"
    line = f"{label}: {num(c.actual)}{unit}, need {rel} {num(c.limit)}{unit}"
    over = (c.actual - c.limit) if rel == "at most" else (c.limit - c.actual)
    if not c.passed and over > 0:
        line += f" ({'over' if rel == 'at most' else 'short'} by {num(over)}{unit})"
    if not c.passed and why:
        line += f"; {why}"
    if c.detail and (c.passed or over <= 0):
        line += f"; {c.detail}"
    return line


def check_lines(checks: Iterable[ConstraintCheck]) -> list[dict[str, Any]]:
    return [{"passed": c.passed, "text": check_phrase(c)} for c in checks]


def entry_view(e: LogEntry) -> dict[str, Any]:
    """Display text for one log entry (sent to the page beside the raw entry)."""
    m = e.message
    label = EVENT_LABEL.get(e.event) or (m.type.value.title() if m else e.event.title())
    return {
        "actor_label": label_of(e.actor),
        "label": label,
        "chips": proposal_chips(m.proposal) if m and m.proposal else [],
        "checks": check_lines(m.constraint_check) if m and m.constraint_check else [],
        "options": [{"rank": o["rank"], "title": o["title"], "chips": o["chips"], "reason": o["reason"]}
                    for o in (e.data or {}).get("options", [])] if e.event == "OPTIONS" else [],
    }


# --- why each round ended ------------------------------------------------------------------------


@dataclass(frozen=True)
class RoundSummary:
    round: int
    kind: str   # consensus | validator_rejected | no_consensus | no_proposal
    short: str  # chip text, e.g. "no consensus (Pilot countered)"
    text: str   # the full sentence
    attempt: int = 1  # 1 = the first negotiation; 2 and up follow a controller rejection

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _first_reason(e: LogEntry) -> str:
    return quote(e.reasons[0] if e.reasons else e.note or "no detail", 100)


def _failures(e: LogEntry) -> list[str]:
    return [check_phrase(c) for c in (e.message.constraint_check if e.message else []) if not c.passed]


def _limited(items: list[str]) -> str:
    shown = items[:MORE_AFTER]
    extra = len(items) - len(shown)
    return "; ".join(shown) + (f"; and {extra} more" if extra else "")


def round_label(r: int, attempt: int = 1) -> str:
    return f"Round {r}" if attempt == 1 else f"Renegotiation {attempt - 1}, round {r}"


def round_summary(entries: Sequence[LogEntry], r: int, attempt: int = 1) -> RoundSummary:
    es = [e for e in entries if e.round == r and e.attempt == attempt]
    head = f"{round_label(r, attempt)} ended:"

    def make(kind: str, short: str, text: str) -> RoundSummary:
        return RoundSummary(r, kind, short, text, attempt)

    prop = next((e for e in es if e.event == "PROPOSAL"), None)
    if any(e.event == "CONSENSUS" for e in es):
        return make("consensus", "consensus",
                    f"{head} consensus. ATC proposed {describe_proposal(prop.message.proposal)}; Ground and Pilot accepted; the validator confirmed it.")

    if prop is None:
        bad = next((e for e in es if e.actor == "ATC" and e.event in ("INVALID_OUTPUT", "AGENT_ERROR")), None)
        what = ("ATC sent invalid output" if bad.event == "INVALID_OUTPUT" else "ATC's call failed") if bad else "ATC gave no answer"
        return make("no_proposal", "no usable ATC proposal",
                    f"{head} no consensus. {what}" + (f" ({_first_reason(bad)})" if bad else "") + ", so nothing was proposed.")

    desc = describe_proposal(prop.message.proposal)
    if prop.ok is False:
        fails = _failures(prop) or [quote(x) for x in prop.reasons]
        return make("validator_rejected", "validator rejected the proposal",
                    f"{head} the validator rejected ATC's proposal ({desc}). {_limited(fails)}.")

    parts: list[str] = []
    objectors: list[str] = []
    for role in RESPONDERS:
        who = label_of(role)
        resp = next((e for e in es if e.actor == role and e.event == "RESPONSE"), None)
        bad = next((e for e in es if e.actor == role and e.event in ("INVALID_OUTPUT", "AGENT_ERROR")), None)
        if resp is not None:
            m = resp.message
            if resp.vetoed:
                parts.append(f"{who}'s accept was vetoed by the validator ({_first_reason(resp)})")
                objectors.append(f"{who}'s accept vetoed")
            elif m.type == MessageType.ACCEPT:
                parts.append(f"{who} accepted")
            elif m.type == MessageType.COUNTER:
                s = f"{who} countered"
                if m.rationale:
                    s += f': "{quote(m.rationale)}"'
                changes = diff_proposal(prop.message.proposal, m.proposal)
                if changes:
                    s += f" (changed: {', '.join(changes)})"
                if resp.ok is False:
                    s += "; the validator rejected that counter as unsafe"
                parts.append(s)
                objectors.append(f"{who} countered")
            else:
                parts.append(f"{who} rejected" + (f': "{quote(m.rationale)}"' if m.rationale else ""))
                objectors.append(f"{who} rejected")
        elif bad is not None:
            what = "sent invalid output" if bad.event == "INVALID_OUTPUT" else "call failed"
            parts.append(f"{who} {what} ({_first_reason(bad)})" if what == "sent invalid output" else f"{who}'s call failed ({_first_reason(bad)})")
            objectors.append(f"{who} {what}" if what == "sent invalid output" else f"{who}'s call failed")
        else:
            parts.append(f"{who} did not reply")
            objectors.append(f"{who} did not reply")

    final = next((e for e in es if e.event == "VALIDATOR_VETO" and e.actor == "SYSTEM"), None)
    if final is not None:
        parts.append(f"both accepted, but the validator's final check failed ({', '.join(final.reasons) or 'no detail'})")
    short = "no consensus (" + (", ".join(objectors) if objectors else "final validator check failed") + ")"
    return make("no_consensus", short, f"{head} no consensus. ATC proposed {desc}. " + "; ".join(parts) + ".")


def all_round_summaries(entries: Sequence[LogEntry]) -> list[RoundSummary]:
    return [round_summary(entries, r, a) for a, r in sorted({(e.attempt, e.round) for e in entries})]


class RoundTracker:
    """Decides when a round has ended while entries arrive one at a time (used by the dashboard and the terminal).

    A round ends when an entry from a later round or a later attempt arrives, when the fallback, escalation or the
    options for the controller arrive, or when the run finishes. Each round is announced once.
    """

    ENDS_THE_ROUND = ("FALLBACK", "ESCALATE", "OPTIONS")

    def __init__(self) -> None:
        self._last: tuple[int, int] | None = None  # (attempt, round)
        self._done: set[tuple[int, int]] = set()

    def _emit(self, entries: Sequence[LogEntry], key: tuple[int, int]) -> list[RoundSummary]:
        if key in self._done:
            return []
        self._done.add(key)
        return [round_summary(entries, key[1], key[0])]

    def before(self, entries: Sequence[LogEntry], entry: LogEntry) -> list[RoundSummary]:
        """Summaries to show before `entry` (which must already be the last item of `entries`)."""
        out: list[RoundSummary] = []
        key = (entry.attempt, entry.round)
        if self._last is not None and (key != self._last or entry.event in self.ENDS_THE_ROUND):
            out = self._emit(entries, self._last)
        self._last = key
        return out

    def finish(self, entries: Sequence[LogEntry]) -> list[RoundSummary]:
        return self._emit(entries, self._last) if self._last is not None else []


def render_with_rounds(log: ConsensusLog) -> str:
    """Terminal view: the log with a plain-language line after every round."""
    tracker, lines, seen = RoundTracker(), [], []
    for e in log.entries:
        seen.append(e)
        lines += [f"      -- {s.text}" for s in tracker.before(seen, e)]
        lines += log.render_entry(e)
    lines += [f"      -- {s.text}" for s in tracker.finish(seen)]
    return "\n".join(lines)
