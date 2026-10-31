"""Consensus log: every message, validator verdict, veto and state change, hash-chained.

Each entry stores the hash of the previous one, so edits or deletions are detectable with verify().
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Callable, Literal

from pydantic import Field

from .schemas import Message, Strict

Event = Literal[
    "PROPOSAL",        # ATC proposal + validator verdict
    "RESPONSE",        # PILOT/GROUND ACCEPT / REJECT / COUNTER + validator verdict
    "VALIDATOR_VETO",  # an ACCEPT the validator overruled
    "INVALID_OUTPUT",  # unparseable or protocol-violating agent output
    "AGENT_ERROR",     # the agent call itself failed
    "ROUND_END",
    "CONSENSUS",
    "FALLBACK",
    "ESCALATE",
]
GENESIS = "0" * 64
logger = logging.getLogger(__name__)


class LogEntry(Strict):
    seq: int
    round: int
    event: Event
    actor: str  # a Role value, or "SYSTEM"
    message: Message | None = None
    ok: bool | None = None
    vetoed: bool = False
    reasons: list[str] = Field(default_factory=list)
    note: str = ""
    prev_hash: str
    hash: str = ""

    def compute_hash(self) -> str:
        body = self.model_dump(mode="json", exclude={"hash"})
        return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


class ConsensusLog:
    def __init__(self, entries: list[LogEntry] | None = None) -> None:
        self.entries: list[LogEntry] = entries or []
        self._subscribers: list[Callable[[LogEntry], None]] = []

    def subscribe(self, callback: Callable[[LogEntry], None]) -> None:
        """Call `callback(entry)` after every add(), on the thread that added it (used by the dashboard)."""
        self._subscribers.append(callback)

    def add(self, round: int, event: Event, actor: str = "SYSTEM", message: Message | None = None,
            ok: bool | None = None, vetoed: bool = False, reasons: list[str] | None = None, note: str = "") -> LogEntry:
        entry = LogEntry(
            seq=len(self.entries) + 1, round=round, event=event, actor=actor, message=message, ok=ok,
            vetoed=vetoed, reasons=reasons or [], note=note,
            prev_hash=self.entries[-1].hash if self.entries else GENESIS,
        )
        entry.hash = entry.compute_hash()
        self.entries.append(entry)
        for callback in list(self._subscribers):
            try:
                callback(entry)
            except Exception:  # noqa: BLE001 - a broken viewer must never break the negotiation
                logger.exception("consensus log subscriber failed")
        return entry

    def verify(self) -> bool:
        prev = GENESIS
        for i, e in enumerate(self.entries, start=1):
            if e.seq != i or e.prev_hash != prev or e.hash != e.compute_hash():
                return False
            prev = e.hash
        return True

    def events(self, event: Event) -> list[LogEntry]:
        return [e for e in self.entries if e.event == event]

    def to_json(self) -> str:
        return json.dumps([e.model_dump(mode="json") for e in self.entries], indent=2)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ConsensusLog":
        return cls([LogEntry.model_validate(d) for d in json.loads(Path(path).read_text(encoding="utf-8"))])

    def render(self) -> str:
        lines = []
        for e in self.entries:
            what = ""
            if e.message is not None:
                m = e.message
                what = m.type.value
                if m.proposal is not None:
                    p = m.proposal
                    what += (f" delay={p.delay_min} hold={p.hold_minutes}@{p.holding_area} sep={p.separation_sec} "
                             f"gate={p.gate} rwy={p.runway}")
                else:
                    what += f" id={m.proposal_id}"
            verdict = "" if e.ok is None else ("  [ok]" if e.ok else "  [REJECTED by validator]")
            if e.vetoed:
                verdict = "  [VETOED]"
            lines.append(f"#{e.seq:02d} r{e.round} {e.event:<14} {e.actor:<7} {what}{verdict}")
            if e.message is not None and e.message.rationale:
                lines.append(f"      why: {e.message.rationale}")
            for r in e.reasons:
                lines.append(f"      - {r}")
            if e.note:
                lines.append(f"      note: {e.note}")
        return "\n".join(lines)
