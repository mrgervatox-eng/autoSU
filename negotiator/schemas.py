"""Structured message schemas shared by every agent, the validator and the log.

All models forbid unknown fields so an LLM that invents keys is rejected at parse time.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Role(str, Enum):
    ATC = "ATC"
    PILOT = "PILOT"
    GROUND = "GROUND"


class MessageType(str, Enum):
    PROPOSE = "PROPOSE"
    COUNTER = "COUNTER"
    ACCEPT = "ACCEPT"
    REJECT = "REJECT"


class Source(str, Enum):
    LLM = "llm"
    SENSOR = "sensor"
    HUMAN = "human"
    DATASET = "dataset"
    SIMULATED = "simulated"
    VALIDATOR = "validator"
    FALLBACK = "fallback"


class Provenance(Strict):
    source: Source
    confidence: float = Field(ge=0.0, le=1.0)
    note: str = ""


class Proposal(Strict):
    """A concrete resolution. Numeric fields are what the validator checks."""

    delay_min: int = Field(ge=0, le=1440, description="ground delay before departure/arrival slot")
    hold_minutes: int = Field(ge=0, le=1440, description="airborne hold")
    separation_sec: int = Field(ge=0, le=3600, description="planned separation to preceding aircraft")
    holding_area: str | None = None
    gate: str | None = None
    runway: str | None = None

    @property
    def proposal_id(self) -> str:
        """Deterministic content hash; ACCEPT/REJECT must reference it."""
        canonical = json.dumps(self.model_dump(), sort_keys=True)
        return hashlib.sha1(canonical.encode()).hexdigest()[:8]


class ConstraintCheck(Strict):
    name: str
    passed: bool
    actual: float | int | str | None = None
    limit: float | int | str | None = None
    detail: str = ""


class Message(Strict):
    sender: Role
    round: int = Field(ge=1)
    type: MessageType
    proposal: Proposal | None = None
    proposal_id: str | None = None
    constraint_check: list[ConstraintCheck] = Field(default_factory=list)
    provenance: Provenance
    rationale: str = Field(default="", max_length=500)
    ts: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @model_validator(mode="after")
    def _shape_matches_type(self) -> "Message":
        if self.type in (MessageType.PROPOSE, MessageType.COUNTER) and self.proposal is None:
            raise ValueError(f"{self.type.value} requires a proposal")
        if self.type in (MessageType.ACCEPT, MessageType.REJECT) and not self.proposal_id:
            raise ValueError(f"{self.type.value} requires proposal_id")
        return self


def parse_message(raw: str | dict[str, Any]) -> tuple[Message | None, str | None]:
    """Parse untrusted (LLM) output. Returns (message, None) or (None, error)."""
    try:
        data = json.loads(raw) if isinstance(raw, str) else raw
        return Message.model_validate(data), None
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        return None, f"invalid message: {exc}"
