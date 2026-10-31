"""Numeric limits, scenario facts and hard constraints the validator checks against.

Every default here is a SIMULATED value for the hackathon, not a real-world standard.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import ConstraintCheck, Proposal, Provenance, Strict


class Margins(Strict):
    """Safety margins. Every default is a SIMULATED policy value for the demo, not a real-world standard.

    Ranges are enforced when config is loaded, so a typo (negative fuel, zero separation) fails loudly
    instead of silently weakening the validator. docs/VALIDATOR.md explains each parameter.
    """

    min_separation_sec: int = Field(default=120, ge=1, le=3600, description="seconds: least separation a proposal may declare")
    fuel_reserve_kg: float = Field(default=1800.0, ge=0, description="kg: fuel that must remain after trip fuel and any hold")
    max_hold_min: int = Field(default=30, ge=0, le=1440, description="minutes: longest airborne hold allowed")
    max_duty_min: int = Field(default=780, ge=1, le=1440, description="minutes: crew duty limit including delay, hold and remaining flight")
    max_turbulence_index: float = Field(default=0.6, gt=0, le=1, description="0-1 proxy: roughest air a hold may be in")
    max_standing_water_mm: float = Field(default=3.0, ge=0, description="mm: deepest standing water allowed on an assigned runway")
    min_observation_confidence: float = Field(default=0.6, ge=0, le=1, description="0-1: readings below this need a human confirm")
    require_gate_and_runway: bool = Field(default=True, description="a proposal must assign both a gate and a runway")


class Limits(Strict):
    """Loop bounds. Nothing in the system may loop past these."""

    max_rounds: int = Field(default=3, ge=1, le=10, description="negotiation rounds before the deterministic fallback")
    max_human_rejections: int = Field(default=2, ge=0, le=5, description="controller rejections before the fallback")
    decision_timeout_s: int = Field(default=60, ge=1, le=3600, description="seconds the controller has to decide")


class TurbulenceObservation(Strict):
    area: str
    index: float = Field(ge=0.0, le=1.0)
    provenance: Provenance
    human_confirmed: bool = False


class RunwayObservation(Strict):
    runway: str
    standing_water_mm: float = Field(ge=0.0)
    provenance: Provenance
    human_confirmed: bool = False


class AircraftState(Strict):
    """SIMULATED scenario-enrichment layer (fuel, gates, crew hours)."""

    tail_number: str
    fuel_on_board_kg: float = Field(description="kg loaded at departure")
    trip_fuel_kg: float = Field(description="kg needed to fly from the origin to the destination")
    hold_burn_kg_per_min: float = Field(description="kg burned per minute of airborne hold near the destination")
    crew_duty_used_min: int = Field(description="minutes of crew duty already used before departure")
    flight_time_remaining_min: int = Field(description="minutes of flying from departure to arrival")
    available_gates: list[str] = Field(default_factory=list, description="free gates at the destination")
    enrichment_label: str = "SIMULATED scenario-enrichment layer"


HardField = Literal["delay_min", "hold_minutes", "separation_sec", "holding_area", "gate", "runway"]
HardOp = Literal["<=", ">=", "==", "!=", "in", "not_in"]


class HardConstraint(Strict):
    """Structured constraint, e.g. parsed from a controller's free-text rejection."""

    field: HardField
    op: HardOp
    value: int | float | str | list[str]
    source_text: str = ""

    @model_validator(mode="after")
    def _value_matches_op(self) -> "HardConstraint":
        is_list = isinstance(self.value, list)
        if self.op in ("in", "not_in") and not is_list:
            raise ValueError(f"op '{self.op}' needs a list value")
        if self.op not in ("in", "not_in") and is_list:
            raise ValueError(f"op '{self.op}' needs a scalar value")
        if self.op in ("<=", ">=") and not isinstance(self.value, (int, float)):
            raise ValueError(f"op '{self.op}' needs a numeric value")
        return self

    @property
    def label(self) -> str:
        return f"hard:{self.field}{self.op}{self.value}"

    def check(self, proposal: Proposal) -> ConstraintCheck:
        actual = getattr(proposal, self.field)
        if self.op in ("<=", ">="):
            # Non-numeric (incl. None) cannot satisfy a numeric bound: fail closed.
            ok = isinstance(actual, (int, float)) and (actual <= self.value if self.op == "<=" else actual >= self.value)
        elif self.op == "==":
            ok = actual == self.value
        elif self.op == "!=":
            ok = actual != self.value
        elif self.op == "in":
            ok = actual in self.value
        else:
            ok = actual not in self.value
        return ConstraintCheck(
            name=self.label,
            passed=ok,
            actual=None if actual is None else actual,
            limit=str(self.value),
            detail=self.source_text,
        )


class ConstraintContext(Strict):
    """Everything the validator needs to judge a proposal."""

    margins: Margins = Field(default_factory=Margins)
    limits: Limits = Field(default_factory=Limits)
    aircraft: AircraftState
    turbulence: list[TurbulenceObservation] = Field(default_factory=list)
    runways: list[RunwayObservation] = Field(default_factory=list)
    hard_constraints: list[HardConstraint] = Field(default_factory=list)
