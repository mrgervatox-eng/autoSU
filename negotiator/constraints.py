"""Numeric limits, scenario facts and hard constraints the validator checks against.

Every default here is a SIMULATED value for the hackathon, not a real-world standard.
"""
from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from .schemas import ConstraintCheck, Proposal, Provenance, Strict


class Margins(Strict):
    """Safety margins (simulated defaults)."""

    min_separation_sec: int = 120
    fuel_reserve_kg: float = 1800.0
    max_hold_min: int = 30
    max_duty_min: int = 780
    max_turbulence_index: float = 0.6
    max_standing_water_mm: float = 3.0
    min_observation_confidence: float = 0.6


class Limits(Strict):
    """Loop bounds. Nothing in the system may loop past these."""

    max_rounds: int = 3
    max_human_rejections: int = 2
    decision_timeout_s: int = 60


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
    fuel_on_board_kg: float
    trip_fuel_kg: float
    hold_burn_kg_per_min: float
    crew_duty_used_min: int
    flight_time_remaining_min: int
    available_gates: list[str] = Field(default_factory=list)
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
