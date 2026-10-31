import pytest

from negotiator.constraints import AircraftState, ConstraintContext, RunwayObservation, TurbulenceObservation
from negotiator.schemas import Message, MessageType, Proposal, Provenance, Role, Source


def prov(conf=0.9, source=Source.SIMULATED):
    return Provenance(source=source, confidence=conf)


@pytest.fixture
def ctx():
    return ConstraintContext(
        aircraft=AircraftState(
            tail_number="N123AA",
            fuel_on_board_kg=6000,
            trip_fuel_kg=3000,
            hold_burn_kg_per_min=40,
            crew_duty_used_min=600,
            flight_time_remaining_min=90,
            available_gates=["A1", "B2"],
        ),
        turbulence=[
            TurbulenceObservation(area="NORTH", index=0.3, provenance=prov()),
            TurbulenceObservation(area="SOUTH", index=0.8, provenance=prov()),
        ],
        runways=[
            RunwayObservation(runway="27L", standing_water_mm=1.0, provenance=prov()),
            RunwayObservation(runway="09R", standing_water_mm=6.0, provenance=prov()),
        ],
    )


@pytest.fixture
def good():
    """A proposal that passes every check against `ctx`."""
    return Proposal(delay_min=20, hold_minutes=10, separation_sec=150, holding_area="NORTH", gate="A1", runway="27L")


def make_msg(type_, proposal=None, proposal_id=None, round_=1, sender=Role.ATC):
    return Message(sender=sender, round=round_, type=type_, proposal=proposal, proposal_id=proposal_id, provenance=prov(source=Source.LLM))
