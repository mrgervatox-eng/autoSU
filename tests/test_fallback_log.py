import pytest

from negotiator.constraints import RunwayObservation, TurbulenceObservation
from negotiator.fallback import FALLBACK_LABEL, build_fallback, fallback_message
from negotiator.log import ConsensusLog
from negotiator.scenario import FIXTURE_CSV, build_scenario
from negotiator.schemas import Source
from negotiator.validator import check_proposal

from .conftest import prov


@pytest.fixture(params=["weather", "late_aircraft"])
def scen(request):
    return build_scenario(FIXTURE_CSV, 1, 1, kind=request.param)


def test_fallback_is_validator_approved(scen):
    proposal, report = build_fallback(scen.context)
    assert report.passed and check_proposal(proposal, scen.context).passed
    assert proposal.hold_minutes <= 10
    assert proposal.separation_sec >= scen.context.margins.min_separation_sec + 60


def test_fallback_avoids_bad_runway_and_rough_area():
    ctx = build_scenario(FIXTURE_CSV, 1, 1, kind="weather").context
    proposal, _ = build_fallback(ctx)
    assert proposal.runway == "27L" and proposal.holding_area == "NORTH"


def test_fallback_is_deterministic(scen):
    assert build_fallback(scen.context)[0] == build_fallback(scen.context)[0]


def test_fallback_holds_zero_when_fuel_is_tight(scen):
    ctx = scen.context
    ctx.aircraft.fuel_on_board_kg = ctx.aircraft.trip_fuel_kg + ctx.margins.fuel_reserve_kg + 100
    proposal, report = build_fallback(ctx)
    assert proposal.hold_minutes == 0 and proposal.holding_area is None and report.passed


def test_fallback_none_when_no_usable_runway(scen):
    scen.context.runways = [RunwayObservation(runway="27L", standing_water_mm=1.0, provenance=prov(conf=0.1))]
    assert build_fallback(scen.context) is None  # low confidence and unconfirmed
    scen.context.runways[0].human_confirmed = True
    assert build_fallback(scen.context) is not None


def test_fallback_skips_unconfirmed_turbulence_area(scen):
    scen.context.turbulence = [TurbulenceObservation(area="NORTH", index=0.1, provenance=prov(conf=0.1))]
    proposal, _ = build_fallback(scen.context)
    assert proposal.hold_minutes == 0


def test_fallback_message_is_labelled(scen):
    proposal, report = build_fallback(scen.context)
    msg = fallback_message(proposal, report, 3)
    assert msg.provenance.source == Source.FALLBACK
    assert FALLBACK_LABEL in msg.provenance.note and FALLBACK_LABEL in msg.rationale


# --- consensus log ---------------------------------------------------------


def test_log_hash_chain_detects_tampering():
    log = ConsensusLog()
    log.add(1, "ROUND_END", note="a")
    log.add(1, "CONSENSUS", note="b")
    assert log.verify()
    log.entries[0].note = "edited"
    assert not log.verify()


def test_log_detects_deleted_entry():
    log = ConsensusLog()
    for i in range(3):
        log.add(1, "ROUND_END", note=str(i))
    del log.entries[1]
    assert not log.verify()


def test_log_save_load_roundtrip(tmp_path):
    log = ConsensusLog()
    log.add(1, "ROUND_END", note="x", reasons=["why"])
    log.save(tmp_path / "d" / "log.json")
    loaded = ConsensusLog.load(tmp_path / "d" / "log.json")
    assert loaded.verify() and [e.note for e in loaded.entries] == ["x"]
