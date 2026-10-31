import pytest

from negotiator.constraints import HardConstraint, RunwayObservation, TurbulenceObservation
from negotiator.schemas import MessageType, Proposal
from negotiator.validator import check_proposal, needs_human_confirm, review_message

from .conftest import make_msg, prov


def failed(report):
    return {r.name for r in report.violations}


def test_good_proposal_passes(ctx, good):
    assert check_proposal(good, ctx).passed


def test_separation_violation(ctx, good):
    p = good.model_copy(update={"separation_sec": 119})
    assert failed(check_proposal(p, ctx)) == {"min_separation_sec"}


def test_separation_boundary_ok(ctx, good):
    p = good.model_copy(update={"separation_sec": 120})
    assert check_proposal(p, ctx).passed


def test_hold_too_long(ctx, good):
    p = good.model_copy(update={"hold_minutes": 31})
    assert "max_hold_min" in failed(check_proposal(p, ctx))


def test_fuel_reserve_violation(ctx, good):
    # 6000 - 40*30 = 4800 >= 4800 passes; one more minute is not allowed by max_hold, so lower fuel instead
    ctx.aircraft.fuel_on_board_kg = 5000
    p = good.model_copy(update={"hold_minutes": 30})
    assert "fuel_reserve_kg" in failed(check_proposal(p, ctx))


def test_crew_duty_violation(ctx, good):
    p = good.model_copy(update={"delay_min": 100})  # 600+100+10+90 = 800 > 780
    assert failed(check_proposal(p, ctx)) == {"crew_duty_min"}


def test_unavailable_gate(ctx, good):
    p = good.model_copy(update={"gate": "Z9"})
    assert failed(check_proposal(p, ctx)) == {"gate_available"}


def test_hold_in_turbulent_area(ctx, good):
    p = good.model_copy(update={"holding_area": "SOUTH"})
    assert failed(check_proposal(p, ctx)) == {"turbulence_at_holding_area"}


def test_hold_needs_area(ctx, good):
    p = good.model_copy(update={"holding_area": None})
    assert failed(check_proposal(p, ctx)) == {"turbulence_at_holding_area"}


def test_no_hold_ignores_turbulence(ctx, good):
    p = good.model_copy(update={"hold_minutes": 0, "holding_area": "SOUTH"})
    assert check_proposal(p, ctx).passed


def test_unknown_area_fails_closed(ctx, good):
    p = good.model_copy(update={"holding_area": "WEST"})
    assert "turbulence_at_holding_area" in failed(check_proposal(p, ctx))


def test_wet_runway_violation(ctx, good):
    p = good.model_copy(update={"runway": "09R"})
    assert failed(check_proposal(p, ctx)) == {"runway_standing_water"}


def test_low_confidence_observation_blocks_until_human_confirms(ctx, good):
    ctx.runways = [RunwayObservation(runway="27L", standing_water_mm=1.0, provenance=prov(conf=0.4))]
    assert needs_human_confirm(ctx.runways[0], ctx.margins)
    assert "runway_standing_water" in failed(check_proposal(good, ctx))
    ctx.runways[0].human_confirmed = True
    assert not needs_human_confirm(ctx.runways[0], ctx.margins)
    assert check_proposal(good, ctx).passed


def test_low_confidence_turbulence_blocks(ctx, good):
    ctx.turbulence = [TurbulenceObservation(area="NORTH", index=0.1, provenance=prov(conf=0.2))]
    assert "turbulence_at_holding_area" in failed(check_proposal(good, ctx))


def test_hard_constraint_enforced(ctx, good):
    ctx.hard_constraints = [HardConstraint(field="gate", op="not_in", value=["A1"], source_text="avoid gate A1")]
    report = check_proposal(good, ctx)
    assert failed(report) == {"hard:gatenot_in['A1']"}
    ok = good.model_copy(update={"gate": "B2"})
    assert check_proposal(ok, ctx).passed


def test_hard_numeric_bound_fails_closed_on_non_numeric_field(ctx, good):
    ctx.hard_constraints = [HardConstraint(field="runway", op="<=", value=5)]  # runway is "27L", not a number
    assert "hard:runway<=5" in failed(check_proposal(good, ctx))


def test_hard_numeric_bound_fails_closed_on_missing_value(ctx, good):
    ctx.hard_constraints = [HardConstraint(field="gate", op="<=", value=5)]
    p = good.model_copy(update={"gate": None})
    assert "hard:gate<=5" in failed(check_proposal(p, ctx))


def test_hard_numeric_bound_on_real_number(ctx, good):
    ctx.hard_constraints = [HardConstraint(field="delay_min", op="<=", value=10)]
    assert "hard:delay_min<=10" in failed(check_proposal(good, ctx))  # good.delay_min == 20
    assert check_proposal(good.model_copy(update={"delay_min": 10}), ctx).passed


def test_hard_constraint_shape_validated():
    with pytest.raises(ValueError):
        HardConstraint(field="gate", op="in", value="A1")
    with pytest.raises(ValueError):
        HardConstraint(field="delay_min", op="<=", value="ten")


# --- review_message -------------------------------------------------------


def test_propose_valid_gets_validator_checks(ctx, good):
    v = review_message(make_msg(MessageType.PROPOSE, good), ctx)
    assert v.ok and not v.vetoed
    assert v.message.constraint_check and all(c.passed for c in v.message.constraint_check)


def test_propose_invalid_not_ok(ctx, good):
    bad = good.model_copy(update={"separation_sec": 10})
    v = review_message(make_msg(MessageType.PROPOSE, bad), ctx)
    assert not v.ok and not v.vetoed and v.reasons


def test_self_reported_checks_are_overwritten(ctx, good):
    from negotiator.schemas import ConstraintCheck

    bad = good.model_copy(update={"separation_sec": 10})
    msg = make_msg(MessageType.PROPOSE, bad).model_copy(
        update={"constraint_check": [ConstraintCheck(name="min_separation_sec", passed=True)]}
    )
    v = review_message(msg, ctx)
    assert not v.ok
    assert not next(c for c in v.message.constraint_check if c.name == "min_separation_sec").passed


def test_accept_valid_proposal(ctx, good):
    v = review_message(make_msg(MessageType.ACCEPT, proposal_id=good.proposal_id), ctx, current=good)
    assert v.ok and not v.vetoed


def test_accept_unsafe_proposal_is_vetoed(ctx, good):
    bad = good.model_copy(update={"separation_sec": 10})
    v = review_message(make_msg(MessageType.ACCEPT, proposal_id=bad.proposal_id), ctx, current=bad)
    assert not v.ok and v.vetoed


def test_accept_wrong_proposal_id_is_vetoed(ctx, good):
    v = review_message(make_msg(MessageType.ACCEPT, proposal_id="deadbeef"), ctx, current=good)
    assert v.vetoed


def test_accept_with_nothing_on_table_is_vetoed(ctx):
    v = review_message(make_msg(MessageType.ACCEPT, proposal_id="deadbeef"), ctx, current=None)
    assert v.vetoed


def test_reject_is_ok(ctx, good):
    v = review_message(make_msg(MessageType.REJECT, proposal_id=good.proposal_id), ctx, current=good)
    assert v.ok


def test_round_limit_enforced(ctx, good):
    v = review_message(make_msg(MessageType.PROPOSE, good, round_=4), ctx)
    assert not v.ok and "max_rounds" in v.reasons[0]


def test_accept_after_round_limit_is_vetoed(ctx, good):
    v = review_message(make_msg(MessageType.ACCEPT, proposal_id=good.proposal_id, round_=4), ctx, current=good)
    assert v.vetoed


def test_proposal_id_is_deterministic_and_content_sensitive(good):
    assert good.proposal_id == good.model_copy().proposal_id
    assert good.proposal_id != good.model_copy(update={"delay_min": 21}).proposal_id
