import json

import pytest

from negotiator.engine import Negotiation, Outcome
from negotiator.fake_llm import FakeLLM
from negotiator.negotiate import main
from negotiator.scenario import FIXTURE_CSV, build_scenario
from negotiator.schemas import Role, Source
from negotiator.validator import check_proposal


@pytest.fixture(params=["weather", "late_aircraft"])
def scen(request):
    return build_scenario(FIXTURE_CSV, 1, 1, kind=request.param)


def run(scen, llm=None):
    llm = llm or FakeLLM()
    return Negotiation(scen, llm).run(), llm


def count(res, event):
    return len(res.log.events(event))


def test_default_agents_reach_consensus(scen):
    res, _ = run(scen)
    assert res.outcome == Outcome.CONSENSUS
    assert 1 <= res.rounds_used <= scen.context.limits.max_rounds
    assert check_proposal(res.proposal, scen.context).passed
    assert res.log.verify()
    assert count(res, "CONSENSUS") == 1 and count(res, "FALLBACK") == 0


def test_weather_scenario_needs_a_revision_round():
    scen = build_scenario(FIXTURE_CSV, 1, 1, kind="weather")
    res, _ = run(scen)
    first = res.log.events("PROPOSAL")[0]
    assert first.ok is False  # naive opening bid picks the wet runway; validator rejects it
    assert any("runway_standing_water" in r for r in first.reasons)
    assert res.rounds_used >= 2 and res.proposal.runway == "27L"


def test_every_logged_message_has_validator_checks(scen):
    res, _ = run(scen)
    for e in res.log.entries:
        if e.message is not None and e.message.type.value in ("PROPOSE", "COUNTER") and e.event != "FALLBACK":
            assert e.message.constraint_check, e
            assert e.message.provenance.source in (Source.SIMULATED, Source.LLM)


def test_agents_are_called_one_role_per_call_and_bounded(scen):
    res, llm = run(scen)
    assert set(llm.calls) <= {Role.ATC, Role.PILOT, Role.GROUND}
    assert len(llm.calls) <= 3 * scen.context.limits.max_rounds


def test_stubborn_pilot_forces_fallback_after_max_rounds(scen):
    res, llm = run(scen, FakeLLM({"PILOT": "stubborn"}))
    assert res.outcome == Outcome.FALLBACK
    assert res.rounds_used == scen.context.limits.max_rounds
    assert llm.calls.count(Role.ATC) == scen.context.limits.max_rounds  # loop is bounded
    assert check_proposal(res.proposal, scen.context).passed
    fb = res.log.events("FALLBACK")[0]
    assert "our conservative default procedure" in fb.note
    assert res.log.verify()


def test_stubborn_atc_unsafe_proposals_never_circulate(scen):
    res, llm = run(scen, FakeLLM({"ATC": "stubborn"}))
    assert res.outcome == Outcome.FALLBACK
    assert Role.PILOT not in llm.calls and Role.GROUND not in llm.calls
    assert all(e.ok is False for e in res.log.events("PROPOSAL"))


def test_reckless_accept_is_vetoed(scen):
    res, _ = run(scen, FakeLLM({"PILOT": "reckless"}))
    assert res.outcome == Outcome.FALLBACK
    assert count(res, "VALIDATOR_VETO") >= 1
    assert all(e.vetoed for e in res.log.events("VALIDATOR_VETO"))


def test_accept_of_unsafe_state_is_vetoed_when_sensors_change(scen):
    """A proposal valid when circulated becomes unsafe before the pilot answers: veto."""
    ctx = scen.context

    class ChangingLLM(FakeLLM):
        def complete(self, role, view):
            if role == Role.PILOT:  # a runway sensor update lands mid-round
                for r in ctx.runways:
                    r.standing_water_mm = 9.0
            return super().complete(role, view)

    res = Negotiation(scen, ChangingLLM({"PILOT": "yes_man"})).run()
    assert count(res, "VALIDATOR_VETO") >= 1
    assert res.outcome != Outcome.CONSENSUS


def test_garbage_output_is_logged_not_fatal(scen):
    res, _ = run(scen, FakeLLM({"GROUND": "garbage"}))
    assert count(res, "INVALID_OUTPUT") >= 1
    assert res.outcome == Outcome.FALLBACK


def test_impostor_sender_is_rejected(scen):
    res, _ = run(scen, FakeLLM({"PILOT": "impostor"}))
    bad = res.log.events("INVALID_OUTPUT")
    assert bad and "does not match role" in bad[0].reasons[0]
    assert res.outcome == Outcome.FALLBACK


def test_agent_exception_is_contained(scen):
    class Boom(FakeLLM):
        def complete(self, role, view):
            if role == Role.GROUND:
                raise TimeoutError("network down")
            return super().complete(role, view)

    res = Negotiation(scen, Boom()).run()
    assert count(res, "AGENT_ERROR") >= 1 and res.outcome == Outcome.FALLBACK


def test_escalates_when_no_safe_fallback_exists(scen):
    scen.context.runways = []  # nowhere safe to land
    res, _ = run(scen)
    assert res.outcome == Outcome.ESCALATED and res.proposal is None
    assert count(res, "ESCALATE") == 1


def test_hard_constraint_is_respected_by_consensus(scen):
    from negotiator.constraints import HardConstraint

    scen.context.hard_constraints = [HardConstraint(field="holding_area", op="==", value="NORTH")]
    res, _ = run(scen)
    assert res.proposal.holding_area == "NORTH" or res.proposal.hold_minutes == 0


def test_max_rounds_is_configurable(scen):
    scen.context.limits.max_rounds = 1
    res, llm = run(scen, FakeLLM({"PILOT": "stubborn"}))
    assert res.rounds_used == 1 and llm.calls.count(Role.ATC) == 1


def test_unknown_persona_rejected():
    with pytest.raises(ValueError):
        FakeLLM({"PILOT": "sleepy"})


# --- CLI -------------------------------------------------------------------


def test_cli_runs_in_fake_mode(capsys, tmp_path):
    out = tmp_path / "log.json"
    assert main(["--csv", str(FIXTURE_CSV), "--kind", "weather", "--log-out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "RESULT: CONSENSUS" in text and "hash chain valid: True" in text and "FAKE MODE" in text
    assert json.loads(out.read_text())


def test_cli_persona_forces_fallback(capsys):
    assert main(["--csv", str(FIXTURE_CSV), "--persona", "PILOT=stubborn"]) == 0
    assert "RESULT: FALLBACK" in capsys.readouterr().out


def test_cli_bad_persona(capsys):
    assert main(["--csv", str(FIXTURE_CSV), "--persona", "PILOT=sleepy"]) == 1
