"""Stage 6: ranked options, and the controller loop (approve / reject / timeout)."""
import random
from collections import Counter

import pytest

from negotiator.constraints import HardConstraint, RunwayObservation, TurbulenceObservation
from negotiator.engine import NegotiationResult, Outcome
from negotiator.fake_llm import PERSONAS, FakeLLM
from negotiator.log import ConsensusLog
from negotiator.options import build_options, metric_lines, metrics_for
from negotiator.parsing import MAX_CONSTRAINTS, RuleParser
from negotiator.scenario import FIXTURE_CSV, build_scenario
from negotiator.schemas import Proposal, Role
from negotiator.session import MAX_UNCLEAR, OUTCOMES, AutoApprove, Decision, ScriptedDecider, Session, parse_script
from negotiator.summary import all_round_summaries, entry_view, round_label
from negotiator.validator import check_proposal
from negotiator.engine import Negotiation

from .test_validator_verification import random_context

APPROVE1, APPROVE2 = Decision("approve", option=1), Decision("approve", option=2)


def reject(text):
    return Decision("reject", text=text)


@pytest.fixture
def scen():
    s = build_scenario(FIXTURE_CSV, 1, 1, kind="weather")
    s.context.runways.append(RunwayObservation(runway="18C", standing_water_mm=1.0, provenance=s.context.runways[0].provenance))
    return s


def run(scen, decisions, personas=None, parser=None):
    decider = ScriptedDecider(decisions)
    result = Session(scen, FakeLLM(personas or {}), parser or RuleParser(), decider).run()
    return result, decider


def events(result, name):
    return [e for e in result.log.entries if e.event == name]


def sequence(result):
    return [e.event for e in result.log.entries if e.event in ("CONSENSUS", "OPTIONS", "DECISION", "CONSTRAINTS", "RENEGOTIATE", "FALLBACK", "ESCALATE", "FINAL")]


# --- options -----------------------------------------------------------------------------------------------


def consensus(scen):
    res = Negotiation(scen, FakeLLM()).run()
    assert res.outcome == Outcome.CONSENSUS
    return res


def test_two_ranked_validator_approved_options_after_consensus(scen):
    res = consensus(scen)
    opts = build_options(res, scen.context)
    assert [o.rank for o in opts] == [1, 2] and opts[0].id == res.proposal.proposal_id
    assert all(check_proposal(o.proposal, scen.context).passed for o in opts)
    assert opts[0].title == "Recommended: what the agents agreed" and "Ground and Pilot accepted" in opts[0].reason
    assert opts[1].title == "Alternative: take the hold on the ground instead"
    assert opts[1].proposal.hold_minutes == 0 and opts[1].proposal.delay_min == res.proposal.delay_min + res.proposal.hold_minutes
    assert opts[1].reason.startswith("Compared with option 1:") and "kg fuel margin" in opts[1].reason and "air hold" in opts[1].reason


def test_option_metrics_and_lines(scen):
    p = Proposal(delay_min=20, hold_minutes=10, separation_sec=150, holding_area="NORTH", gate="A1", runway="27L")
    m = metrics_for(p, scen.context)
    a, mg = scen.context.aircraft, scen.context.margins
    assert m["total_delay_min"] == 30
    assert m["fuel_margin_kg"] == round(a.fuel_on_board_kg - a.hold_burn_kg_per_min * 10 - a.trip_fuel_kg - mg.fuel_reserve_kg, 1)
    assert m["duty_margin_min"] == mg.max_duty_min - a.crew_duty_used_min - 20 - 10 - a.flight_time_remaining_min
    assert m["runway_margin_mm"] is not None and m["turbulence_margin"] is not None
    lines = metric_lines(m)
    assert lines[0] == "Total delay absorbed: 30 min" and any(l.startswith("Runway water under the limit by") for l in lines)
    none = metrics_for(p.model_copy(update={"hold_minutes": 0, "holding_area": None, "runway": None}), scen.context)
    assert none["runway_margin_mm"] is None and none["turbulence_margin"] is None and metric_lines(none)[-1] == "No airborne hold"


def result_of(outcome, proposal):
    return NegotiationResult(outcome, proposal, 1, "x", ConsensusLog())


def test_no_hold_means_the_alternative_is_the_conservative_default(scen):
    p = Proposal(delay_min=30, hold_minutes=0, separation_sec=150, gate="A1", runway="27L")
    opts = build_options(result_of(Outcome.CONSENSUS, p), scen.context)
    assert len(opts) == 2 and opts[1].title == "Alternative: our conservative default procedure"


def test_a_fallback_result_leads_with_the_fallback(scen):
    from negotiator.fallback import build_fallback

    fb = build_fallback(scen.context)[0]
    opts = build_options(result_of(Outcome.FALLBACK, fb), scen.context)
    assert opts[0].title == "Recommended: our conservative default procedure" and "did not agree" in opts[0].reason
    assert all(o.title != "Alternative: our conservative default procedure" for o in opts)  # it is already option 1
    assert len(opts) <= 2 and all(check_proposal(o.proposal, scen.context).passed for o in opts)


def test_only_one_option_when_there_is_no_valid_alternative(scen):
    from negotiator.fallback import build_fallback

    fb = build_fallback(scen.context)[0].model_copy(update={"hold_minutes": 0, "holding_area": None})
    scen.context.hard_constraints = [HardConstraint(field="delay_min", op="<=", value=fb.delay_min)]
    opts = build_options(result_of(Outcome.FALLBACK, fb), scen.context)
    assert [o.rank for o in opts] == [1]


def test_an_alternative_that_breaks_a_rule_is_dropped(scen):
    res = consensus(scen)
    scen.context.hard_constraints = [HardConstraint(field="delay_min", op="<=", value=res.proposal.delay_min)]  # the hold-to-ground option would exceed it
    opts = build_options(res, scen.context)
    assert all(check_proposal(o.proposal, scen.context).passed for o in opts)
    assert all(o.proposal.hold_minutes > 0 or o.proposal.delay_min <= res.proposal.delay_min for o in opts)


def test_nothing_to_recommend_is_an_error(scen):
    with pytest.raises(ValueError):
        build_options(result_of(Outcome.ESCALATED, None), scen.context)


def test_option_dict_is_ready_for_the_page(scen):
    d = build_options(consensus(scen), scen.context)[0].as_dict()
    assert {"rank", "id", "title", "reason", "proposal", "chips", "metrics", "metric_lines"} <= set(d)
    assert d["chips"][0].startswith("Ground delay ")


def test_metrics_use_the_proposals_own_area_and_the_worst_reading(ctx):
    p = Proposal(delay_min=0, hold_minutes=10, separation_sec=150, holding_area="SOUTH", gate="A1", runway="09R")
    m = metrics_for(p, ctx)
    assert m["turbulence_margin"] == -0.2 and m["runway_margin_mm"] == -3.0  # SOUTH 0.8 vs 0.6; runway 09R 6.0 mm vs 3.0 mm
    ctx.runways.append(RunwayObservation(runway="27L", standing_water_mm=2.5, provenance=ctx.runways[0].provenance))  # a second, worse 27L reading
    m2 = metrics_for(p.model_copy(update={"holding_area": "NORTH", "runway": "27L"}), ctx)
    assert m2["turbulence_margin"] == 0.3 and m2["runway_margin_mm"] == 0.5
    ctx.turbulence.append(TurbulenceObservation(area="NORTH", index=0.5, provenance=ctx.runways[0].provenance))  # a rougher second NORTH reading
    assert metrics_for(p.model_copy(update={"holding_area": "NORTH"}), ctx)["turbulence_margin"] == 0.1


def test_turbulence_margin_needs_both_a_hold_and_a_reading(ctx):
    base = Proposal(delay_min=0, hold_minutes=0, separation_sec=150, holding_area="NORTH", gate="A1", runway="27L")
    assert metrics_for(base, ctx)["turbulence_margin"] is None  # an area is named but nobody holds
    assert metrics_for(base.model_copy(update={"hold_minutes": 5, "holding_area": "WEST"}), ctx)["turbulence_margin"] is None  # nothing observed there


def test_the_comparison_sentence_is_exact():
    from negotiator.options import _compare

    a = Proposal(delay_min=20, hold_minutes=10, separation_sec=150, holding_area="NORTH", gate="A1", runway="27L")
    b = a.model_copy(update={"delay_min": 30, "hold_minutes": 0, "holding_area": None})
    assert _compare(a, b, {"fuel_margin_kg": 100.0, "duty_margin_min": 50}, {"fuel_margin_kg": 500.0, "duty_margin_min": 40}) == (
        "Compared with option 1: ground delay 20 → 30 min; air hold 10 → 0 min; hold area North → –; +400 kg fuel margin; -10 min crew-duty room.")
    assert _compare(a, a, {"fuel_margin_kg": 100.0, "duty_margin_min": 50}, {"fuel_margin_kg": 40.0, "duty_margin_min": 70}) == (
        "Compared with option 1: -60 kg fuel margin; +20 min crew-duty room.")
    same = {"fuel_margin_kg": 1.0, "duty_margin_min": 1}
    assert _compare(a, a, same, same) == "Compared with option 1: no numeric difference."


def test_options_decisions_and_results_cannot_be_changed_afterwards(scen):
    from dataclasses import FrozenInstanceError

    opt = build_options(consensus(scen), scen.context)[0]
    with pytest.raises(FrozenInstanceError):
        opt.rank = 2
    with pytest.raises(FrozenInstanceError):
        APPROVE1.option = 2


@pytest.mark.parametrize("update, sentence", [
    ({}, "Keep the flight on the ground for 20 minutes, then hold for 10 minutes in the North area; land on runway 27L and park at gate A1, "
         "keeping 150 seconds behind the aircraft ahead."),
    ({"hold_minutes": 0, "holding_area": None}, "Keep the flight on the ground for 20 minutes, with no airborne hold; land on runway 27L and park at gate A1, "
                                                "keeping 150 seconds behind the aircraft ahead."),
    ({"delay_min": 0}, "No ground delay; hold for 10 minutes in the North area; land on runway 27L and park at gate A1, keeping 150 seconds behind the aircraft ahead."),
    ({"delay_min": 0, "hold_minutes": 0, "holding_area": None}, "No ground delay and no airborne hold; land on runway 27L and park at gate A1, "
                                                                "keeping 150 seconds behind the aircraft ahead."),
    ({"runway": None, "gate": None}, "Keep the flight on the ground for 20 minutes, then hold for 10 minutes in the North area; no runway assigned and "
                                     "no gate assigned, keeping 150 seconds behind the aircraft ahead."),
])
def test_plain_plan_reads_like_a_sentence(update, sentence):
    from negotiator.options import plain_plan

    p = Proposal(delay_min=20, hold_minutes=10, separation_sec=150, holding_area="NORTH", gate="A1", runway="27L").model_copy(update=update)
    assert plain_plan(p) == sentence


def test_every_option_carries_its_plain_sentence(scen):
    for o in build_options(consensus(scen), scen.context):
        assert o.as_dict()["plain"].endswith("behind the aircraft ahead.") and f"runway {o.proposal.runway}" in o.as_dict()["plain"]


def test_the_no_hold_quick_reason_works(scen):
    res, _ = run(scen, [reject("no hold"), APPROVE1])
    assert res.outcome == "APPROVED" and res.proposal.hold_minutes == 0 and [c.describe() for c in res.constraints] == ["air hold ≤ 0 min"]


def test_the_add_60_seconds_quick_reason_works(scen):
    sep = build_options(consensus(scen.model_copy(deep=True)), scen.context)[0].proposal.separation_sec
    res, _ = run(scen, [reject(f"separation at least {sep + 60}"), APPROVE1])
    assert res.outcome == "APPROVED" and res.proposal.separation_sec >= sep + 60
    assert [c.describe() for c in res.constraints] == [f"separation ≥ {sep + 60} s"] and check_proposal(res.proposal, scen.context).passed


def test_both_quick_reasons_together_are_one_rejection(scen):
    sep = build_options(consensus(scen.model_copy(deep=True)), scen.context)[0].proposal.separation_sec
    res, decider = run(scen, [reject(f"no hold; separation at least {sep + 60}"), APPROVE1])
    assert res.rejections == 1 and res.proposal.hold_minutes == 0 and res.proposal.separation_sec >= sep + 60 and len(res.constraints) == 2


def test_the_page_has_the_quick_reason_buttons_and_the_plain_sentence():
    from negotiator.server import STATIC

    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for needed in ("No airborne hold", "Add 60 s separation", "separation at least ${", "Quick reasons", "addReason", "class: 'plain'", "o.plain"):
        assert needed in html, needed


def test_options_always_valid_under_random_scenarios():
    rng, base = random.Random(5), build_scenario(FIXTURE_CSV, 1, 1, kind="weather")
    made = Counter()
    for i in range(200):
        s = base.model_copy(deep=True)
        s.context = random_context(rng, benign=rng.random() < 0.6)
        res = Negotiation(s, FakeLLM()).run()
        if res.proposal is None:
            continue
        opts = build_options(res, s.context)
        made[len(opts)] += 1
        assert 1 <= len(opts) <= 2 and opts[0].id == res.proposal.proposal_id and len({o.id for o in opts}) == len(opts), i
        assert [o.rank for o in opts] == list(range(1, len(opts) + 1)), i
        assert all(check_proposal(o.proposal, s.context).passed for o in opts), i
    assert made[1] > 0 and made[2] > 0, made


# --- approving ------------------------------------------------------------------------------------------------


def test_approving_option_one(scen):
    res, decider = run(scen, [APPROVE1])
    assert res.outcome == "APPROVED" and res.option == 1 and res.negotiations == 1 and res.rejections == 0
    assert res.proposal.model_dump() == events(res, "OPTIONS")[0].data["options"][0]["proposal"] == events(res, "FINAL")[0].data["proposal"]
    assert sequence(res)[-4:] == ["CONSENSUS", "OPTIONS", "DECISION", "FINAL"] and res.log.verify()
    assert events(res, "FINAL")[0].ok is True and events(res, "FINAL")[0].data["outcome"] == "APPROVED"
    assert len(decider.prompts) == 1 and decider.prompts[0]["rejections_left"] == 2 and decider.prompts[0]["timeout_s"] == 60.0


def test_approving_option_two_uses_the_alternative(scen):
    res, _ = run(scen, [APPROVE2])
    opts = events(res, "OPTIONS")[0].data["options"]
    assert res.outcome == "APPROVED" and res.option == 2 and res.proposal.model_dump() == opts[1]["proposal"]
    assert res.proposal.hold_minutes == 0


def test_the_auto_approver_picks_option_one(scen):
    res = Session(scen, FakeLLM(), RuleParser(), AutoApprove()).run()
    assert res.outcome == "APPROVED" and res.option == 1


def test_controller_entries_carry_the_round_the_agents_finished_in(scen):
    res, _ = run(scen, [APPROVE1])
    last_round = max(e.round for e in res.log.entries if e.event == "PROPOSAL")
    assert last_round > 1  # the opening bid on this weather scenario is always rejected, so the agents need a second round
    for name in ("OPTIONS", "DECISION", "FINAL"):
        assert [e.round for e in events(res, name)] == [last_round], name


def test_decision_entries_say_whether_the_answer_was_usable(scen):
    approved, _ = run(scen, [APPROVE1])
    assert [e.ok for e in events(approved, "DECISION")] == [True]
    timed_out, _ = run(build_scenario(FIXTURE_CSV, 1, 1, kind="weather"), [])
    assert [e.ok for e in events(timed_out, "DECISION")] == [False] and events(timed_out, "FALLBACK")[0].ok is True
    mixed, _ = run(build_scenario(FIXTURE_CSV, 1, 1, kind="weather"), [Decision("approve", option=9), Decision("maybe"), reject("avoid gate A1"), APPROVE1])
    assert [e.ok for e in events(mixed, "DECISION")] == [False, False, True, True]
    assert "Choose an option to approve, or reject with a reason." in events(mixed, "DECISION")[1].note


def test_options_are_logged_with_their_rank_rule(scen):
    res, _ = run(scen, [APPROVE1])
    e = events(res, "OPTIONS")[0]
    assert e.actor == "ATC" and "1)" in e.note and "2)" in e.note and "re-checked by the validator" in e.data["rank_rule"]


# --- rejecting -----------------------------------------------------------------------------------------------------


def chosen_gate(scen):
    return consensus(scen.model_copy(deep=True)).proposal.gate


def test_a_rejection_becomes_a_rule_and_the_agents_negotiate_again(scen):
    gate = chosen_gate(scen)
    res, decider = run(scen, [reject(f"avoid gate {gate}"), APPROVE1])
    assert res.outcome == "APPROVED" and res.negotiations == 2 and res.rejections == 1
    assert res.proposal.gate != gate and check_proposal(res.proposal, scen.context).passed
    assert [c.describe() for c in res.constraints] == [f"gate ≠ {gate}"]
    assert sequence(res).count("RENEGOTIATE") == 1 and events(res, "CONSTRAINTS")[0].ok is True
    assert f"gate ≠ {gate}" in events(res, "CONSTRAINTS")[0].note and events(res, "RENEGOTIATE")[0].data["renegotiation"] == 1
    assert decider.prompts[1]["rules"] == [f"gate ≠ {gate}"] and decider.prompts[1]["rejections_left"] == 1


def test_the_rule_reaches_the_atc_agent_only_after_the_rejection(scen):
    class Recording(FakeLLM):
        def __init__(self):
            super().__init__()
            self.atc = []

        def complete(self, role, view):
            if role == Role.ATC:
                self.atc.append((view["round"], view["hard_constraints"]))
            return super().complete(role, view)

    llm, gate = Recording(), chosen_gate(scen)
    Session(scen, llm, RuleParser(), ScriptedDecider([reject(f"avoid gate {gate}"), APPROVE1])).run()
    first, later = llm.atc[0], llm.atc[-1]
    assert first[1] == [] and later[1] and later[1][0]["field"] == "gate" and later[1][0]["op"] == "!="


def test_rounds_restart_in_each_negotiation_and_attempts_are_marked(scen):
    res, _ = run(scen, [reject(f"avoid gate {chosen_gate(scen)}"), APPROVE1])
    attempts = [e.attempt for e in res.log.entries]
    assert attempts == sorted(attempts) and set(attempts) == {1, 2}
    second = [e for e in res.log.entries if e.attempt == 2 and e.event == "PROPOSAL"]
    assert second and second[0].round == 1  # a fresh negotiation starts again at round 1
    labels = [s.text.split(" ended")[0] for s in all_round_summaries(res.log.entries)]
    assert "Round 1" in labels and any(l.startswith("Renegotiation 1, round") for l in labels)
    assert round_label(2, 3) == "Renegotiation 2, round 2" and round_label(2) == "Round 2"


def test_two_rejections_are_allowed_and_each_renegotiates(scen):
    res, _ = run(scen, [reject("avoid gate A1"), reject("no hold"), APPROVE1])
    assert res.outcome == "APPROVED" and res.negotiations == 3 and res.rejections == 2
    assert res.proposal.hold_minutes == 0 and check_proposal(res.proposal, scen.context).passed
    assert len(res.constraints) == 2


def test_a_third_rejection_ends_in_the_fallback_which_still_obeys_the_rules(scen):
    res, decider = run(scen, [reject("avoid gate A1"), reject("no hold"), reject("avoid runway 27L")])
    assert res.outcome in ("REJECTED_FALLBACK", "ESCALATED") and res.negotiations == 3 and res.rejections == 3
    assert len(decider.prompts) == 3 and sequence(res).count("RENEGOTIATE") == 2  # only two renegotiations were allowed
    assert len(res.constraints) == 3
    if res.proposal is not None:  # the fallback is validator-checked against every rule the controller set
        assert check_proposal(res.proposal, scen.context).passed
        assert res.proposal.gate != "A1" and res.proposal.hold_minutes == 0 and res.proposal.runway != "27L"
        assert events(res, "FALLBACK") and "our conservative default procedure" in events(res, "FALLBACK")[0].note
    assert res.log.verify() and events(res, "FINAL")[0].data["rejections"] == 3


def test_rejections_allowed_follow_the_configured_limit(scen):
    scen.context.limits.max_human_rejections = 0
    res, decider = run(scen, [reject("avoid gate A1")])
    assert res.outcome in ("REJECTED_FALLBACK", "ESCALATED") and res.negotiations == 1 and sequence(res).count("RENEGOTIATE") == 0
    assert decider.prompts[0]["rejections_left"] == 0


def test_text_that_cannot_become_a_rule_does_not_use_up_a_rejection(scen):
    res, decider = run(scen, [reject("make it better"), APPROVE1])
    assert res.outcome == "APPROVED" and res.rejections == 0 and res.negotiations == 1 and res.constraints == []
    assert decider.prompts[1]["problem"] and "I did not understand" in decider.prompts[1]["problem"]
    assert decider.prompts[1]["rejections_left"] == decider.prompts[0]["rejections_left"] == 2
    assert events(res, "CONSTRAINTS")[0].ok is False


def test_endless_unclear_answers_end_in_the_fallback(scen):
    res, decider = run(scen, [reject("hmm")] * 10)
    assert res.outcome == "UNCLEAR_FALLBACK" and len(decider.prompts) == MAX_UNCLEAR and res.rejections == 0
    assert "could not be understood" in res.reason and res.proposal is not None and check_proposal(res.proposal, scen.context).passed


def test_a_choice_that_does_not_exist_is_refused_and_asked_again(scen):
    res, decider = run(scen, [Decision("approve", option=9), APPROVE1])
    assert res.outcome == "APPROVED" and "There is no option 9" in decider.prompts[1]["problem"]
    res, decider = run(build_scenario(FIXTURE_CSV, 1, 1, kind="weather"), [Decision("approve", option=9)] * 5)
    assert res.outcome == "UNCLEAR_FALLBACK" and len(decider.prompts) == MAX_UNCLEAR


def test_a_parser_that_always_fails_ends_in_the_fallback(scen):
    from negotiator.parsing import ParseResult

    class Broken:
        name = "broken"

        def parse(self, text, known):
            return ParseResult((), "The rule parser failed (TimeoutError); please rephrase or try again.", "broken")

    res, decider = run(scen, [reject("avoid gate A1")] * 5, parser=Broken())
    assert res.outcome == "UNCLEAR_FALLBACK" and len(decider.prompts) == MAX_UNCLEAR and res.rejections == 0


def test_the_parser_only_sees_values_that_exist_here(scen):
    res, decider = run(scen, [reject("avoid gate Z9"), APPROVE1])
    assert "Z9" in decider.prompts[1]["problem"] and res.rejections == 0


# --- timing out ------------------------------------------------------------------------------------------------------------


def test_no_decision_in_time_applies_the_conservative_default(scen):
    res, decider = run(scen, [])  # nobody answers
    assert res.outcome == "TIMEOUT_FALLBACK" and len(decider.prompts) == 1
    assert res.proposal is not None and check_proposal(res.proposal, scen.context).passed
    assert events(res, "DECISION")[0].data == {"action": "timeout"} and "did not decide within 60 s" in events(res, "DECISION")[0].note
    assert events(res, "FALLBACK") and sequence(res)[-2:] == ["FALLBACK", "FINAL"]


def test_timeout_after_a_rejection_still_respects_the_rule(scen):
    res, _ = run(scen, [reject("avoid gate A1"), Decision("timeout")])
    assert res.outcome == "TIMEOUT_FALLBACK" and res.negotiations == 2 and res.proposal.gate != "A1"


def test_the_timeout_comes_from_the_limits(scen):
    scen.context.limits.decision_timeout_s = 7
    _, decider = run(scen, [APPROVE1])
    assert decider.prompts[0]["timeout_s"] == 7.0


# --- when there is nothing to approve ----------------------------------------------------------------------------------------


def test_an_escalated_negotiation_never_asks_the_controller(scen):
    scen.context.runways = []  # nowhere safe to land
    res, decider = run(scen, [APPROVE1])
    assert res.outcome == "ESCALATED" and res.proposal is None and decider.prompts == []
    assert events(res, "FINAL")[0].data["outcome"] == "ESCALATED" and not events(res, "OPTIONS")


def test_a_rule_that_leaves_no_safe_plan_escalates(scen):
    res, _ = run(scen, [reject("use runway 27L"), reject("avoid runway 27L"), APPROVE1])
    assert res.outcome in ("ESCALATED", "APPROVED")  # contradictory rules: nothing can satisfy both
    if res.outcome == "ESCALATED":
        assert res.proposal is None


def test_the_validator_vetoes_an_approval_if_the_world_changed(scen):
    ctx = scen.context

    class ChangingDecider(ScriptedDecider):
        changed = None

        def decide(self, options, **kw):
            self.changed = options[0].proposal.runway  # a sensor update lands while the controller is deciding
            for r in ctx.runways:
                if r.runway == self.changed:
                    r.standing_water_mm = 9.0
            return super().decide(options, **kw)

    decider = ChangingDecider([APPROVE1])
    res = Session(scen, FakeLLM(), RuleParser(), decider).run()
    assert [e for e in res.log.entries if e.event == "VALIDATOR_VETO" and e.vetoed and e.ok is False]
    assert res.outcome == "VETOED_FALLBACK" and "failed the validator's re-check" in res.reason
    assert res.proposal.runway != decider.changed  # the fallback picked a runway that is still dry
    assert check_proposal(res.proposal, ctx).passed


# --- the script format -------------------------------------------------------------------------------------------------------------


def test_parse_script():
    assert parse_script("reject: avoid gate A1 | approve 2 | approve | timeout") == [
        Decision("reject", text="avoid gate A1"), Decision("approve", option=2), Decision("approve", option=1), Decision("timeout")]
    assert parse_script("Reject no hold") == [Decision("reject", text="no hold")]
    assert parse_script("") == []


@pytest.mark.parametrize("bad", ["maybe", "approve two", "time out"])
def test_parse_script_rejects_nonsense(bad):
    with pytest.raises(ValueError, match="cannot read decision"):
        parse_script(bad)


def test_scripted_decider_times_out_when_the_script_ends(scen):
    d = ScriptedDecider([])
    assert d.decide([], attempt=1, rejections_left=2, timeout_s=1, rules=[], problem=None) == Decision("timeout")


# --- the log ----------------------------------------------------------------------------------------------------------------------------


def test_controller_events_have_display_text(scen):
    res, _ = run(scen, [reject(f"avoid gate {chosen_gate(scen)}"), APPROVE1])
    views = {e.event: entry_view(e) for e in res.log.entries if e.event in ("OPTIONS", "DECISION", "CONSTRAINTS", "RENEGOTIATE", "FINAL")}
    assert views["OPTIONS"]["label"] == "Options for the controller" and views["OPTIONS"]["options"][0]["rank"] == 1
    assert views["DECISION"]["actor_label"] == "Controller" and views["FINAL"]["label"] == "Final decision"
    assert views["CONSTRAINTS"]["label"] == "Controller rules" and views["RENEGOTIATE"]["label"] == "Renegotiate"


def test_final_is_logged_once_and_last(scen):
    for decisions in ([APPROVE1], [], [reject("avoid gate A1"), APPROVE2], [reject("hmm")] * 3):
        res, _ = run(build_scenario(FIXTURE_CSV, 1, 1, kind="weather"), decisions)
        assert len(events(res, "FINAL")) == 1 and res.log.entries[-1].event == "FINAL" and res.log.verify()
        assert res.outcome in OUTCOMES and events(res, "FINAL")[0].data["outcome"] == res.outcome


# --- invariants under random scenarios and random controllers -------------------------------------------------------------------------------


RANDOM_TEXTS = ["avoid gate A1", "avoid gate B2", "no hold", "use runway 27L", "avoid runway 27L", "hold at most 5 minutes", "delay at most 60 minutes",
                "separation at least 200", "don't hold in the north", "use gate A1", "make it better", "asdf", "", "avoid gate Z9"]


def random_decisions(rng):
    out = []
    for _ in range(rng.randint(0, 6)):
        r = rng.random()
        out.append(reject(rng.choice(RANDOM_TEXTS)) if r < 0.55 else Decision("approve", option=rng.choice([1, 1, 2, 9])) if r < 0.9 else Decision("timeout"))
    return out


def test_session_invariants_under_random_scenarios_and_controllers():
    rng, base = random.Random(2468), build_scenario(FIXTURE_CSV, 1, 1, kind="weather")
    outcomes = Counter()
    for i in range(250):
        s = base.model_copy(deep=True)
        s.context = random_context(rng, benign=rng.random() < 0.65)
        s.context.limits.max_human_rejections = rng.randint(0, 3)
        decisions = random_decisions(rng)
        personas = {r.value: rng.choice(PERSONAS) for r in Role if rng.random() < 0.3}
        decider = ScriptedDecider(decisions)
        initial = len(s.context.hard_constraints)  # some random contexts start with rules of their own
        res = Session(s, FakeLLM(personas), RuleParser(), decider).run()
        ctx, max_rej = s.context, s.context.limits.max_human_rejections
        outcomes[res.outcome] += 1
        tag = f"run {i}: {res.outcome} decisions={decisions} personas={personas}"

        assert res.outcome in OUTCOMES, tag
        assert res.log.verify() and len(events(res, "FINAL")) == 1 and res.log.entries[-1].event == "FINAL", tag
        # bounded
        assert res.negotiations <= 1 + max_rej and res.rejections <= max_rej + 1, tag
        assert len(decider.prompts) <= (1 + max_rej) * MAX_UNCLEAR, tag
        assert sequence(res).count("RENEGOTIATE") == min(res.rejections, max_rej), tag
        assert sorted({e.attempt for e in res.log.entries}) == list(range(1, res.negotiations + 1)), tag
        assert len(res.constraints) == len(ctx.hard_constraints) and len(res.constraints) - initial <= MAX_CONSTRAINTS * res.rejections, tag
        # nothing unsafe or rule-breaking is ever the final answer
        if res.proposal is None:
            assert res.outcome == "ESCALATED", tag
        else:
            assert check_proposal(res.proposal, ctx).passed, tag  # includes every rule the controller added
        if res.outcome == "APPROVED":
            last = [e for e in events(res, "OPTIONS") if e.attempt == res.negotiations][0]
            assert res.proposal.model_dump() == last.data["options"][res.option - 1]["proposal"], tag
        if res.outcome == "ESCALATED":
            assert res.proposal is None, tag
    assert {"APPROVED", "TIMEOUT_FALLBACK", "ESCALATED"} <= set(outcomes) and len(outcomes) >= 5, outcomes  # the fuzz reaches most endings
