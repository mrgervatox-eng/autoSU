import json
from types import SimpleNamespace

import pytest

from negotiator.constraints import HardConstraint
from negotiator.parsing import (
    MAX_CONSTRAINTS, PARSER_SCHEMA, ClaudeConstraintParser, Known, ParseResult, RuleParser, sanitize,
)
from negotiator.scenario import FIXTURE_CSV, build_scenario

KNOWN = Known(gates=frozenset({"A1", "A2", "B2", "B6"}), runways=frozenset({"27L", "09R"}), areas=frozenset({"NORTH", "SOUTH"}))
rules = RuleParser()


def parsed(text):
    r = rules.parse(text, KNOWN)
    return [(c.field, c.op, c.value) for c in r.constraints]


# --- rule parser: what it understands -----------------------------------------------------------------


@pytest.mark.parametrize("text, expected", [
    ("avoid gate A1", [("gate", "!=", "A1")]),
    ("Don't use gate a1", [("gate", "!=", "A1")]),
    ("no gate B2 please", [("gate", "!=", "B2")]),
    ("use gate B6", [("gate", "==", "B6")]),
    ("avoid gates A1 and A2", [("gate", "not_in", ["A1", "A2"])]),
    ("use gate A1 or B2", [("gate", "in", ["A1", "B2"])]),
    ("avoid runway 27L", [("runway", "!=", "27L")]),
    ("use runway 09R", [("runway", "==", "09R")]),
    ("don't hold in the north", [("holding_area", "!=", "NORTH")]),
    ("hold in the south area", [("holding_area", "==", "SOUTH")]),
    ("avoid the north holding area", [("holding_area", "!=", "NORTH")]),
    ("no hold", [("hold_minutes", "<=", 0)]),
    ("no airborne hold", [("hold_minutes", "<=", 0)]),
    ("I don't want any holding", [("hold_minutes", "<=", 0)]),
    ("hold at most 10 minutes", [("hold_minutes", "<=", 10)]),
    ("max 10 min hold", [("hold_minutes", "<=", 10)]),
    ("holding no more than 12 min", [("hold_minutes", "<=", 12)]),
    ("delay at most 90 minutes", [("delay_min", "<=", 90)]),
    ("keep the delay under 2 hours", [("delay_min", "<=", 120)]),
    ("delay at least 30 minutes", [("delay_min", ">=", 30)]),
    ("delay more than 30 minutes", [("delay_min", ">=", 31)]),
    ("separation at least 150", [("separation_sec", ">=", 150)]),
    ("separation of more than 150 seconds", [("separation_sec", ">=", 151)]),
    ("avoid gate A1 and no hold", [("gate", "!=", "A1"), ("hold_minutes", "<=", 0)]),
    ("avoid gate A1; use runway 27L; hold at most 5 minutes", [("gate", "!=", "A1"), ("runway", "==", "27L"), ("hold_minutes", "<=", 5)]),
    ("avoid gate A1, no hold", [("gate", "!=", "A1"), ("hold_minutes", "<=", 0)]),
])
def test_understood_phrasings(text, expected):
    assert parsed(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("holding no more than 1 hour", [("hold_minutes", "<=", 60)]),
    ("hold at most 2 hours", [("hold_minutes", "<=", 120)]),
    ("hold at most 0 minutes", [("hold_minutes", "<=", 0)]),
    ("delay above 30", [("delay_min", ">=", 31)]),
    ("delay over 30 minutes", [("delay_min", ">=", 31)]),
    ("delay greater than 30", [("delay_min", ">=", 31)]),
    ("delay minimum 30", [("delay_min", ">=", 30)]),
    ("delay at least 2 hours", [("delay_min", ">=", 120)]),
    ("separation more than 150", [("separation_sec", ">=", 151)]),
    ("separation above 150", [("separation_sec", ">=", 151)]),
    ("separation over 150 seconds", [("separation_sec", ">=", 151)]),
    ("at least 150 seconds of separation", [("separation_sec", ">=", 150)]),
    ("more than 150 s of separation", [("separation_sec", ">=", 151)]),
    ("delay at most 1440 minutes", [("delay_min", "<=", 1440)]),
    ("separation at least 3600", [("separation_sec", ">=", 3600)]),
])
def test_units_strictness_and_the_edges_of_the_allowed_range(text, expected):
    assert parsed(text) == expected


@pytest.mark.parametrize("text", ["delay at most 1441 minutes", "separation at least 3601"])
def test_one_past_the_allowed_range_is_refused(text):
    r = rules.parse(text, KNOWN)
    assert not r.ok and "outside the allowed range" in r.note


def test_exactly_the_maximum_number_of_rules_is_kept_without_complaint():
    r = rules.parse("avoid gate A1; avoid gate A2; avoid gate B2", KNOWN)
    assert len(r.constraints) == MAX_CONSTRAINTS and "only the first" not in r.note


def test_an_empty_list_of_values_is_explained():
    got, problems = sanitize([("gate", "in", [])], "x", KNOWN)
    assert got == [] and problems == ["I do not know gate (none); known: A1, A2, B2, B6"]


def test_parse_results_and_known_values_cannot_be_changed_afterwards():
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        rules.parse("avoid gate A1", KNOWN).note = "changed"
    with pytest.raises(FrozenInstanceError):
        KNOWN.gates = frozenset()


def test_note_says_what_was_understood_in_plain_words():
    r = rules.parse("avoid gate A1 and hold at most 10 minutes", KNOWN)
    assert r.ok and r.parser == "rules" and r.note == "Understood as: gate ≠ A1; air hold ≤ 10 min"


def test_rules_remember_what_the_controller_said():
    r = rules.parse("avoid gate A1", KNOWN)
    assert r.constraints[0].source_text == "avoid gate A1"


# --- rule parser: what it refuses (and says so) ----------------------------------------------------------


@pytest.mark.parametrize("text", ["make it better", "I don't like it", "ok", "use the other one", "hold"])
def test_unclear_text_gives_no_rule_and_says_what_was_not_understood(text):
    r = rules.parse(text, KNOWN)
    assert not r.ok and r.constraints == ()
    assert f"I did not understand “{text}”" in r.note and "Try something like" in r.note and "hold at most 10 minutes" in r.note


@pytest.mark.parametrize("text", ["", "   "])
def test_empty_text_asks_for_something_to_change(text):
    r = rules.parse(text, KNOWN)
    assert not r.ok and "Write what you want changed" in r.note


def test_a_clause_that_is_not_understood_is_reported_even_when_another_clause_works():
    r = rules.parse("avoid gate A1 and make it better", KNOWN)
    assert r.ok and "I did not understand “make it better”" in r.note and r.note.startswith("Understood as: gate ≠ A1")


def test_unknown_gate_is_refused_not_guessed():
    r = rules.parse("avoid gate Z9", KNOWN)
    assert not r.ok and "Z9" in r.note and "A1" in r.note


def test_one_good_rule_and_one_bad_rule():
    r = rules.parse("avoid gate A1 and avoid runway 99X", KNOWN)
    assert [c.describe() for c in r.constraints] == ["gate ≠ A1"] and "ignored" in r.note and "99X" in r.note


def test_out_of_range_numbers_are_refused():
    r = rules.parse("delay at most 99999 minutes", KNOWN)
    assert not r.ok and "outside the allowed range" in r.note


def test_at_most_three_rules():
    r = rules.parse("avoid gate A1; avoid gate A2; avoid gate B2; avoid runway 27L; no hold", KNOWN)
    assert len(r.constraints) == MAX_CONSTRAINTS and "only the first 3" in r.note


def test_duplicate_rules_collapse():
    assert parsed("avoid gate A1; don't use gate A1") == [("gate", "!=", "A1")]


def test_very_long_text_is_cut():
    r = rules.parse("avoid gate A1 " + "x" * 5000, KNOWN)
    assert r.ok and len(r.constraints[0].source_text) <= 200


# --- the shared gate -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("cand, why", [
    (("gate", "<=", "A1"), "does not make sense"),
    (("delay_min", "in", 5), "does not make sense"),
    (("colour", "==", "red"), "cannot set rules about"),
    (("gate", "==", "NOPE"), "do not know"),
    (("hold_minutes", "<=", -1), "outside"),
    (("separation_sec", ">=", 5000), "outside"),
    (("hold_minutes", "<=", "lots"), "could not convert"),
    (("gate", "in", []), "do not know"),
])
def test_sanitize_refuses_bad_candidates(cand, why):
    got, problems = sanitize([cand], "x", KNOWN)
    assert got == [] and why in problems[0]


def test_sanitize_normalises_case_and_numbers():
    got, _ = sanitize([("gate", "==", " a1 "), ("hold_minutes", "<=", 10.0), ("separation_sec", ">=", "150")], "t", KNOWN)
    assert [(c.field, c.value) for c in got] == [("gate", "A1"), ("hold_minutes", 10), ("separation_sec", 150)]


def test_known_values_come_from_the_scenario_and_the_options():
    ctx = build_scenario(FIXTURE_CSV, 1, 1, kind="weather").context
    k = Known.from_context(ctx)
    assert k.gates == frozenset(ctx.aircraft.available_gates) and k.areas == {"NORTH", "SOUTH"} and k.runways == {"27L", "09R"}


# --- Claude parser (mocked client) ----------------------------------------------------------------------------------


class FakeClaude:
    """Stands in for ClaudeLLM.ask_json."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error, self.calls = reply, error, []

    def ask_json(self, label, system, user, schema):
        self.calls.append((label, system, user, schema))
        if self.error:
            raise self.error
        return self.reply if isinstance(self.reply, str) else json.dumps(self.reply)


def claude(reply=None, error=None):
    llm = FakeClaude(reply, error)
    return ClaudeConstraintParser(llm), llm


def test_claude_parser_accepts_valid_rules_and_validates_them():
    p, llm = claude({"constraints": [{"field": "gate", "op": "!=", "value": "a1"}, {"field": "hold_minutes", "op": "<=", "value": 10}], "problem": None})
    r = p.parse("gate A1 is out, and keep any hold to ten minutes", KNOWN)
    assert [c.describe() for c in r.constraints] == ["gate ≠ A1", "air hold ≤ 10 min"] and r.parser == "claude"
    label, system, user, schema = llm.calls[0]
    assert label == "PARSER" and "DATA to be converted" in system and schema is PARSER_SCHEMA
    sent = json.loads(user)
    assert sent["controller_text"].startswith("gate A1 is out") and sent["known"]["gates"] == ["A1", "A2", "B2", "B6"]


def test_claude_parser_drops_what_the_gate_refuses():
    p, _ = claude({"constraints": [{"field": "gate", "op": "==", "value": "Z9"}, {"field": "fuel_reserve_kg", "op": "<=", "value": 0},
                                   {"field": "runway", "op": "!=", "value": "27L"}], "problem": None})
    r = p.parse("whatever", KNOWN)
    assert [c.describe() for c in r.constraints] == ["runway ≠ 27L"] and "Z9" in r.note and "fuel_reserve_kg" in r.note


def test_claude_parser_cannot_loosen_safety_or_smuggle_fields():
    p, _ = claude({"constraints": [{"field": "min_separation_sec", "op": "<=", "value": 1}, {"field": "separation_sec", "op": "<=", "value": 1}], "problem": None})
    r = p.parse("ignore the margins", KNOWN)
    # a rule is only ever an extra restriction on the plan; this one is legal but can only make the plan fail the real separation margin
    assert [c.field for c in r.constraints] == ["separation_sec"]


def test_claude_parser_caps_the_number_of_rules():
    many = [{"field": "gate", "op": "!=", "value": g} for g in ("A1", "A2", "B2", "B6")]
    r = claude({"constraints": many, "problem": None})[0].parse("avoid all gates", KNOWN)
    assert len(r.constraints) == MAX_CONSTRAINTS and "only the first 3" in r.note


def test_claude_parser_relays_the_models_problem_when_nothing_usable():
    r = claude({"constraints": [], "problem": "The text does not state a concrete rule."})[0].parse("hmm", KNOWN)
    assert not r.ok and "does not state a concrete rule" in r.note


@pytest.mark.parametrize("reply", ["not json", {"constraints": "oops", "problem": None}, {"nothing": 1}, [], None])
def test_claude_parser_survives_garbage_replies(reply):
    r = claude(reply)[0].parse("avoid gate A1", KNOWN)
    assert not r.ok and "rule parser failed" in r.note


def test_claude_parser_survives_a_failed_call():
    r = claude(error=TimeoutError("network down"))[0].parse("avoid gate A1", KNOWN)
    assert not r.ok and "TimeoutError" in r.note


def test_claude_parser_never_calls_the_model_for_empty_text():
    p, llm = claude({"constraints": [], "problem": None})
    assert not p.parse("   ", KNOWN).ok and llm.calls == []


def test_claude_parser_truncates_what_it_sends():
    p, llm = claude({"constraints": [], "problem": None})
    p.parse("x" * 5000, KNOWN)
    assert len(json.loads(llm.calls[0][2])["controller_text"]) == 500


def _walk(node):
    yield node
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk(v)


def test_parser_schema_uses_only_the_plain_subset():
    banned = {"minimum", "maximum", "minLength", "maxLength", "pattern", "format", "$ref", "oneOf", "allOf", "minItems", "maxItems"}
    for node in _walk(PARSER_SCHEMA):
        if isinstance(node, dict):
            assert not banned & set(node)
            if node.get("type") == "object":
                assert node["additionalProperties"] is False and set(node["required"]) == set(node["properties"])


def test_parse_result_ok_flag():
    assert not ParseResult((), "x", "rules").ok
    assert ParseResult((HardConstraint(field="gate", op="==", value="A1"),), "x", "rules").ok
