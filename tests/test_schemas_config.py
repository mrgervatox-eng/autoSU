import json

import pytest

from negotiator.config import load_config
from negotiator.schemas import parse_message

VALID = {
    "sender": "ATC",
    "round": 1,
    "type": "PROPOSE",
    "proposal": {"delay_min": 10, "hold_minutes": 0, "separation_sec": 150},
    "provenance": {"source": "llm", "confidence": 0.8},
}


def test_parse_valid_dict_and_json():
    assert parse_message(VALID)[0] is not None
    assert parse_message(json.dumps(VALID))[0] is not None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.update(extra_field=1),                       # unknown key
        lambda d: d.update(sender="TOWER"),                      # bad role
        lambda d: d.update(round=0),                             # bad round
        lambda d: d.pop("proposal"),                             # PROPOSE without proposal
        lambda d: d["proposal"].update(delay_min=-5),            # negative number
        lambda d: d["provenance"].update(confidence=1.5),        # out of range
        lambda d: d.update(type="ACCEPT", proposal=None),        # ACCEPT without proposal_id
    ],
)
def test_parse_rejects_bad_messages(mutate):
    data = json.loads(json.dumps(VALID))
    mutate(data)
    msg, err = parse_message(data)
    assert msg is None and err


def test_parse_rejects_non_json():
    msg, err = parse_message("not json at all")
    assert msg is None and err


def test_example_config_loads():
    cfg = load_config()
    assert cfg.llm.fake_mode is True
    assert cfg.limits.max_rounds == 3
    assert cfg.limits.max_human_rejections == 2
    assert cfg.margins.min_separation_sec == 120


def test_missing_explicit_config_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "nope.toml")
