import csv

import pytest

from negotiator.constraints import Limits, Margins
from negotiator.scenario import (
    FIXTURE_CSV,
    FlightLeg,
    build_context,
    build_scenario,
    hhmm_to_min,
    load_scenario,
    main,
    read_day,
    render_cascade,
    save_scenario,
)
from negotiator.schemas import Proposal
from negotiator.validator import check_proposal


def test_fixture_rows_have_dot_column_count():
    with open(FIXTURE_CSV, newline="") as fh:
        rows = list(csv.reader(fh))
    assert {len(r) for r in rows} == {31}


def test_time_parsing():
    assert hhmm_to_min("0005") == 5
    assert hhmm_to_min("1305") == 785
    assert hhmm_to_min("2400") == 1440
    assert hhmm_to_min("") is None


def test_read_day_filters_by_date():
    assert len(read_day(FIXTURE_CSV, 1, 1)) == 8
    assert len(read_day(FIXTURE_CSV, 1, 2)) == 1
    assert read_day(FIXTURE_CSV, 1, 3) == []


def test_seed_is_flight_with_longest_cascade():
    s = build_scenario(FIXTURE_CSV, 1, 1)
    assert s.seed.flight_number == "102"
    assert [l.flight_number for l in s.chain] == ["101", "102", "103", "104"]  # TAIL chain in time order
    assert s.seed_index == 1
    assert s.disruption.kind == "LATE_AIRCRAFT"
    c = s.cascade
    assert (c.legs_in_chain, c.legs_after_seed, c.delayed_after_seed) == (4, 2, 2)
    assert c.total_late_aircraft_delay_after_seed_min == 60


def test_tail_override_and_weather_kind():
    s = build_scenario(FIXTURE_CSV, 1, 1, tail="N222BB")
    assert s.seed.flight_number == "201"
    assert s.disruption.kind == "WEATHER" and s.disruption.delay_min == 25


def test_cancelled_and_tailless_flights_never_seed():
    with pytest.raises(ValueError):
        build_scenario(FIXTURE_CSV, 1, 1, tail="N333CC")  # only a cancelled leg
    s = build_scenario(FIXTURE_CSV, 1, 1)
    assert s.seed.flight_number not in ("301", "401")


def test_empty_day_raises():
    with pytest.raises(ValueError, match="no flights"):
        build_scenario(FIXTURE_CSV, 1, 3)


def test_enrichment_is_labelled_and_deterministic():
    a = build_scenario(FIXTURE_CSV, 1, 1)
    b = build_scenario(FIXTURE_CSV, 1, 1)
    assert a.model_dump_json() == b.model_dump_json()
    assert "SIMULATED" in a.enrichment_note
    assert "SIMULATED" in a.context.aircraft.enrichment_label
    assert all(o.provenance.source.value == "simulated" for o in a.context.turbulence + a.context.runways)


def _safe(ctx, hold=0, area="NORTH"):
    return Proposal(delay_min=0, hold_minutes=hold, separation_sec=ctx.margins.min_separation_sec,
                    holding_area=area if hold else None, gate=ctx.aircraft.available_gates[0], runway="27L")


def test_weather_scenario_has_trouble_but_a_safe_option():
    ctx = build_scenario(FIXTURE_CSV, 1, 1, tail="N222BB").context
    assert check_proposal(_safe(ctx), ctx).passed
    assert check_proposal(_safe(ctx, hold=10), ctx).passed
    assert not check_proposal(_safe(ctx, hold=10, area="SOUTH"), ctx).passed  # rough air
    wet = _safe(ctx).model_copy(update={"runway": "09R"})
    assert not check_proposal(wet, ctx).passed  # standing water


@pytest.mark.parametrize("sched_time", [None, 45, 120, 300, 515])
@pytest.mark.parametrize("kind", ["WEATHER", "LATE_AIRCRAFT"])
def test_a_safe_resolution_always_exists(sched_time, kind):
    for n in range(25):  # different seeds -> different random draws
        leg = FlightLeg(date="2015-01-01", airline="AA", flight_number=str(n), tail_number=f"N{n}",
                        origin="A", destination="B", scheduled_departure_min=0, scheduled_time_min=sched_time,
                        air_time_min=None if sched_time is None else sched_time - 20)
        ctx = build_context(leg, kind, Margins(), Limits())
        assert check_proposal(_safe(ctx), ctx).passed
        assert check_proposal(_safe(ctx, hold=10), ctx).passed


def test_save_load_roundtrip(tmp_path):
    s = build_scenario(FIXTURE_CSV, 1, 1)
    save_scenario(s, tmp_path / "sub" / "s.json")
    assert load_scenario(tmp_path / "sub" / "s.json") == s


def test_render_and_cli(tmp_path, capsys):
    out = tmp_path / "scenario.json"
    assert main(["--csv", str(FIXTURE_CSV), "--month", "1", "--day", "1", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert ">>" in text and "SIMULATED" in text and "N111AA" in text
    assert out.is_file()
    assert "Cascade: 2 later leg(s)" in render_cascade(load_scenario(out))


def test_cli_reports_error_for_empty_day(capsys):
    assert main(["--csv", str(FIXTURE_CSV), "--day", "3", "--no-save"]) == 1
    assert "no flights" in capsys.readouterr().err
