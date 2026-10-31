"""Stage 2: seed a disruption scenario from the 2015 DOT flights CSV.

Real data: the flight legs and their delays.
SIMULATED (scenario-enrichment layer): fuel, gates, crew hours and sensor readings.

CLI:  python -m negotiator.scenario --month 1 --day 1 [--tail N376AA] [--csv PATH] [--out PATH]
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path
from typing import Literal

from pydantic import Field

from .config import ROOT, load_config
from .constraints import (
    AircraftState,
    ConstraintContext,
    Limits,
    Margins,
    RunwayObservation,
    TurbulenceObservation,
)
from .schemas import Provenance, Source, Strict

ENRICHMENT_NOTE = (
    "Flight legs and delays come from the 2015 DOT flights data. Fuel, gates, crew hours and "
    "sensor readings are SIMULATED (scenario-enrichment layer)."
)
FIXTURE_CSV = ROOT / "tests" / "fixtures" / "mini_flights.csv"

TRIP_BURN_KG_PER_MIN = 45.0
HOLD_BURN_KG_PER_MIN = 40.0
GATE_POOL = [f"{c}{n}" for c in "AB" for n in range(1, 9)]


class FlightLeg(Strict):
    date: str  # YYYY-MM-DD
    airline: str
    flight_number: str
    tail_number: str
    origin: str
    destination: str
    scheduled_departure_min: int
    scheduled_time_min: int | None = None
    air_time_min: int | None = None
    departure_delay_min: int | None = None
    arrival_delay_min: int | None = None
    weather_delay_min: int = 0
    late_aircraft_delay_min: int = 0
    cancelled: bool = False
    diverted: bool = False

    @property
    def label(self) -> str:
        return f"{self.airline}{self.flight_number}"


class Disruption(Strict):
    kind: Literal["WEATHER", "LATE_AIRCRAFT"]
    delay_min: int
    description: str


class CascadeSummary(Strict):
    legs_in_chain: int
    legs_after_seed: int
    delayed_after_seed: int
    total_late_aircraft_delay_after_seed_min: int


class Scenario(Strict):
    scenario_id: str
    provenance: Provenance
    enrichment_note: str = ENRICHMENT_NOTE
    seed_index: int
    chain: list[FlightLeg]
    disruption: Disruption
    cascade: CascadeSummary
    context: ConstraintContext = Field(description="margins, limits, simulated aircraft state, simulated sensors")

    @property
    def seed(self) -> FlightLeg:
        return self.chain[self.seed_index]


# --- parsing --------------------------------------------------------------


def _int(value: str | None) -> int | None:
    if value is None or value.strip() == "":
        return None
    return int(float(value))


def hhmm_to_min(value: str | None) -> int | None:
    n = _int(value)
    return None if n is None else (n // 100) * 60 + n % 100


def min_to_hhmm(minutes: int) -> str:
    return f"{minutes // 60:02d}{minutes % 60:02d}"


def _leg(row: dict[str, str]) -> FlightLeg:
    return FlightLeg(
        date=f"{int(row['YEAR']):04d}-{int(row['MONTH']):02d}-{int(row['DAY']):02d}",
        airline=row["AIRLINE"],
        flight_number=row["FLIGHT_NUMBER"],
        tail_number=row["TAIL_NUMBER"].strip(),
        origin=row["ORIGIN_AIRPORT"],
        destination=row["DESTINATION_AIRPORT"],
        scheduled_departure_min=hhmm_to_min(row["SCHEDULED_DEPARTURE"]) or 0,
        scheduled_time_min=_int(row["SCHEDULED_TIME"]),
        air_time_min=_int(row["AIR_TIME"]),
        departure_delay_min=_int(row["DEPARTURE_DELAY"]),
        arrival_delay_min=_int(row["ARRIVAL_DELAY"]),
        weather_delay_min=_int(row["WEATHER_DELAY"]) or 0,
        late_aircraft_delay_min=_int(row["LATE_AIRCRAFT_DELAY"]) or 0,
        cancelled=row["CANCELLED"].strip() in ("1", "1.0"),
        diverted=row["DIVERTED"].strip() in ("1", "1.0"),
    )


def read_day(csv_path: str | Path, month: int, day: int) -> list[FlightLeg]:
    """Stream the CSV and keep one date. Stops early once past it (the DOT file is date-sorted)."""
    legs: list[FlightLeg] = []
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            key = (int(row["MONTH"]), int(row["DAY"]))
            if key == (month, day):
                legs.append(_leg(row))
            elif legs and key > (month, day):
                break
    return legs


# --- seed selection and cascade ------------------------------------------


def is_disrupted(leg: FlightLeg) -> bool:
    """Seed criterion from the brief; cancelled/diverted legs and legs with no tail can't seed a chain."""
    return (
        (leg.weather_delay_min > 15 or leg.late_aircraft_delay_min > 20)
        and not leg.cancelled
        and not leg.diverted
        and leg.tail_number != ""
    )


def tail_chain(legs: list[FlightLeg], tail: str) -> list[FlightLeg]:
    return sorted((l for l in legs if l.tail_number == tail), key=lambda l: (l.scheduled_departure_min, l.flight_number))


def _delayed_after(chain: list[FlightLeg], index: int) -> int:
    return sum(1 for l in chain[index + 1 :] if (l.departure_delay_min or 0) > 0)


def pick_seed(legs: list[FlightLeg], tail: str | None = None) -> tuple[list[FlightLeg], int]:
    """Return (tail chain, seed index). Most delayed legs downstream wins, then biggest delay."""
    best: tuple[tuple[int, int], list[FlightLeg], int] | None = None
    for t in sorted({l.tail_number for l in legs if l.tail_number and (tail is None or l.tail_number == tail)}):
        chain = tail_chain(legs, t)
        for i, leg in enumerate(chain):
            if not is_disrupted(leg):
                continue
            score = (_delayed_after(chain, i), max(leg.weather_delay_min, leg.late_aircraft_delay_min))
            if best is None or score > best[0]:
                best = (score, chain, i)
    if best is None:
        raise ValueError("no flight meets WEATHER_DELAY > 15 or LATE_AIRCRAFT_DELAY > 20" + (f" for tail {tail}" if tail else ""))
    return best[1], best[2]


def summarize_cascade(chain: list[FlightLeg], seed_index: int) -> CascadeSummary:
    later = chain[seed_index + 1 :]
    return CascadeSummary(
        legs_in_chain=len(chain),
        legs_after_seed=len(later),
        delayed_after_seed=_delayed_after(chain, seed_index),
        total_late_aircraft_delay_after_seed_min=sum(l.late_aircraft_delay_min for l in later),
    )


def describe_disruption(seed: FlightLeg) -> Disruption:
    if seed.weather_delay_min > 15:
        return Disruption(kind="WEATHER", delay_min=seed.weather_delay_min,
                          description=f"{seed.label} {seed.origin}->{seed.destination}: {seed.weather_delay_min} min weather delay")
    return Disruption(kind="LATE_AIRCRAFT", delay_min=seed.late_aircraft_delay_min,
                      description=f"{seed.label} {seed.origin}->{seed.destination}: {seed.late_aircraft_delay_min} min late-aircraft delay")


# --- simulated enrichment layer ------------------------------------------


def build_context(seed: FlightLeg, kind: str, margins: Margins, limits: Limits) -> ConstraintContext:
    """Deterministic (seeded by the flight) simulated aircraft state and sensor readings.

    Designed so a safe resolution always exists: zero hold fits fuel and crew duty, and one
    gate, one runway and one holding area are always acceptable.
    """
    rng = random.Random(f"{seed.date}|{seed.tail_number}|{seed.flight_number}")
    flight_time = seed.scheduled_time_min or 120
    trip_fuel = round((seed.air_time_min or flight_time) * TRIP_BURN_KG_PER_MIN)
    fuel = trip_fuel + margins.fuel_reserve_kg + rng.randint(600, 1400)  # ~15-35 min of hold in spare

    duty_hi = max(margins.max_duty_min - flight_time - 90, 0)
    duty_used = rng.randint(min(300, duty_hi), duty_hi)

    weather = kind == "WEATHER"
    calm = lambda: round(rng.uniform(0.1, 0.4), 2)  # noqa: E731
    rough = lambda: round(rng.uniform(0.75, 0.9), 2)  # noqa: E731
    sim = Provenance(source=Source.SIMULATED, confidence=0.9, note="simulated sensor (fake mode)")

    return ConstraintContext(
        margins=margins,
        limits=limits,
        aircraft=AircraftState(
            tail_number=seed.tail_number,
            fuel_on_board_kg=float(fuel),
            trip_fuel_kg=float(trip_fuel),
            hold_burn_kg_per_min=HOLD_BURN_KG_PER_MIN,
            crew_duty_used_min=duty_used,
            flight_time_remaining_min=flight_time,
            available_gates=sorted(rng.sample(GATE_POOL, 4)),
        ),
        turbulence=[
            TurbulenceObservation(area="NORTH", index=calm(), provenance=sim),
            TurbulenceObservation(area="SOUTH", index=rough() if weather else calm(), provenance=sim),
        ],
        runways=[
            RunwayObservation(runway="27L", standing_water_mm=round(rng.uniform(0.5, 2.5), 1), provenance=sim),
            RunwayObservation(runway="09R", standing_water_mm=round(rng.uniform(5.0, 9.0), 1) if weather else round(rng.uniform(0.5, 2.5), 1), provenance=sim),
        ],
    )


# --- build / save / load / render ----------------------------------------


def build_scenario(csv_path: str | Path, month: int, day: int, margins: Margins | None = None,
                   limits: Limits | None = None, tail: str | None = None) -> Scenario:
    legs = read_day(csv_path, month, day)
    if not legs:
        raise ValueError(f"no flights found for month={month} day={day} in {csv_path}")
    chain, idx = pick_seed(legs, tail)
    seed = chain[idx]
    disruption = describe_disruption(seed)
    return Scenario(
        scenario_id=f"{seed.date}-{seed.tail_number}-{seed.flight_number}",
        provenance=Provenance(source=Source.DATASET, confidence=1.0, note="2015 DOT flights CSV"),
        seed_index=idx,
        chain=chain,
        disruption=disruption,
        cascade=summarize_cascade(chain, idx),
        context=build_context(seed, disruption.kind, margins or Margins(), limits or Limits()),
    )


def save_scenario(scenario: Scenario, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(scenario.model_dump_json(indent=2), encoding="utf-8")


def load_scenario(path: str | Path) -> Scenario:
    return Scenario.model_validate_json(Path(path).read_text(encoding="utf-8"))


def render_cascade(s: Scenario) -> str:
    c, a = s.cascade, s.context.aircraft
    lines = [
        f"SCENARIO {s.scenario_id}",
        f"  {s.enrichment_note}",
        f"Disruption ({s.disruption.kind}): {s.disruption.description}",
        f"Tail {s.seed.tail_number} chain on {s.seed.date} (>> = disrupted leg):",
    ]
    for i, leg in enumerate(s.chain):
        mark = ">>" if i == s.seed_index else "  "
        dep = "n/a" if leg.departure_delay_min is None else f"{leg.departure_delay_min:+d}"
        flags = " CANCELLED" if leg.cancelled else ""
        lines.append(
            f" {mark} {min_to_hhmm(leg.scheduled_departure_min)} {leg.label:<8} {leg.origin}->{leg.destination}"
            f"  dep {dep:>4}  late-aircraft {leg.late_aircraft_delay_min:>3}  weather {leg.weather_delay_min:>3}{flags}"
        )
    lines += [
        f"Cascade: {c.legs_after_seed} later leg(s), {c.delayed_after_seed} delayed, "
        f"{c.total_late_aircraft_delay_after_seed_min} min late-aircraft delay downstream",
        f"[SIMULATED] fuel {a.fuel_on_board_kg:.0f} kg (trip {a.trip_fuel_kg:.0f}), crew duty used {a.crew_duty_used_min} min, "
        f"gates {','.join(a.available_gates)}",
        "[SIMULATED] turbulence " + ", ".join(f"{o.area}={o.index}" for o in s.context.turbulence)
        + " | runway water mm " + ", ".join(f"{o.runway}={o.standing_water_mm}" for o in s.context.runways),
    ]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Seed a disruption scenario from the 2015 DOT flights CSV.")
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--tail", default=None, help="restrict seed to this TAIL_NUMBER")
    ap.add_argument("--csv", default=None, help="flights CSV (default: config data.flights_csv)")
    ap.add_argument("--out", default=None, help="where to write scenario JSON (default: config data.scenario_path)")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config()
    csv_path = Path(args.csv or ROOT / cfg.data.flights_csv)
    if not csv_path.is_file():
        print(f"[fake mode] {csv_path} not found, using fixture {FIXTURE_CSV}")
        csv_path = FIXTURE_CSV
    try:
        scenario = build_scenario(csv_path, args.month, args.day, cfg.margins, cfg.limits, args.tail)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(render_cascade(scenario))
    if not args.no_save:
        out = Path(args.out or ROOT / cfg.data.scenario_path)
        save_scenario(scenario, out)
        print(f"saved -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
