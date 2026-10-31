# The validator: what it checks, with which numbers, and what it does not know

The validator is ordinary deterministic Python (`negotiator/validator.py`). No AI is involved.
It checks every proposal and every ACCEPT, and it can veto an agent. This page is the written
specification the tests are checked against.

## Status of the numbers

**Every default below is a simulated policy value chosen for this demo.** They are plausible in
order of magnitude for a narrow-body jet (about 45 kg/min trip burn, about 40 kg/min hold burn),
but they are **not real-world regulations or limits, and we do not cite any**. They live in
`config.toml` / `config.example.toml`, are range-checked when loaded, and can be changed per run.
A real deployment would take them from the operator's own approved procedures.

## The flight model (what is being decided)

The story: **an inbound flight into a disrupted airport.** The agents decide how much of the delay to
absorb **on the ground at the origin** versus **in the air near the destination**.

| Field | Where it happens |
|---|---|
| `delay_min` | on the ground at the origin, before departure |
| `hold_minutes`, `holding_area` | in the air, near the destination |
| `separation_sec`, `gate`, `runway` | at the destination airport |
| runway water, holding-area turbulence, free gates (simulated) | at the destination airport |

The fuel check follows from this: `fuel_on_board_kg` is the fuel **loaded at departure**, and it must
cover the trip, any airborne hold, and the reserve. That is why the formula subtracts the hold burn
and still requires the full trip fuel. Crew duty counts the ground delay, the hold and the flight itself.

This is a deliberate simplification. The disruption seeded from the 2015 data (for example a weather delay
on the first leg of a tail's day) tells us the flight is badly late; it does not say the weather is at the
destination. The destination conditions are **simulated** and labelled so. A flight that is already
airborne and near the destination would have much less trip fuel left to cover, and a departing flight
does not circle, so the model is not meant for those cases.

## Parameters (`[margins]` and `[limits]`)

| Parameter | Default | Allowed range | Meaning |
|---|---|---|---|
| `min_separation_sec` | 120 | 1 to 3600 | least separation (seconds) a proposal may declare |
| `fuel_reserve_kg` | 1800 | 0 and up | fuel that must remain after trip fuel and any hold |
| `max_hold_min` | 30 | 0 to 1440 | longest airborne hold (minutes) |
| `max_duty_min` | 780 | 1 to 1440 | crew duty limit (minutes), counting delay, hold and remaining flight |
| `max_turbulence_index` | 0.6 | above 0 to 1 | roughest air (0 to 1 proxy) a hold may be in |
| `max_standing_water_mm` | 3.0 | 0 and up | deepest standing water on an assigned runway |
| `min_observation_confidence` | 0.6 | 0 to 1 | a reading below this is not trusted until a human confirms it |
| `require_gate_and_runway` | true | true or false | a proposal must assign both a gate and a runway |
| `max_rounds` | 3 | 1 to 10 | negotiation rounds before the deterministic fallback |
| `max_human_rejections` | 2 | 0 to 5 | controller rejections before the fallback (Stage 6) |
| `decision_timeout_s` | 60 | 1 to 3600 | seconds the controller has to decide (Stage 6) |

An unknown key (for example a typo such as `min_seperation_sec`) or an out-of-range value makes
loading the config fail with an error. It is never silently ignored.

## The checks

Each proposal is judged against all of these. A proposal passes only if **every** check passes.

| Check | Passes when |
|---|---|
| `min_separation_sec` | `separation_sec >= min_separation_sec` |
| `max_hold_min` | `hold_minutes <= max_hold_min` |
| `fuel_reserve_kg` | `fuel_on_board - hold_burn_per_min * hold_minutes >= trip_fuel + fuel_reserve_kg` |
| `crew_duty_min` | `duty_used + delay_min + hold_minutes + flight_time_remaining <= max_duty_min` |
| `gate_available` | no gate named, or the gate is in the available list |
| `assignment_complete` | gate and runway are both assigned (when `require_gate_and_runway`) |
| `turbulence_at_holding_area` | no hold, or the holding area's readings pass the sensor rule below |
| `runway_standing_water` | no runway named, or the runway's readings pass the sensor rule below |
| `hard:...` | one per controller constraint (Stage 6): `field op value` holds on the proposal |

Every comparison is exact at the limit: equal to the limit passes, one unit beyond fails.

### Sensor rule (turbulence and runway), fail closed

Looking at **all** readings for the area or runway in question:

1. no reading at all: **fail** (silence is never safe);
2. **any** reading over the limit: **fail**, even a low-confidence one (a doubtful bad reading blocks);
3. no reading is trustworthy (all below `min_observation_confidence` and none human-confirmed): **fail**,
   it needs a human confirm;
4. otherwise: pass, judged on the worst reading.

A reading for a different area or runway never counts.

### What ACCEPT and REJECT must satisfy

- An ACCEPT must name the `proposal_id` of the proposal actually on the table, and that proposal is
  re-checked at that moment. Wrong id, nothing on the table, or a proposal that no longer passes
  (for example a sensor changed) means the ACCEPT is **vetoed**.
- Any message beyond `max_rounds` is refused.
- An agent's own `constraint_check` is never trusted. The validator overwrites it.

## What the validator does not know (limits of the claims)

- **Separation is declared, not derived.** There is no traffic model. The check is that the proposal
  *states* at least the minimum separation, not that a real schedule achieves it.
- **It does not check that the plan absorbs the disruption.** A plan with a short delay for a flight
  that is already very late can pass. It checks safety margins, not usefulness.
- **Parameters are simulated** (see above) and sensor values are simulated or come from a phone.
- **Sensor age is not checked** (no timestamps yet). This matters once phones supply live readings.
- **One aircraft only.** No conflicts with other traffic, weather forecast or airspace structure.
- It is a decision-support prototype, not certified, and not for operational use.

## How the validator itself is verified

`tests/test_validator_verification.py` (run with `python -m pytest`):

1. **Independent oracle**: the rules above are re-implemented separately in the test and compared with
   the validator on thousands of seeded random cases. The cases deliberately sit on and next to the
   limits, and the test fails if the fuzz does not exercise both outcomes of every check.
2. **Exact boundaries**: for each limit, at the limit passes, one unit beyond fails, and exactly one
   check flips.
3. **Properties**: tightening a margin never turns a failure into a pass, relaxing never turns a pass
   into a failure, a strictly safer proposal never regresses, missing data always fails closed.
4. **System invariants**: random scenarios with misbehaving agents (stubborn, reckless, garbage,
   impostor): no accepted result ever violates a margin, rounds and calls stay bounded, the log chain
   verifies, and consensus is declared only when both responders accepted the same proposal unvetoed.
5. **Mutation check** (`python tools/mutation_check.py`): deliberately breaks the safety code (validator,
   constraints, engine, fallback, and the plain-language round summaries), one operator at a time, and
   confirms the tests notice. Every comparison, arithmetic operator, and/or/not, True/False, max/min and
   any/all is flipped in turn. The first run killed 64 of 69 validator mutants and 17 of 30 engine and
   fallback mutants; the survivors exposed real test gaps (round-limit boundary, REJECT handling, the
   engine's last-moment re-check, the fallback's fuel buffer), and a later run on the summary code found
   more (failures exactly at a limit, wrong-role entries). All now have tests.
   Current result: **179 of 179 killed**.

Limit of this evidence: the oracle and the validator were written by the same team from the same
specification, so they share any misunderstanding of what the rules *should* be. The tests prove the
code does what this page says. Whether these are the right rules for real operations is a question
for domain experts, which is why the numbers are configurable and labelled simulated.
