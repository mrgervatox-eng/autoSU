"""Run one negotiation in FAKE MODE (mock LLM, simulated sensors; no network, no phones).

  python -m negotiator.negotiate                                  # seed from config's CSV (or fixture)
  python -m negotiator.negotiate --kind weather --month 1 --day 1
  python -m negotiator.negotiate --scenario data/scenario.json
  python -m negotiator.negotiate --persona PILOT=stubborn         # force a failure -> fallback
  python -m negotiator.negotiate --log-out logs/run.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import ROOT, load_config
from .engine import Negotiation
from .fake_llm import PERSONAS, FakeLLM
from .scenario import FIXTURE_CSV, build_scenario, load_scenario, render_cascade


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run a fake-mode ATC / Pilot / Ground negotiation.")
    ap.add_argument("--scenario", default=None, help="scenario JSON written by negotiator.scenario")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--tail", default=None)
    ap.add_argument("--kind", choices=["weather", "late_aircraft"], default=None)
    ap.add_argument("--persona", action="append", default=[], metavar="ROLE=NAME",
                    help=f"make a fake agent misbehave; NAME in {', '.join(PERSONAS)}")
    ap.add_argument("--log-out", default=None, help="write the consensus log JSON here")
    args = ap.parse_args(argv)

    cfg = load_config()
    if not cfg.llm.fake_mode:
        print("config llm.fake_mode is false: real Claude agents arrive in Stage 4. Set fake_mode = true.", file=sys.stderr)
        return 2

    try:
        personas = dict(p.split("=", 1) for p in args.persona)
        llm = FakeLLM({k.upper(): v for k, v in personas.items()})
        if args.scenario:
            scenario = load_scenario(args.scenario)
        else:
            csv_path = Path(args.csv or ROOT / cfg.data.flights_csv)
            if not csv_path.is_file():
                print(f"[fake mode] {csv_path} not found, using fixture {FIXTURE_CSV}")
                csv_path = FIXTURE_CSV
            scenario = build_scenario(csv_path, args.month, args.day, cfg.margins, cfg.limits, args.tail, args.kind)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(render_cascade(scenario))
    print("\n--- negotiation (FAKE MODE) ---")
    result = Negotiation(scenario, llm).run()
    print(result.log.render())
    p = result.proposal
    print(f"\nRESULT: {result.outcome.value} - {result.reason}")
    if p is not None:
        print(f"  delay={p.delay_min} min, hold={p.hold_minutes} min in {p.holding_area}, separation={p.separation_sec} s, "
              f"gate={p.gate}, runway={p.runway}  (id {p.proposal_id})")
    print(f"log entries: {len(result.log.entries)}, hash chain valid: {result.log.verify()}")
    if args.log_out:
        result.log.save(args.log_out)
        print(f"log saved -> {args.log_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
