"""Run one negotiation. FAKE MODE by default (mock LLM, simulated sensors; no network, no phones).

  python -m negotiator.negotiate                                  # fake mode, config's CSV (or fixture)
  python -m negotiator.negotiate --kind weather --month 1 --day 1
  python -m negotiator.negotiate --scenario data/scenario.json
  python -m negotiator.negotiate --persona PILOT=stubborn         # fake mode: force a failure -> fallback
  python -m negotiator.negotiate --live                           # real Claude agents (needs an API key)
  python -m negotiator.negotiate --log-out logs/run.json

Real agents are used when config llm.fake_mode = false, or with --live.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import ROOT, load_config
from .engine import Negotiation
from .fake_llm import PERSONAS, FakeLLM
from .scenario import FIXTURE_CSV, build_scenario, load_scenario, render_cascade
from .summary import render_with_rounds


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run an ATC / Pilot / Ground negotiation.")
    ap.add_argument("--scenario", default=None, help="scenario JSON written by negotiator.scenario")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--month", type=int, default=1)
    ap.add_argument("--day", type=int, default=1)
    ap.add_argument("--tail", default=None)
    ap.add_argument("--kind", choices=["weather", "late_aircraft"], default=None)
    ap.add_argument("--live", action="store_true", help="use real Claude agents even if config fake_mode is true")
    ap.add_argument("--persona", action="append", default=[], metavar="ROLE=NAME",
                    help=f"fake mode only: make a mock agent misbehave; NAME in {', '.join(PERSONAS)}")
    ap.add_argument("--log-out", default=None, help="write the consensus log JSON here (live mode also writes <name>.llm_calls.json)")
    args = ap.parse_args(argv)

    cfg = load_config()
    live = args.live or not cfg.llm.fake_mode
    if live and args.persona:
        print("error: --persona only works in fake mode", file=sys.stderr)
        return 1

    try:
        if live:
            from .claude_llm import ClaudeLLM  # imported lazily so fake mode never needs the anthropic package

            llm = ClaudeLLM(cfg.llm)
        else:
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
    except Exception as exc:  # noqa: BLE001 - e.g. no API credentials found
        print(f"error: could not start the agents: {exc}", file=sys.stderr)
        return 2

    mode = f"LIVE: {cfg.llm.model}" if live else "FAKE MODE"
    print(render_cascade(scenario))
    print(f"\n--- negotiation ({mode}) ---")
    result = Negotiation(scenario, llm).run()
    print(render_with_rounds(result.log))
    p = result.proposal
    print(f"\nRESULT: {result.outcome.value} - {result.reason}")
    if p is not None:
        print(f"  delay={p.delay_min} min, hold={p.hold_minutes} min in {p.holding_area}, separation={p.separation_sec} s, "
              f"gate={p.gate}, runway={p.runway}  (id {p.proposal_id})")
    print(f"log entries: {len(result.log.entries)}, hash chain valid: {result.log.verify()}")

    if live:
        print(f"LLM usage: {llm.usage_summary()}")
        failures = len(result.log.events("AGENT_ERROR")) + len(result.log.events("INVALID_OUTPUT"))
        if failures:
            print(f"warning: {failures} agent call(s) failed or were invalid; see the log above"
                  + (f" (client disabled: {llm.disabled_reason})" if llm.disabled_reason else ""))
    if args.log_out:
        result.log.save(args.log_out)
        print(f"log saved -> {args.log_out}")
        if live:
            calls = Path(args.log_out).with_suffix(".llm_calls.json")
            calls.write_text(json.dumps(llm.calls_as_dicts(), indent=2), encoding="utf-8")
            print(f"llm calls saved -> {calls}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
