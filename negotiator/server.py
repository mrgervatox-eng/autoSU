"""FastAPI + WebSocket dashboard for the negotiation (Stage 5), with the controller's decision step (Stage 6).

  python -m negotiator.server            # then open http://127.0.0.1:8000

The state machine still owns the loop. This server starts a run, forwards each consensus-log entry to
every connected browser as it happens, replays the run to late joiners, and (with controller "manual")
waits for the human's decision between negotiations.

Events sent on /ws (JSON):
  idle           {scenario}                   nothing has run yet; scenario is the preview
  run_started    {run_id, mode, controller, model, scenario}
  entry          {run_id, entry, view}        one consensus-log entry; `view` is its display text
  round_summary  {run_id, attempt, round, kind, short, text}   why a round ended, in plain words
  options        {run_id, prompt_id, attempt, rejections_left, timeout_s, remaining_s, problem, rules, options, rank_rule}
                                              the controller is being asked to decide
  prompt_closed  {run_id, prompt_id}          that question is finished (answered or timed out)
  run_finished   {run_id, outcome, reason, rounds_used, proposal, proposal_chips, round_summaries, log_valid,
                  head_hash, entries, llm_usage, controller, option, negotiations, rejections, rules}
  run_error      {run_id, message}

Other endpoints: GET /api/state, GET /api/preview (a scenario without running it), POST /api/run,
POST /api/decision (the controller's answer), GET /api/log (the current run's consensus log as JSON).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import queue
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT, AppConfig, load_config
from .constraints import TurbulenceObservation
from .engine import LLM, Negotiation
from .fake_llm import PERSONAS, FakeLLM
from .log import ConsensusLog, LogEntry
from .options import Option
from .parsing import ConstraintParser, RuleParser
from .scenario import FIXTURE_CSV, Scenario, build_scenario, load_scenario
from .schemas import Provenance, Role, Source
from .session import RANK_RULE, Decision, Session, SessionResult
from .summary import RoundTracker, all_round_summaries, entry_view, proposal_chips

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
PREVIEW_CACHE_SIZE = 64
LIVE_TTL_S = 120.0  # a phone reading older than this is ignored (a phone left in a pocket must not decide a run)

LLMFactory = Callable[[str, dict[str, str], AppConfig], LLM]
ParserFactory = Callable[[str, LLM, AppConfig], ConstraintParser]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["file", "csv"] = "file"  # saved scenario JSON, or build from the flights CSV
    month: int = Field(default=1, ge=1, le=12)
    day: int = Field(default=1, ge=1, le=31)
    kind: Literal["weather", "late_aircraft"] | None = None
    tail: str | None = None
    mode: Literal["fake", "live"] = "fake"
    controller: Literal["off", "manual"] = "off"  # "manual" waits for the human controller after the agents finish
    personas: dict[str, str] = Field(default_factory=dict)  # fake mode only, e.g. {"PILOT": "stubborn"}
    step_delay_s: float = Field(default=0.7, ge=0, le=5)  # pacing between log entries in fake mode


class TurbulenceReading(BaseModel):
    """What the pilot phone sends: which holding area, the 0-1 turbulence proxy, and how much to trust it."""

    model_config = ConfigDict(extra="forbid")

    area: str = Field(min_length=1, max_length=20)
    index: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompt_id: str
    action: Literal["approve", "reject"]
    option: int | None = None
    text: str | None = Field(default=None, max_length=500)


def default_llm_factory(mode: str, personas: dict[str, str], cfg: AppConfig) -> LLM:
    if mode == "live":
        from .claude_llm import ClaudeLLM  # lazy: fake mode never needs the anthropic package

        return ClaudeLLM(cfg.llm)
    return FakeLLM(personas)


def default_parser_factory(mode: str, llm: LLM, cfg: AppConfig) -> ConstraintParser:
    if mode == "live":
        from .parsing import ClaudeConstraintParser

        return ClaudeConstraintParser(llm)
    return RuleParser()


@dataclass
class Pending:
    """One open question to the controller."""

    prompt_id: str
    options: list[Option]
    attempt: int
    rejections_left: int
    timeout_s: float
    rules: list[str]
    problem: str | None
    opened: float = field(default_factory=time.monotonic)
    box: "queue.Queue[Decision]" = field(default_factory=lambda: queue.Queue(maxsize=1))
    answered: bool = False

    def event(self, run_id: str | None) -> dict[str, Any]:
        remaining = max(0.0, self.timeout_s - (time.monotonic() - self.opened))
        return {"type": "options", "run_id": run_id, "prompt_id": self.prompt_id, "attempt": self.attempt,
                "rejections_left": self.rejections_left, "timeout_s": self.timeout_s, "remaining_s": round(remaining, 1),
                "problem": self.problem, "rules": self.rules, "options": [o.as_dict() for o in self.options], "rank_rule": RANK_RULE}


class WebDecider:
    """Lets the session (on its worker thread) ask the browser and wait, with a timeout, for the answer."""

    def __init__(self, manager: "RunManager", loop: asyncio.AbstractEventLoop) -> None:
        self.manager, self.loop = manager, loop

    def decide(self, options, *, attempt, rejections_left, timeout_s, rules, problem) -> Decision:
        pending = Pending(uuid.uuid4().hex[:8], list(options), attempt, rejections_left, timeout_s, list(rules), problem)
        self.loop.call_soon_threadsafe(self.manager.open_prompt, pending)
        try:
            decision = pending.box.get(timeout=timeout_s)
        except queue.Empty:
            decision = Decision("timeout")
        self.loop.call_soon_threadsafe(self.manager.close_prompt, pending.prompt_id)
        return decision


class RunManager:
    """One run at a time. Keeps the current run's events so a browser that connects late sees all of it."""

    def __init__(self, preview: Scenario) -> None:
        self.status: Literal["idle", "running", "done"] = "idle"
        self.preview = preview
        self.scenario = preview
        self.events: list[dict[str, Any]] = []
        self.subscribers: set[asyncio.Queue] = set()
        self.last_log: ConsensusLog | None = None
        self.last_run_id: str | None = None
        self.pending: Pending | None = None
        self.live: dict[str, dict[str, float]] = {}  # area -> latest phone reading
        self._task: asyncio.Task | None = None

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        self.broadcast(event)

    def broadcast(self, event: dict[str, Any]) -> None:
        """Send to every connected browser without adding to the run's replayable history."""
        for q in list(self.subscribers):
            q.put_nowait(event)

    # --- live conditions from phones ------------------------------------------------------------

    def set_live(self, area: str, index: float, confidence: float) -> dict[str, Any]:
        self.live[area] = {"index": index, "confidence": confidence, "at": time.monotonic()}
        return {"type": "observation", "kind": "turbulence", "area": area, "index": index, "confidence": confidence, "source": "sensor"}

    def live_snapshot(self) -> dict[str, dict[str, float]]:
        now = time.monotonic()
        return {a: {"index": r["index"], "confidence": r["confidence"], "age_s": round(now - r["at"], 1)}
                for a, r in self.live.items() if now - r["at"] <= LIVE_TTL_S}

    def apply_live(self, scenario: Scenario) -> Scenario:
        """Replace the simulated turbulence reading of an area with the fresh phone reading, marked source=sensor."""
        fresh = self.live_snapshot()
        scenario.context.turbulence = [
            TurbulenceObservation(area=o.area, index=fresh[o.area]["index"], provenance=Provenance(
                source=Source.SENSOR, confidence=fresh[o.area]["confidence"], note="pilot phone accelerometer"))
            if o.area in fresh else o
            for o in scenario.context.turbulence]
        return scenario

    def snapshot(self) -> list[dict[str, Any]]:
        events = list(self.events) if self.events else [{"type": "idle", "scenario": self.preview.model_dump(mode="json")}]
        if self.pending is not None:  # a browser joining mid-question sees that question once, with the time that is really left
            events = [e for e in events if not (e["type"] == "options" and e.get("prompt_id") == self.pending.prompt_id)]
            events.append(self.pending.event(self.last_run_id))
        return events

    def open_prompt(self, pending: Pending) -> None:
        self.pending = pending
        self.publish(pending.event(self.last_run_id))

    def close_prompt(self, prompt_id: str) -> None:
        if self.pending is not None and self.pending.prompt_id == prompt_id:
            self.pending = None
            self.publish({"type": "prompt_closed", "run_id": self.last_run_id, "prompt_id": prompt_id})

    def start(self, scenario: Scenario, llm: LLM, mode: str, model: str, delay: float, controller: str = "off",
              parser: ConstraintParser | None = None) -> str:
        if self.status == "running":
            raise RuntimeError("a run is already in progress")
        run_id = uuid.uuid4().hex[:8]
        self.status, self.scenario, self.events, self.pending = "running", scenario, [], None
        self.last_log, self.last_run_id = ConsensusLog(), run_id
        self.publish({"type": "run_started", "run_id": run_id, "mode": mode, "controller": controller,
                      "model": model if mode == "live" else "fake", "scenario": scenario.model_dump(mode="json")})
        self._task = asyncio.create_task(self._run(asyncio.get_running_loop(), run_id, scenario, llm, delay, self.last_log, controller, parser))
        return run_id

    async def _run(self, loop: asyncio.AbstractEventLoop, run_id: str, scenario: Scenario, llm: LLM, delay: float,
                   log: ConsensusLog, controller: str, parser: ConstraintParser | None) -> None:
        tracker = RoundTracker()

        def summary_event(s) -> dict[str, Any]:
            return {"type": "round_summary", "run_id": run_id, **s.as_dict()}

        def on_entry(entry: LogEntry) -> None:  # runs on the worker thread
            # a round's summary goes out just before the first entry that follows it, so the page can draw the divider in place
            for s in tracker.before(log.entries, entry):
                loop.call_soon_threadsafe(self.publish, summary_event(s))
            loop.call_soon_threadsafe(self.publish, {"type": "entry", "run_id": run_id, "entry": entry.model_dump(mode="json"),
                                                     "view": entry_view(entry)})
            if delay:
                time.sleep(delay)

        log.subscribe(on_entry)
        try:
            if controller == "manual":
                work = Session(scenario, llm, parser or RuleParser(), WebDecider(self, loop), log).run
            else:
                work = Negotiation(scenario, llm, log).run
            result = await asyncio.to_thread(work)
            for s in tracker.finish(log.entries):
                self.publish(summary_event(s))
            final = self._finished(run_id, result, log, llm, controller)
        except Exception as exc:  # noqa: BLE001 - report to the browser instead of dying silently
            logger.exception("run failed")
            final = {"type": "run_error", "run_id": run_id, "message": f"{type(exc).__name__}: {exc}"}
        self.pending = None
        self.status = "done"
        self.publish(final)

    @staticmethod
    def _finished(run_id: str, result: Any, log: ConsensusLog, llm: LLM, controller: str) -> dict[str, Any]:
        is_session = isinstance(result, SessionResult)
        outcome = result.outcome if is_session else result.outcome.value
        last_attempt = log.entries[-1].attempt if log.entries else 1
        final: dict[str, Any] = {
            "type": "run_finished", "run_id": run_id, "controller": controller, "outcome": outcome, "reason": result.reason,
            "rounds_used": max((e.round for e in log.entries if e.attempt == last_attempt), default=0) if is_session else result.rounds_used,
            "log_valid": log.verify(), "entries": len(log.entries), "head_hash": log.entries[-1].hash if log.entries else None,
            "llm_usage": llm.usage_summary() if hasattr(llm, "usage_summary") else None,
            "round_summaries": [s.as_dict() for s in all_round_summaries(log.entries)],
            "proposal": None if result.proposal is None else result.proposal.model_dump() | {"proposal_id": result.proposal.proposal_id},
            "proposal_chips": None if result.proposal is None else proposal_chips(result.proposal),
        }
        if is_session:
            final |= {"option": result.option, "negotiations": result.negotiations, "rejections": result.rejections,
                      "rules": [c.describe() for c in result.constraints]}
        return final


def _csv_path(cfg: AppConfig) -> Path:
    path = Path(cfg.data.flights_csv)
    path = path if path.is_absolute() else ROOT / path
    return path if path.is_file() else FIXTURE_CSV  # no big file on this machine: use the bundled sample


def create_app(cfg: AppConfig | None = None, llm_factory: LLMFactory = default_llm_factory,
               scenario_path: str | Path | None = None, parser_factory: ParserFactory = default_parser_factory) -> FastAPI:
    cfg = cfg or load_config()
    path = Path(scenario_path) if scenario_path else ROOT / cfg.data.scenario_path
    if path.is_file():
        saved = load_scenario(path)
    else:  # no saved scenario yet: use the bundled fixture so the page is never empty
        saved = build_scenario(FIXTURE_CSV, 1, 1, cfg.margins, cfg.limits, kind="weather")
    manager = RunManager(saved)
    app = FastAPI(title="ATC disruption negotiator")
    app.state.manager = manager

    cache: OrderedDict[tuple, Scenario] = OrderedDict()
    cache_lock = threading.Lock()

    def scenario_for(source: str, month: int, day: int, kind: str | None, tail: str | None) -> Scenario:
        """The scenario a preview shows and a run negotiates (the same one: building is cached and deterministic)."""
        if source == "file":
            return manager.apply_live(saved.model_copy(deep=True))
        key = (month, day, kind, tail)
        with cache_lock:
            hit = cache.get(key)
            if hit is not None:
                cache.move_to_end(key)
        if hit is None:
            built = build_scenario(_csv_path(cfg), month, day, cfg.margins, cfg.limits, tail, kind)  # slow for late-year dates: not under the lock
            with cache_lock:
                cache[key] = hit = built
                while len(cache) > PREVIEW_CACHE_SIZE:
                    cache.popitem(last=False)
        return manager.apply_live(hit.model_copy(deep=True))

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/pilot")
    def pilot_page() -> FileResponse:
        return FileResponse(STATIC / "pilot.html")

    @app.post("/api/observation/turbulence", status_code=202)
    def turbulence(reading: TurbulenceReading) -> dict[str, bool]:
        """A pilot phone reports turbulence for a holding area; the next run (and preview) uses it."""
        area = reading.area.strip().upper()
        if area not in {o.area for o in saved.context.turbulence}:
            raise HTTPException(400, f"unknown holding area {area}; known: {', '.join(sorted({o.area for o in saved.context.turbulence}))}")
        manager.broadcast(manager.set_live(area, round(reading.index, 3), round(reading.confidence, 2)))
        return {"ok": True}

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {
            "status": manager.status,
            "allow_live": cfg.server.allow_live,
            "model": cfg.llm.model,
            "personas": list(PERSONAS),
            "roles": [r.value for r in Role],
            "max_rounds": cfg.limits.max_rounds,
            "max_human_rejections": cfg.limits.max_human_rejections,
            "decision_timeout_s": cfg.limits.decision_timeout_s,
            "live": manager.live_snapshot(),
            "areas": sorted({o.area for o in saved.context.turbulence}),
            "scenario": manager.scenario.model_dump(mode="json"),
        }

    @app.get("/api/preview")
    async def preview(source: Literal["file", "csv"] = "file", month: int = Query(1, ge=1, le=12), day: int = Query(1, ge=1, le=31),
                      kind: Literal["weather", "late_aircraft"] | None = None, tail: str | None = None) -> dict[str, Any]:
        """The scenario these settings would negotiate, without running anything."""
        try:
            scenario = await asyncio.to_thread(scenario_for, source, month, day, kind, tail)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        return scenario.model_dump(mode="json")

    @app.get("/api/log")
    def download_log() -> Response:
        if manager.last_log is None:
            raise HTTPException(404, "no run yet")
        return Response(manager.last_log.to_json(), media_type="application/json",
                        headers={"Content-Disposition": f'attachment; filename="consensus_log_{manager.last_run_id}.json"'})

    @app.post("/api/run", status_code=202)
    async def run(req: RunRequest) -> dict[str, str]:
        if manager.status == "running":
            raise HTTPException(409, "a run is already in progress")
        if req.mode == "live" and not cfg.server.allow_live:
            raise HTTPException(403, "live runs are disabled; set server.allow_live = true in config.toml")
        if req.mode == "live" and req.personas:
            raise HTTPException(400, "personas only work in fake mode")
        try:
            personas = {k.upper(): v for k, v in req.personas.items()}
            scenario = await asyncio.to_thread(scenario_for, req.source, req.month, req.day, req.kind, req.tail)
            llm = llm_factory(req.mode, personas, cfg)
            parser = parser_factory(req.mode, llm, cfg) if req.controller == "manual" else None
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - e.g. no API credentials
            raise HTTPException(500, f"could not start the agents: {exc}") from exc
        if manager.status == "running":  # another request won the race while we were building
            raise HTTPException(409, "a run is already in progress")
        run_id = manager.start(scenario, llm, req.mode, cfg.llm.model, req.step_delay_s if req.mode == "fake" else 0.0,
                               req.controller, parser)
        return {"run_id": run_id}

    @app.post("/api/decision", status_code=202)
    async def decide(req: DecisionRequest) -> dict[str, bool]:
        """The controller's answer to the question currently open."""
        pending = manager.pending
        if manager.status != "running" or pending is None:
            raise HTTPException(409, "no decision is being asked for right now")
        if req.prompt_id != pending.prompt_id:
            raise HTTPException(409, "that question is no longer open")
        if pending.answered:
            raise HTTPException(409, "that question has already been answered")
        if req.action == "approve":
            if req.option not in {o.rank for o in pending.options}:
                raise HTTPException(400, "choose one of the options shown")
            decision = Decision("approve", option=req.option)
        else:
            text = (req.text or "").strip()
            if not text:
                raise HTTPException(400, "write what you want changed")
            decision = Decision("reject", text=text)
        pending.answered = True
        pending.box.put_nowait(decision)
        return {"ok": True}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        queue_: asyncio.Queue = asyncio.Queue()
        for event in manager.snapshot():  # snapshot + subscribe happen with no await between: nothing is missed or doubled
            queue_.put_nowait(event)
        manager.subscribers.add(queue_)
        try:
            while True:
                await socket.send_json(await queue_.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            manager.subscribers.discard(queue_)

    return app


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    cfg = load_config()
    ap = argparse.ArgumentParser(description="Serve the negotiation dashboard.")
    ap.add_argument("--host", default=cfg.server.host)
    ap.add_argument("--port", type=int, default=cfg.server.port)
    args = ap.parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(f"warning: listening on {args.host} exposes this unauthenticated dashboard to your network"
              + ("; live runs are ENABLED" if cfg.server.allow_live else ""))
    uvicorn.run(create_app(cfg), host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
