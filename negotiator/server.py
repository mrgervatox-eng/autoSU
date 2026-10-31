"""Stage 5: FastAPI + WebSocket dashboard for the negotiation.

  python -m negotiator.server            # then open http://127.0.0.1:8000

The state machine still owns the loop. This server only starts a run, forwards each consensus-log
entry to every connected browser as it happens, and replays the run to late joiners.

Events sent on /ws (JSON):
  idle           {scenario}                   nothing has run yet; scenario is the preview
  run_started    {run_id, mode, model, scenario}
  entry          {run_id, entry, view}        one consensus-log entry; `view` is its display text
  round_summary  {run_id, round, kind, short, text}   why a round ended, in plain words
  run_finished   {run_id, outcome, reason, rounds_used, proposal, proposal_chips, round_summaries,
                  log_valid, head_hash, entries, llm_usage}
  run_error      {run_id, message}

Other endpoints: GET /api/state, GET /api/preview (a scenario without running it),
POST /api/run, GET /api/log (the current run's consensus log as JSON).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, Query, Response, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT, AppConfig, load_config
from .engine import LLM, Negotiation
from .fake_llm import PERSONAS, FakeLLM
from .log import ConsensusLog, LogEntry
from .scenario import FIXTURE_CSV, Scenario, build_scenario, load_scenario
from .schemas import Role
from .summary import RoundTracker, all_round_summaries, entry_view, proposal_chips

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"
PREVIEW_CACHE_SIZE = 64

LLMFactory = Callable[[str, dict[str, str], AppConfig], LLM]


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["file", "csv"] = "file"  # saved scenario JSON, or build from the flights CSV
    month: int = Field(default=1, ge=1, le=12)
    day: int = Field(default=1, ge=1, le=31)
    kind: Literal["weather", "late_aircraft"] | None = None
    tail: str | None = None
    mode: Literal["fake", "live"] = "fake"
    personas: dict[str, str] = Field(default_factory=dict)  # fake mode only, e.g. {"PILOT": "stubborn"}
    step_delay_s: float = Field(default=0.7, ge=0, le=5)  # pacing between log entries in fake mode


def default_llm_factory(mode: str, personas: dict[str, str], cfg: AppConfig) -> LLM:
    if mode == "live":
        from .claude_llm import ClaudeLLM  # lazy: fake mode never needs the anthropic package

        return ClaudeLLM(cfg.llm)
    return FakeLLM(personas)


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
        self._task: asyncio.Task | None = None

    def publish(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for q in list(self.subscribers):
            q.put_nowait(event)

    def snapshot(self) -> list[dict[str, Any]]:
        return list(self.events) if self.events else [{"type": "idle", "scenario": self.preview.model_dump(mode="json")}]

    def start(self, scenario: Scenario, llm: LLM, mode: str, model: str, delay: float) -> str:
        if self.status == "running":
            raise RuntimeError("a run is already in progress")
        run_id = uuid.uuid4().hex[:8]
        self.status, self.scenario, self.events = "running", scenario, []
        self.last_log, self.last_run_id = ConsensusLog(), run_id
        self.publish({"type": "run_started", "run_id": run_id, "mode": mode, "model": model if mode == "live" else "fake",
                      "scenario": scenario.model_dump(mode="json")})
        self._task = asyncio.create_task(self._run(asyncio.get_running_loop(), run_id, scenario, llm, delay, self.last_log))
        return run_id

    async def _run(self, loop: asyncio.AbstractEventLoop, run_id: str, scenario: Scenario, llm: LLM, delay: float,
                   log: ConsensusLog) -> None:
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
            result = await asyncio.to_thread(Negotiation(scenario, llm, log).run)
            for s in tracker.finish(log.entries):
                self.publish(summary_event(s))
            usage = llm.usage_summary() if hasattr(llm, "usage_summary") else None
            final: dict[str, Any] = {
                "type": "run_finished", "run_id": run_id, "outcome": result.outcome.value, "reason": result.reason,
                "rounds_used": result.rounds_used, "log_valid": log.verify(), "entries": len(log.entries),
                "head_hash": log.entries[-1].hash if log.entries else None, "llm_usage": usage,
                "round_summaries": [s.as_dict() for s in all_round_summaries(log.entries)],
                "proposal": None if result.proposal is None else result.proposal.model_dump() | {"proposal_id": result.proposal.proposal_id},
                "proposal_chips": None if result.proposal is None else proposal_chips(result.proposal),
            }
        except Exception as exc:  # noqa: BLE001 - report to the browser instead of dying silently
            logger.exception("run failed")
            final = {"type": "run_error", "run_id": run_id, "message": f"{type(exc).__name__}: {exc}"}
        self.status = "done"
        self.publish(final)


def _csv_path(cfg: AppConfig) -> Path:
    path = Path(cfg.data.flights_csv)
    path = path if path.is_absolute() else ROOT / path
    return path if path.is_file() else FIXTURE_CSV  # no big file on this machine: use the bundled sample


def create_app(cfg: AppConfig | None = None, llm_factory: LLMFactory = default_llm_factory,
               scenario_path: str | Path | None = None) -> FastAPI:
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
            return saved.model_copy(deep=True)
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
        return hit.model_copy(deep=True)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "index.html")

    @app.get("/api/state")
    def state() -> dict[str, Any]:
        return {
            "status": manager.status,
            "allow_live": cfg.server.allow_live,
            "model": cfg.llm.model,
            "personas": list(PERSONAS),
            "roles": [r.value for r in Role],
            "max_rounds": cfg.limits.max_rounds,
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
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 - e.g. no API credentials
            raise HTTPException(500, f"could not start the agents: {exc}") from exc
        if manager.status == "running":  # another request won the race while we were building
            raise HTTPException(409, "a run is already in progress")
        run_id = manager.start(scenario, llm, req.mode, cfg.llm.model, req.step_delay_s if req.mode == "fake" else 0.0)
        return {"run_id": run_id}

    @app.websocket("/ws")
    async def ws(socket: WebSocket) -> None:
        await socket.accept()
        queue: asyncio.Queue = asyncio.Queue()
        for event in manager.snapshot():  # snapshot + subscribe happen with no await between: nothing is missed or doubled
            queue.put_nowait(event)
        manager.subscribers.add(queue)
        try:
            while True:
                await socket.send_json(await queue.get())
        except (WebSocketDisconnect, RuntimeError):
            pass
        finally:
            manager.subscribers.discard(queue)

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
