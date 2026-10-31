"""Stage 5: FastAPI + WebSocket dashboard for the negotiation.

  python -m negotiator.server            # then open http://127.0.0.1:8000

The state machine still owns the loop. This server only starts a run, forwards each consensus-log
entry to every connected browser as it happens, and replays the run to late joiners.

Events sent on /ws (JSON):
  idle          {scenario}                   nothing has run yet; scenario is the preview
  run_started   {run_id, mode, model, scenario}
  entry         {run_id, entry}              one consensus-log entry
  run_finished  {run_id, outcome, reason, rounds_used, proposal, log_valid, head_hash, entries, llm_usage}
  run_error     {run_id, message}
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Callable, Literal

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from .config import ROOT, AppConfig, load_config
from .engine import LLM, Negotiation
from .fake_llm import PERSONAS, FakeLLM
from .log import ConsensusLog, LogEntry
from .scenario import FIXTURE_CSV, Scenario, build_scenario, load_scenario
from .schemas import Role

logger = logging.getLogger(__name__)
STATIC = Path(__file__).parent / "static"

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
        self.publish({"type": "run_started", "run_id": run_id, "mode": mode, "model": model if mode == "live" else "fake",
                      "scenario": scenario.model_dump(mode="json")})
        self._task = asyncio.create_task(self._run(asyncio.get_running_loop(), run_id, scenario, llm, delay))
        return run_id

    async def _run(self, loop: asyncio.AbstractEventLoop, run_id: str, scenario: Scenario, llm: LLM, delay: float) -> None:
        log = ConsensusLog()

        def on_entry(entry: LogEntry) -> None:  # runs on the worker thread
            loop.call_soon_threadsafe(self.publish, {"type": "entry", "run_id": run_id, "entry": entry.model_dump(mode="json")})
            if delay:
                time.sleep(delay)

        log.subscribe(on_entry)
        try:
            result = await asyncio.to_thread(Negotiation(scenario, llm, log).run)
            usage = llm.usage_summary() if hasattr(llm, "usage_summary") else None
            final: dict[str, Any] = {
                "type": "run_finished", "run_id": run_id, "outcome": result.outcome.value, "reason": result.reason,
                "rounds_used": result.rounds_used, "log_valid": log.verify(), "entries": len(log.entries),
                "head_hash": log.entries[-1].hash if log.entries else None, "llm_usage": usage,
                "proposal": None if result.proposal is None else result.proposal.model_dump() | {"proposal_id": result.proposal.proposal_id},
            }
        except Exception as exc:  # noqa: BLE001 - report to the browser instead of dying silently
            logger.exception("run failed")
            final = {"type": "run_error", "run_id": run_id, "message": f"{type(exc).__name__}: {exc}"}
        self.status = "done"
        self.publish(final)


def _build_scenario(req: RunRequest, cfg: AppConfig, preview: Scenario) -> Scenario:
    if req.source == "file":
        return preview.model_copy(deep=True)  # the saved scenario (or the bundled fixture one)
    csv_path = Path(cfg.data.flights_csv)
    csv_path = csv_path if csv_path.is_absolute() else ROOT / csv_path
    if not csv_path.is_file():
        csv_path = FIXTURE_CSV
    return build_scenario(csv_path, req.month, req.day, cfg.margins, cfg.limits, req.tail, req.kind)


def create_app(cfg: AppConfig | None = None, llm_factory: LLMFactory = default_llm_factory,
               scenario_path: str | Path | None = None) -> FastAPI:
    cfg = cfg or load_config()
    path = Path(scenario_path) if scenario_path else ROOT / cfg.data.scenario_path
    if path.is_file():
        preview = load_scenario(path)
    else:  # no saved scenario yet: preview from the bundled fixture so the page is never empty
        preview = build_scenario(FIXTURE_CSV, 1, 1, cfg.margins, cfg.limits, kind="weather")
    manager = RunManager(preview)
    app = FastAPI(title="ATC disruption negotiator")
    app.state.manager = manager

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
            scenario = await asyncio.to_thread(_build_scenario, req, cfg, preview)
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
