"""Loads config.toml (model name, API key, fake-mode switch, limits, margins)."""
from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import Field

from .constraints import Limits, Margins
from .schemas import Strict

ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(Strict):
    model: str = "claude-opus-5-5"
    api_key: str = ""  # empty = let the SDK resolve ANTHROPIC_API_KEY / `ant auth login`
    fake_mode: bool = True
    effort: Literal["low", "medium", "high", "xhigh", "max"] = "low"
    max_tokens: int = Field(default=16000, ge=1024)  # thinking tokens count toward this
    max_repairs: int = Field(default=1, ge=0, le=2)  # extra calls to fix an invalid reply; hard-capped at 2
    timeout_s: float = Field(default=60.0, gt=0)
    refusal_fallback: bool = True  # server-side `fallbacks: "default"` (Claude API only)


class DataConfig(Strict):
    flights_csv: str = "flights.csv"
    scenario_path: str = "data/scenario.json"


class ServerConfig(Strict):
    host: str = "127.0.0.1"  # use 0.0.0.0 only when phones must reach it (Stage 7)
    port: int = Field(default=8000, ge=1, le=65535)
    allow_live: bool = False  # lets the dashboard start real-Claude runs (spends API credit)


class AppConfig(Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    server: ServerConfig = Field(default_factory=ServerConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    limits: Limits = Field(default_factory=Limits)
    margins: Margins = Field(default_factory=Margins)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load `path`, else $NEGOTIATOR_CONFIG, else config.toml, else config.example.toml, else defaults."""
    path = path or os.environ.get("NEGOTIATOR_CONFIG")
    candidates = [Path(path)] if path else [ROOT / "config.toml", ROOT / "config.example.toml"]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("rb") as fh:
                return AppConfig.model_validate(tomllib.load(fh))
    if path:
        raise FileNotFoundError(path)
    return AppConfig()
