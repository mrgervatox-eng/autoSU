"""Loads config.toml (model name, API key, fake-mode switch, limits, margins)."""
from __future__ import annotations

import tomllib
from pathlib import Path

from pydantic import Field

from .constraints import Limits, Margins
from .schemas import Strict

ROOT = Path(__file__).resolve().parent.parent


class LLMConfig(Strict):
    model: str = "claude-sonnet-5-5"
    api_key: str = ""
    fake_mode: bool = True


class DataConfig(Strict):
    flights_csv: str = "flights.csv"
    scenario_path: str = "data/scenario.json"


class AppConfig(Strict):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    limits: Limits = Field(default_factory=Limits)
    margins: Margins = Field(default_factory=Margins)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load `path`, else config.toml, else config.example.toml, else defaults."""
    candidates = [Path(path)] if path else [ROOT / "config.toml", ROOT / "config.example.toml"]
    for candidate in candidates:
        if candidate.is_file():
            with candidate.open("rb") as fh:
                return AppConfig.model_validate(tomllib.load(fh))
    if path:
        raise FileNotFoundError(path)
    return AppConfig()
