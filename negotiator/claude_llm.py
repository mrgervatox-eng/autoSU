"""Stage 4: real Claude agents behind the same `complete(role, view) -> JSON string` interface.

One call per role per turn, each with its own system prompt. The model only fills in
type / proposal / proposal_id / rationale / confidence; this module adds sender, round and
provenance, so an agent cannot impersonate another role or claim a different round.

Safety rails, all bounded:
  - structured output (`output_config.format`) so the reply is schema-valid JSON,
  - at most `max_repairs` (<= 2) extra calls to fix a reply that still fails validation,
  - auth / permission / model / bad-request errors disable the client for the rest of the run,
  - anything else (rate limits, outages) surfaces as an exception the engine logs as AGENT_ERROR.
No sampling parameters, no forced tool choice and no assistant prefill: current models reject them.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from typing import Any

import anthropic

from .config import LLMConfig
from .engine import ALLOWED
from .prompts import REPLY_FIELDS, reply_schema, repair_message, system_prompt, user_message
from .schemas import Role, parse_message

FALLBACK_BETA = "server-side-fallback-2026-07-01"
RAW_KEEP = 2000  # chars of raw model text kept per call, for traceability


class LLMError(Exception):
    """A call failed in a way the engine should log as AGENT_ERROR."""


@dataclass
class CallRecord:
    role: str
    attempt: int  # 0 = first try, 1+ = repair
    model: str
    input_tokens: int
    output_tokens: int
    stop_reason: str | None
    request_id: str | None
    raw: str


def _clamp01(value: Any, default: float = 0.5) -> float:
    try:
        return min(1.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return default


def _strip_fences(text: str) -> str:
    m = re.fullmatch(r"\s*```(?:json)?\s*(.*?)\s*```\s*", text, flags=re.DOTALL)
    return m.group(1) if m else text.strip()


class ClaudeLLM:
    def __init__(self, cfg: LLMConfig, client: Any | None = None) -> None:
        self.cfg = cfg
        if client is None:
            kwargs: dict[str, Any] = {"timeout": cfg.timeout_s}
            if cfg.api_key:
                kwargs["api_key"] = cfg.api_key
            client = anthropic.Anthropic(**kwargs)  # else resolves ANTHROPIC_API_KEY / `ant auth login`
        self.client = client
        self.calls: list[CallRecord] = []
        self.disabled_reason: str | None = None

    # --- LLM protocol -------------------------------------------------------

    def complete(self, role: Role, view: dict[str, Any]) -> str:
        if self.disabled_reason:
            raise LLMError(f"client disabled after earlier failure: {self.disabled_reason}")
        system = system_prompt(role, view["max_rounds"])
        messages: list[dict[str, Any]] = [{"role": "user", "content": user_message(view)}]
        text = ""
        for attempt in range(1 + min(self.cfg.max_repairs, 2)):
            text = self._call(role.value, reply_schema(role), system, messages, attempt)
            assembled, problem = self._assemble(role, view, text)
            if assembled is not None:
                return json.dumps(assembled)
            messages += [
                {"role": "assistant", "content": text or "(empty reply)"},
                {"role": "user", "content": repair_message(problem or "invalid reply")},
            ]
        return text  # still invalid: the engine logs it as INVALID_OUTPUT

    # --- one API call ---------------------------------------------------------

    def ask_json(self, label: str, system: str, user: str, schema: dict[str, Any]) -> str:
        """One structured-output call for something that is not a negotiating role (e.g. the controller's rule parser).

        Same request shape, error handling, usage tracking and 'disable after a bad key' behaviour as the agents; no repair loop.
        """
        if self.disabled_reason:
            raise LLMError(f"client disabled after earlier failure: {self.disabled_reason}")
        return self._call(label, schema, system, [{"role": "user", "content": user}], 0)

    def _call(self, label: str, schema: dict[str, Any], system: str, messages: list[dict[str, Any]], attempt: int) -> str:
        kwargs: dict[str, Any] = dict(
            model=self.cfg.model,
            max_tokens=self.cfg.max_tokens,
            system=system,
            messages=messages,
            output_config={
                "effort": self.cfg.effort,
                "format": {"type": "json_schema", "schema": schema},
            },
        )
        try:
            if self.cfg.refusal_fallback:
                resp = self.client.beta.messages.create(betas=[FALLBACK_BETA], fallbacks="default", **kwargs)
            else:
                resp = self.client.messages.create(**kwargs)
        except (anthropic.AuthenticationError, anthropic.PermissionDeniedError,
                anthropic.NotFoundError, anthropic.BadRequestError) as exc:
            # These repeat on every call (bad key, wrong model name, bad request shape): stop spending calls.
            self.disabled_reason = f"{type(exc).__name__}: {getattr(exc, 'message', exc)}"
            raise LLMError(self.disabled_reason) from exc

        text = "".join(b.text for b in resp.content if getattr(b, "type", None) == "text")
        usage = getattr(resp, "usage", None)
        self.calls.append(CallRecord(
            role=label, attempt=attempt, model=getattr(resp, "model", self.cfg.model),
            input_tokens=getattr(usage, "input_tokens", 0) or 0, output_tokens=getattr(usage, "output_tokens", 0) or 0,
            stop_reason=resp.stop_reason, request_id=getattr(resp, "_request_id", None), raw=text[:RAW_KEEP],
        ))
        if resp.stop_reason == "refusal":
            category = getattr(getattr(resp, "stop_details", None), "category", None)
            raise LLMError(f"model refused (category: {category})")
        return text

    # --- reply -> full Message dict ---------------------------------------------

    def _assemble(self, role: Role, view: dict[str, Any], text: str) -> tuple[dict[str, Any] | None, str | None]:
        try:
            reply = json.loads(_strip_fences(text))
        except json.JSONDecodeError as exc:
            return None, f"not valid JSON ({exc})"
        if not isinstance(reply, dict):
            return None, "reply must be a JSON object"
        extra = set(reply) - set(REPLY_FIELDS)
        if extra:
            return None, f"unknown fields {sorted(extra)}"

        msg: dict[str, Any] = {
            "sender": role.value,  # set by us, never by the model
            "round": view["round"],
            "type": reply.get("type"),
            "rationale": str(reply.get("rationale") or "")[:500],
            "provenance": {"source": "llm", "confidence": _clamp01(reply.get("confidence")), "note": self.cfg.model},
        }
        if reply.get("proposal") is not None:
            msg["proposal"] = reply["proposal"]
        if reply.get("proposal_id"):
            msg["proposal_id"] = reply["proposal_id"]

        parsed, err = parse_message(msg)
        if parsed is None:
            return None, err
        if parsed.type not in ALLOWED[role]:
            allowed = ", ".join(sorted(t.value for t in ALLOWED[role]))
            return None, f"{role.value} may only send: {allowed}"
        return msg, None

    # --- traceability --------------------------------------------------------------

    def usage_summary(self) -> dict[str, Any]:
        return {
            "calls": len(self.calls),
            "repairs": sum(1 for c in self.calls if c.attempt > 0),
            "input_tokens": sum(c.input_tokens for c in self.calls),
            "output_tokens": sum(c.output_tokens for c in self.calls),
            "models": sorted({c.model for c in self.calls}),
        }

    def calls_as_dicts(self) -> list[dict[str, Any]]:
        return [asdict(c) for c in self.calls]
