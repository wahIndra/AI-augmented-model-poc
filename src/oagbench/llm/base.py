"""Provider-neutral LLM interface. Pipelines only depend on this module."""

import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Protocol

# USD per million tokens: (input, output, cache read). Unknown models are costed at 0 and flagged.
PRICES: dict[str, tuple[float, float, float]] = {
    "claude-fable-5-1": (10.0, 50.0, 0.25),
    "claude-opus-5-5": (4.0, 20.0, 0.20),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


class LLMError(RuntimeError):
    pass


class RefusalError(LLMError):
    pass


@dataclass(frozen=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0


@dataclass(frozen=True)
class LLMResult:
    text: str
    data: dict[str, Any] | None
    usage: Usage
    model: str
    latency_ms: int

    def trace(self, purpose: str) -> dict[str, Any]:
        return {
            "purpose": purpose,
            "model": self.model,
            **asdict(self.usage),
            "latency_ms": self.latency_ms,
            "cost_usd": cost_usd(self.model, self.usage),
            "cost_known": self.model in PRICES,
        }


class LLMProvider(Protocol):
    name: str
    model: str

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8000,
        model: str | None = None,
    ) -> LLMResult:
        """One request/response. With `schema`, `data` holds the parsed JSON object."""
        ...


def cost_usd(model: str, usage: Usage) -> float:
    price = PRICES.get(model)
    if price is None:
        return 0.0
    p_in, p_out, p_cache = price
    return (usage.input_tokens * p_in + usage.output_tokens * p_out + usage.cache_read_tokens * p_cache) / 1e6


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from model text, tolerating code fences or surrounding prose."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise LLMError(f"model did not return JSON: {text[:200]!r}") from None
        try:
            value = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError(f"model returned invalid JSON: {exc}") from None
    if not isinstance(value, dict):
        raise LLMError("model returned JSON that is not an object")
    return value
