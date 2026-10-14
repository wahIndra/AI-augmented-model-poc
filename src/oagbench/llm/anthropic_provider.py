"""Claude via the official Anthropic SDK."""

import time
from typing import Any

from oagbench.llm.base import LLMError, LLMResult, RefusalError, Usage, parse_json_object


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, *, api_key: str, model: str, effort: str, fallbacks: bool, client: Any = None):
        if client is None:
            import anthropic

            client = anthropic.Anthropic(api_key=api_key, max_retries=4)
        self._client = client
        self.model = model
        self.effort = effort
        self.fallbacks = fallbacks

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8000,
        model: str | None = None,
    ) -> LLMResult:
        output_config: dict[str, Any] = {"effort": self.effort}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        request: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            # The system prompt is the stable prefix (instructions, ontology schema, few-shot), so cache it.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }

        start = time.perf_counter()
        if self.fallbacks:
            response = self._client.beta.messages.create(
                betas=["server-side-fallback-2026-07-01"], fallbacks="default", **request
            )
        else:
            response = self._client.messages.create(**request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        if response.stop_reason == "refusal":
            raise RefusalError(f"request declined: {getattr(response, 'stop_details', None)}")
        if response.stop_reason == "max_tokens":
            raise LLMError(f"response truncated at max_tokens={max_tokens}")

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return LLMResult(
            text=text,
            data=parse_json_object(text) if schema is not None else None,
            usage=Usage(
                input_tokens=usage.input_tokens + (getattr(usage, "cache_creation_input_tokens", 0) or 0),
                output_tokens=usage.output_tokens,
                cache_read_tokens=getattr(usage, "cache_read_input_tokens", 0) or 0,
            ),
            model=response.model,
            latency_ms=latency_ms,
        )
