"""Any OpenAI-compatible chat endpoint (OpenAI, Ollama, vLLM, LM Studio)."""

import json
import time
from typing import Any

from oagbench.llm.base import LLMError, LLMResult, Usage, parse_json_object


class OpenAICompatibleProvider:
    name = "openai_compatible"

    def __init__(self, *, base_url: str, api_key: str, model: str, client: Any = None):
        if client is None:
            from openai import OpenAI

            # Local servers ignore the key, but the client requires a non-empty value.
            client = OpenAI(base_url=base_url, api_key=api_key or "not-needed", max_retries=4)
        self._client = client
        self.model = model

    def complete(
        self,
        *,
        system: str,
        prompt: str,
        schema: dict[str, Any] | None = None,
        max_tokens: int = 8000,
        model: str | None = None,
    ) -> LLMResult:
        # Strict json_schema support varies across compatible servers. json_object plus the schema in the
        # prompt works on all of them; callers validate the parsed result.
        if schema is not None:
            system = f"{system}\n\nRespond with one JSON object matching this JSON Schema:\n{json.dumps(schema)}"
        request: dict[str, Any] = {
            "model": model or self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
        }
        if schema is not None:
            request["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        response = self._client.chat.completions.create(**request)
        latency_ms = int((time.perf_counter() - start) * 1000)

        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise LLMError(f"response truncated at max_tokens={max_tokens}")
        text = choice.message.content or ""
        usage = response.usage
        return LLMResult(
            text=text,
            data=parse_json_object(text) if schema is not None else None,
            usage=Usage(
                input_tokens=getattr(usage, "prompt_tokens", 0) or 0,
                output_tokens=getattr(usage, "completion_tokens", 0) or 0,
            ),
            model=response.model or request["model"],
            latency_ms=latency_ms,
        )
