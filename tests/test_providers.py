from types import SimpleNamespace

import pytest

from oagbench.config import ConfigError, Settings
from oagbench.llm import RefusalError, build_provider
from oagbench.llm.anthropic_provider import AnthropicProvider
from oagbench.llm.base import LLMError, parse_json_object
from oagbench.llm.openai_provider import OpenAICompatibleProvider

SCHEMA = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"],
          "additionalProperties": False}


def test_none_provider_means_fake_mode() -> None:
    assert build_provider(Settings(_env_file=None, ai_provider="none")) is None


def test_half_configured_provider_fails_fast() -> None:
    with pytest.raises(ConfigError):
        build_provider(Settings(_env_file=None, ai_provider="anthropic", anthropic_api_key=""))
    with pytest.raises(ConfigError):
        build_provider(Settings(_env_file=None, ai_provider="openai_compatible", openai_model=""))


def test_describe_never_contains_secrets() -> None:
    s = Settings(_env_file=None, ai_provider="anthropic", anthropic_api_key="sk-secret-value")
    assert "sk-secret-value" not in str(s.describe())


class _Recorder:
    def __init__(self, response):
        self.kwargs = None
        self._response = response

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self._response


def _anthropic_response(text='{"x": "1"}', stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason, stop_details=None, model="claude-opus-5-5",
        content=[SimpleNamespace(type="thinking"), SimpleNamespace(type="text", text=text)],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=2,
                              cache_read_input_tokens=100),
    )


def test_anthropic_request_shape_with_fallbacks() -> None:
    beta = _Recorder(_anthropic_response())
    client = SimpleNamespace(beta=SimpleNamespace(messages=beta), messages=_Recorder(None))
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5", effort="medium", fallbacks=True, client=client)
    result = provider.complete(system="sys", prompt="hi", schema=SCHEMA)

    kw = beta.kwargs
    assert kw["betas"] == ["server-side-fallback-2026-07-01"] and kw["fallbacks"] == "default"
    assert kw["output_config"] == {"effort": "medium", "format": {"type": "json_schema", "schema": SCHEMA}}
    assert kw["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "thinking" not in kw and "temperature" not in kw
    assert result.data == {"x": "1"}
    assert (result.usage.input_tokens, result.usage.cache_read_tokens) == (12, 100)
    assert result.trace("t")["cost_usd"] > 0


def test_anthropic_without_fallbacks_uses_ga_endpoint() -> None:
    plain = _Recorder(_anthropic_response("hello"))
    client = SimpleNamespace(beta=SimpleNamespace(messages=_Recorder(None)), messages=plain)
    provider = AnthropicProvider(api_key="k", model="claude-opus-5-5", effort="low", fallbacks=False, client=client)
    assert provider.complete(system="s", prompt="p").text == "hello"
    assert "betas" not in plain.kwargs


def test_anthropic_refusal_and_truncation_raise() -> None:
    for stop, error in (("refusal", RefusalError), ("max_tokens", LLMError)):
        client = SimpleNamespace(beta=SimpleNamespace(messages=_Recorder(_anthropic_response(stop_reason=stop))))
        provider = AnthropicProvider(api_key="k", model="m", effort="low", fallbacks=True, client=client)
        with pytest.raises(error):
            provider.complete(system="s", prompt="p", schema=SCHEMA)


def test_openai_compatible_request_shape() -> None:
    response = SimpleNamespace(
        model="llama3.1", usage=SimpleNamespace(prompt_tokens=7, completion_tokens=3),
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content='```json\n{"x": "2"}\n```'))],
    )
    rec = _Recorder(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=rec))
    provider = OpenAICompatibleProvider(base_url="http://localhost:11434/v1", api_key="", model="llama3.1",
                                        client=client)
    result = provider.complete(system="sys", prompt="hi", schema=SCHEMA)
    assert rec.kwargs["response_format"] == {"type": "json_object"}
    assert "JSON Schema" in rec.kwargs["messages"][0]["content"]
    assert result.data == {"x": "2"}
    assert result.trace("t")["cost_known"] is False


def test_parse_json_object_tolerates_prose() -> None:
    assert parse_json_object('Sure! {"a": 1} done') == {"a": 1}
    with pytest.raises(LLMError):
        parse_json_object("no json here")
