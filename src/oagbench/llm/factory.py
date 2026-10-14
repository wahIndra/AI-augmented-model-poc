from oagbench.config import AIProvider, Settings
from oagbench.llm.base import LLMProvider


def build_provider(settings: Settings) -> LLMProvider | None:
    """The configured provider, or None when AI_PROVIDER=none (fake mode)."""
    settings.validate_provider()
    match settings.ai_provider:
        case AIProvider.NONE:
            return None
        case AIProvider.ANTHROPIC:
            from oagbench.llm.anthropic_provider import AnthropicProvider

            assert settings.anthropic_api_key is not None
            return AnthropicProvider(
                api_key=settings.anthropic_api_key.get_secret_value(),
                model=settings.anthropic_model,
                effort=settings.anthropic_effort,
                fallbacks=settings.anthropic_fallbacks,
            )
        case AIProvider.OPENAI_COMPATIBLE:
            from oagbench.llm.openai_provider import OpenAICompatibleProvider

            key = settings.openai_api_key.get_secret_value() if settings.openai_api_key else ""
            return OpenAICompatibleProvider(
                base_url=settings.openai_base_url, api_key=key, model=settings.openai_model
            )
