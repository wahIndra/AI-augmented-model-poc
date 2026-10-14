"""Runtime configuration, read from environment variables and an optional .env file."""

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class AIProvider(StrEnum):
    NONE = "none"
    ANTHROPIC = "anthropic"
    OPENAI_COMPATIBLE = "openai_compatible"


class StoreBackend(StrEnum):
    MEMORY = "memory"
    SERVICES = "services"


class EmbedderKind(StrEnum):
    HASH = "hash"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class ConfigError(RuntimeError):
    pass


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    ai_provider: AIProvider = AIProvider.NONE

    anthropic_api_key: SecretStr | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: str = "medium"
    anthropic_fallbacks: bool = True

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: SecretStr | None = None
    openai_model: str = ""

    judge_model: str = ""

    store_backend: StoreBackend = StoreBackend.MEMORY
    qdrant_url: str = "http://localhost:6333"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_oracle_uri: str = "bolt://localhost:7688"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("")
    neo4j_readonly_user: str = ""
    neo4j_readonly_password: SecretStr = SecretStr("")

    embedder: EmbedderKind = EmbedderKind.HASH
    embedding_model: str = "BAAI/bge-base-en-v1.5"

    data_dir: Path = Path("data")
    results_dir: Path = Path("results")
    context_token_budget: int = 6000
    run_budget_usd: float = 80.0
    llm_concurrency: int = 4

    @property
    def ai_active(self) -> bool:
        return self.ai_provider is not AIProvider.NONE

    def validate_provider(self) -> None:
        """Fail fast on a half-configured provider instead of silently falling back to fake mode."""
        if self.ai_provider is AIProvider.ANTHROPIC and not _filled(self.anthropic_api_key):
            raise ConfigError("AI_PROVIDER=anthropic but ANTHROPIC_API_KEY is empty")
        if self.ai_provider is AIProvider.OPENAI_COMPATIBLE and not self.openai_model:
            raise ConfigError("AI_PROVIDER=openai_compatible but OPENAI_MODEL is empty")

    def describe(self) -> dict[str, str]:
        """Settings safe to print or store in a run record (no secrets)."""
        model = {
            AIProvider.NONE: "-",
            AIProvider.ANTHROPIC: self.anthropic_model,
            AIProvider.OPENAI_COMPATIBLE: self.openai_model,
        }[self.ai_provider]
        return {
            "ai_provider": self.ai_provider.value,
            "model": model,
            "judge_model": self.judge_model or model,
            "effort": self.anthropic_effort if self.ai_provider is AIProvider.ANTHROPIC else "-",
            "store_backend": self.store_backend.value,
            "embedder": self.embedder.value,
            "embedding_model": self.embedding_model if self.embedder is EmbedderKind.SENTENCE_TRANSFORMERS else "-",
            "context_token_budget": str(self.context_token_budget),
            "run_budget_usd": str(self.run_budget_usd),
        }


def _filled(secret: SecretStr | None) -> bool:
    return secret is not None and bool(secret.get_secret_value().strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
