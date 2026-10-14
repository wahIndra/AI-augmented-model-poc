import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from oagbench.config import Settings
from oagbench.generate import Corpus, generate_corpus, load_corpus
from oagbench.llm.base import LLMResult, Usage


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> Corpus:
    return load_corpus(generate_corpus(tmp_path_factory.mktemp("data"), "small", 7))


@pytest.fixture
def settings(tmp_path: Path, corpus: Corpus) -> Settings:
    # _env_file=None keeps a developer's local .env out of the tests.
    return Settings(_env_file=None, ai_provider="none", data_dir=corpus.path.parent,
                    results_dir=tmp_path / "results")


class StubProvider:
    """Deterministic stand-in for a real provider; answers with the first entity named in the context."""

    name = "stub"
    model = "stub-model"

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def complete(self, *, system: str, prompt: str, schema: dict | None = None, max_tokens: int = 8000,
                 model: str | None = None) -> LLMResult:
        self.calls.append({"system": system, "prompt": prompt, "schema": schema})
        props = (schema or {}).get("properties", {})
        if "cypher" in props:
            data: dict[str, Any] = {"cypher": "MATCH (n) RETURN n.name AS name LIMIT 1"}
        elif "faithfulness" in props:
            data = {"faithfulness": 1.0, "correctness": 0.5, "reason": "stub"}
        else:
            first = prompt.split("] (doc: ", 1)
            doc = first[1].split(")", 1)[0] if len(first) > 1 else ""
            data = {"answer": "stub", "items": ["stub"], "citations": [doc] if doc else [], "abstained": not doc}
        return LLMResult(json.dumps(data), data, Usage(100, 20), self.model, 5)


@pytest.fixture
def stub_provider() -> StubProvider:
    return StubProvider()


def fake_sdk_response(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)
