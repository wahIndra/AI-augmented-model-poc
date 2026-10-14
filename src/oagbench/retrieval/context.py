"""Retrieved context shared by all pipelines; the answer step only ever sees this."""

from dataclasses import asdict, dataclass, field
from typing import Any, Protocol


def approx_tokens(text: str) -> int:
    """Cheap token estimate (about 4 characters per token for English)."""
    return max(1, len(text) // 4)


@dataclass
class ContextItem:
    kind: str  # chunk | fact | row
    source_doc: str
    content: str
    score: float = 0.0

    @property
    def tokens(self) -> int:
        return approx_tokens(self.content)


@dataclass
class Context:
    items: list[ContextItem] = field(default_factory=list)
    cypher: str = ""
    fallback_used: bool = False
    error: str = ""
    llm_calls: list[dict[str, Any]] = field(default_factory=list)
    timings_ms: dict[str, int] = field(default_factory=dict)

    @property
    def tokens(self) -> int:
        return sum(i.tokens for i in self.items)

    def render(self) -> str:
        return "\n\n".join(f"[{n}] (doc: {i.source_doc}) {i.content}" for n, i in enumerate(self.items, start=1))

    def to_trace(self) -> dict[str, Any]:
        return {
            "items": [asdict(i) | {"tokens": i.tokens} for i in self.items],
            "cypher": self.cypher,
            "fallback_used": self.fallback_used,
            "error": self.error,
        }


def fit_budget(items: list[ContextItem], budget: int) -> list[ContextItem]:
    kept, used = [], 0
    for item in items:
        if used + item.tokens > budget:
            break
        kept.append(item)
        used += item.tokens
    return kept


class Retriever(Protocol):
    name: str

    def retrieve(self, question: str, budget: int) -> Context: ...
