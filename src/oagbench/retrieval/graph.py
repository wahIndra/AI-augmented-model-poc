"""Pipeline B: knowledge-graph retrieval with text-to-Cypher, falling back to neighbourhood expansion."""

import json
import time
from collections import deque
from typing import Any

from oagbench.generate import GoldQuestion
from oagbench.graph import Graph
from oagbench.llm import LLMError, LLMProvider
from oagbench.ontology import Ontology
from oagbench.retrieval.context import Context, ContextItem, fit_budget
from oagbench.retrieval.graph_store import CypherRejected, GraphStore

CYPHER_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["cypher"],
    "properties": {"cypher": {"type": "string"}},
}

CYPHER_RULES = """You translate questions about the company Kalimaya Digital into one read-only Neo4j Cypher query.

Rules:
- Read-only: MATCH, OPTIONAL MATCH, WITH, WHERE, RETURN, ORDER BY, LIMIT, UNWIND only.
- Find entities by their `key` property: the name lowercased, with every run of characters other than letters
  and digits replaced by one space (for example 'pg-core-03' -> 'pg core 03', 'Payments API' -> 'payments api').
  If unsure which name the graph uses, test membership in `keys` (all normalised names and aliases):
  WHERE 'payments api' IN s.keys
- DEPENDS_ON is transitive for impact questions: use variable-length paths such as -[:DEPENDS_ON*0..]->.
- Functional relations (DEPLOYED_IN, HOSTED_IN, ...) may hold superseded facts; prefer the latest `valid_from`.
- Return the evidence: include names, and `source_doc` of relationships or `source_docs` of nodes where possible.
- Never invent labels, relationship types or properties that are not in the schema.

Schema:
{schema}

Examples:
{examples}"""

MAX_EXPANSION_DEPTH = 3


class GraphRetriever:
    def __init__(self, name: str, store: GraphStore, ontology: Ontology, provider: LLMProvider | None,
                 few_shot: list[GoldQuestion]):
        self.name = name
        self.store = store
        self.provider = provider
        self.graph = store.snapshot()
        self.alias_index = self.graph.alias_index()
        examples = "\n\n".join(f"Question: {q.question}\nCypher: {q.cypher}" for q in few_shot if q.cypher)
        self.system = CYPHER_RULES.format(schema=ontology.schema_text(), examples=examples or "(none)")

    def retrieve(self, question: str, budget: int) -> Context:
        ctx = Context()
        start = time.perf_counter()
        if self.provider is not None and self.store.supports_cypher:
            ctx.items = fit_budget(self._cypher_items(question, ctx), budget)
        if not ctx.items:
            ctx.fallback_used = self.store.supports_cypher and self.provider is not None
            ctx.items = fit_budget(expand_neighbourhood(self.graph, question, self.alias_index), budget)
        ctx.timings_ms["retrieve"] = int((time.perf_counter() - start) * 1000)
        return ctx

    def _cypher_items(self, question: str, ctx: Context) -> list[ContextItem]:
        assert self.provider is not None
        prompt = f"Question: {question}"
        for attempt in range(2):
            try:
                result = self.provider.complete(system=self.system, prompt=prompt, schema=CYPHER_SCHEMA,
                                                max_tokens=4000)
                ctx.llm_calls.append(result.trace("cypher"))
                ctx.cypher = str((result.data or {}).get("cypher", ""))
                rows = self.store.run_cypher(ctx.cypher)
                return [_row_item(row) for row in rows]
            except (LLMError, CypherRejected) as exc:
                ctx.error = f"attempt {attempt + 1}: {exc}"
            except Exception as exc:  # neo4j raises driver-specific errors for bad Cypher
                ctx.error = f"attempt {attempt + 1}: {type(exc).__name__}: {exc}"
            prompt = f"Question: {question}\n\nYour previous query failed:\n{ctx.cypher}\nError: {ctx.error}\nFix it."
        return []


def _row_item(row: dict[str, Any]) -> ContextItem:
    docs = []
    for key, value in row.items():
        if "doc" in key.lower():
            docs += value if isinstance(value, list) else [value]
    content = "; ".join(f"{k} = {json.dumps(v, default=str)}" for k, v in row.items())
    return ContextItem("row", ",".join(str(d) for d in docs if d) or "graph", content)


def expand_neighbourhood(g: Graph, question: str, alias_index: dict[str, str] | None = None) -> list[ContextItem]:
    """Facts around the entities named in the question, nearest first (breadth-first, both directions)."""
    anchors = g.find_mentions(question, alias_index)
    items: list[ContextItem] = []
    seen_rel: set = set()
    seen_node: set[str] = set()
    queue = deque((a, 0) for a in anchors)
    while queue:
        node, depth = queue.popleft()
        if node in seen_node or node not in g.entities:
            continue
        seen_node.add(node)
        entity = g.entities[node]
        if entity.props:
            props = ", ".join(f"{k}={v}" for k, v in sorted(entity.props.items()))
            items.append(ContextItem("fact", ",".join(entity.source_docs[:2]) or "graph",
                                     f"{entity.name} ({entity.label}): {props}", 1.0 / (depth + 1)))
        if depth >= MAX_EXPANSION_DEPTH:
            continue
        for rel in g.relations_of(node):
            if rel in seen_rel:
                continue
            seen_rel.add(rel)
            since = f" (since {rel.valid_from})" if rel.valid_from else ""
            items.append(ContextItem(
                "fact", rel.source_doc or "graph",
                f"{g.name_of(rel.src)} ({g.entities[rel.src].label if rel.src in g.entities else '?'}) "
                f"-[{rel.type}]-> {g.name_of(rel.dst)} ({g.entities[rel.dst].label if rel.dst in g.entities else '?'})"
                f"{since}",
                1.0 / (depth + 1),
            ))
            other = rel.dst if rel.src == node else rel.src
            queue.append((other, depth + 1))
    return items
