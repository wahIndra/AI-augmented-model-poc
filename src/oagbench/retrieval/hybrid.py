"""Pipeline C: vector and graph retrieval in parallel, graph facts first, then chunks."""

import time
from concurrent.futures import ThreadPoolExecutor

from oagbench.retrieval.context import Context, ContextItem, Retriever, fit_budget


def _tokens(items: list[ContextItem]) -> int:
    return sum(i.tokens for i in items)


class HybridRetriever:
    name = "C"

    def __init__(self, vector: Retriever, graph: Retriever):
        self.vector = vector
        self.graph = graph

    def retrieve(self, question: str, budget: int) -> Context:
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=2) as pool:
            graph_future = pool.submit(self.graph.retrieve, question, budget)
            vector_future = pool.submit(self.vector.retrieve, question, budget)
            g, v = graph_future.result(), vector_future.result()

        # Each side gets half the budget; whatever the vector side leaves unused goes to the graph side,
        # and the vector side then fills what the graph side leaves.
        half = budget // 2
        graph_budget = half + max(0, half - _tokens(fit_budget(v.items, half)))
        graph_items = fit_budget(g.items, graph_budget)
        vector_items = fit_budget(v.items, budget - _tokens(graph_items))
        return Context(
            items=graph_items + vector_items,
            cypher=g.cypher,
            fallback_used=g.fallback_used,
            error=g.error,
            llm_calls=g.llm_calls + v.llm_calls,
            timings_ms={"retrieve": int((time.perf_counter() - start) * 1000)},
        )
