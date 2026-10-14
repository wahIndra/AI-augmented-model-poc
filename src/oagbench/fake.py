"""Fake results for development without an AI provider (AI_PROVIDER=none).

Responses are simulated from the gold answers with per-pipeline, per-category accuracy profiles, so the harness,
scoring and dashboard can be built and tested offline. Every record is marked `mode: "fake"`; the numbers say
nothing about the real architectures.
"""

import random
from typing import Any

from oagbench.generate import Corpus, GoldQuestion
from oagbench.llm.base import Usage, cost_usd
from oagbench.retrieval.context import ContextItem, fit_budget
from oagbench.retrieval.graph import expand_neighbourhood

FAKE_MODEL = "fake-provider"
COST_MODEL = "claude-opus-5-5"  # simulated token counts are priced like this model, flagged as simulated

# Probability that a pipeline answers a category fully correctly. Shaped like the hypotheses in the BRD so the
# dashboard has something realistic to show; not a prediction.
PROFILES: dict[str, dict[str, float]] = {
    "A": {"direct": 0.85, "multihop_2": 0.55, "multihop_3plus": 0.25, "constraint": 0.30, "summary": 0.70,
          "unanswerable": 0.60},
    "B": {"direct": 0.80, "multihop_2": 0.78, "multihop_3plus": 0.68, "constraint": 0.72, "summary": 0.45,
          "unanswerable": 0.85},
    "C": {"direct": 0.88, "multihop_2": 0.80, "multihop_3plus": 0.70, "constraint": 0.72, "summary": 0.72,
          "unanswerable": 0.82},
    "B_oracle": {"direct": 0.95, "multihop_2": 0.92, "multihop_3plus": 0.88, "constraint": 0.90, "summary": 0.50,
                 "unanswerable": 0.90},
}
LATENCY_MS = {"A": (1800, 300), "B": (3300, 600), "C": (3700, 650), "B_oracle": (3100, 550)}


def fake_response(corpus: Corpus, q: GoldQuestion, pipeline: str, repeat: int, seed: int,
                  budget: int) -> dict[str, Any]:
    rng = random.Random(f"{seed}:{q.qid}:{pipeline}:{repeat}")
    g = corpus.canonical
    correct = rng.random() < PROFILES.get(pipeline, PROFILES["A"])[q.category]
    items, abstained = _items(g, q, correct, rng)
    answer = "NO_ANSWER" if abstained else _sentence(q, items)
    citations = (q.supporting_docs if correct else rng.sample([d.doc_id for d in corpus.documents], 2))[:6]

    context = _context(corpus, q, pipeline, rng, budget)
    uses_cypher = pipeline != "A"
    calls = []
    if uses_cypher:
        calls.append(_call("cypher", rng, 3800, 700))
    calls.append(_call("answer", rng, 6500, 900))
    mean, sd = LATENCY_MS.get(pipeline, (2500, 500))
    latency = max(300, int(rng.gauss(mean, sd)))
    fallback = uses_cypher and not correct and rng.random() < 0.4
    return {
        "qid": q.qid,
        "category": q.category,
        "pipeline": pipeline,
        "repeat": repeat,
        "mode": "fake",
        "answer": answer,
        "items": items,
        "citations": citations,
        "abstained": abstained,
        "context": {
            "items": [vars(i) | {"tokens": i.tokens} for i in context],
            "cypher": q.cypher if uses_cypher and not fallback else "",
            "fallback_used": fallback,
            "error": "simulated Cypher error" if fallback else "",
        },
        "llm_calls": calls,
        "latency_ms": {"retrieve": int(latency * 0.3), "answer": int(latency * 0.7), "total": latency},
        "cost_usd": round(sum(c["cost_usd"] for c in calls), 6),
        "error": "",
    }


def _call(purpose: str, rng: random.Random, tokens_in: int, tokens_out: int) -> dict[str, Any]:
    usage = Usage(int(rng.gauss(tokens_in, tokens_in * 0.1)), int(rng.gauss(tokens_out, tokens_out * 0.2)))
    return {"purpose": purpose, "model": FAKE_MODEL, "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens, "cache_read_tokens": 0, "latency_ms": 0,
            "cost_usd": cost_usd(COST_MODEL, usage), "cost_known": True, "simulated": True}


def _items(g, q: GoldQuestion, correct: bool, rng: random.Random) -> tuple[list[str], bool]:
    if q.answer_type == "no_answer":
        if correct:
            return [], True
        return [rng.choice(g.by_label("Team")).name], False
    if correct:
        return list(q.answer), False
    if rng.random() < 0.2:
        return [], True  # false abstention
    if q.answer_type == "number":
        return [str(max(0, int(q.answer[0]) + rng.choice([-1, 1, 2])))], False
    if q.answer_type == "value":
        return [rng.choice(["postgres", "redis", "kafka", "mongo", "s3"])], False
    if q.answer_type == "summary":
        keep = max(1, int(len(q.answer) * rng.uniform(0.3, 0.7)))
        return rng.sample(q.answer, keep), False
    label = g.entities[q.answer_ids[0]].label if q.answer_ids else "Team"
    wrong = rng.choice(g.by_label(label)).name
    kept = rng.sample(q.answer, k=max(0, len(q.answer) - rng.randint(1, max(1, len(q.answer) // 2))))
    return sorted({*kept, wrong}), False


def _sentence(q: GoldQuestion, items: list[str]) -> str:
    if not items:
        return "NO_ANSWER"
    if len(items) == 1:
        return f"{items[0]}."
    return f"{', '.join(items[:-1])} and {items[-1]}."


def _context(corpus: Corpus, q: GoldQuestion, pipeline: str, rng: random.Random, budget: int) -> list[ContextItem]:
    docs = [corpus.doc(d) for d in q.supporting_docs]
    others = rng.sample(corpus.documents, 4)
    chunks = [ContextItem("chunk", d.doc_id, d.text[:900], round(rng.uniform(0.4, 0.9), 3))
              for d in [*docs, *others] if d is not None]
    rng.shuffle(chunks)
    if pipeline == "A":
        return fit_budget(chunks, budget)
    facts = expand_neighbourhood(corpus.canonical, q.question)
    if pipeline == "C":
        return fit_budget(facts, budget // 2) + fit_budget(chunks, budget // 2)
    return fit_budget(facts, budget)
