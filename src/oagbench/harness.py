"""Benchmark run loop. Real mode calls the configured AI provider; fake mode (AI_PROVIDER=none) simulates."""

import hashlib
import json
import logging
import subprocess
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from oagbench import __version__
from oagbench.answer import ANSWER_SYSTEM, answer_question
from oagbench.config import Settings, StoreBackend
from oagbench.fake import fake_response
from oagbench.generate import Corpus, GoldQuestion
from oagbench.graph import Graph
from oagbench.llm import LLMProvider
from oagbench.ontology import load_ontology
from oagbench.retrieval.context import Retriever
from oagbench.retrieval.graph import CYPHER_RULES, GraphRetriever
from oagbench.retrieval.graph_store import GraphStore, MemoryGraphStore, Neo4jGraphStore
from oagbench.retrieval.hybrid import HybridRetriever
from oagbench.retrieval.vector import VectorRetriever, build_embedder, open_vector_store

log = logging.getLogger(__name__)

PIPELINES = ("A", "B", "C", "B_oracle")
FEW_SHOT = 8


def extracted_graph_path(corpus: Corpus) -> Path:
    return corpus.path / "index" / "extracted-graph.json"


def few_shot_examples(corpus: Corpus) -> list[GoldQuestion]:
    """One dev question per template, so examples cover query shapes without touching the test split."""
    seen: set[str] = set()
    examples = []
    for q in corpus.questions("dev"):
        if q.cypher and q.template not in seen:
            seen.add(q.template)
            examples.append(q)
    return examples[:FEW_SHOT]


def open_graph_store(settings: Settings, corpus: Corpus, oracle: bool) -> GraphStore:
    if settings.store_backend is StoreBackend.SERVICES:
        return Neo4jGraphStore(
            settings.neo4j_oracle_uri if oracle else settings.neo4j_uri,
            settings.neo4j_user, settings.neo4j_password.get_secret_value(),
            query_user=settings.neo4j_readonly_user,
            query_password=settings.neo4j_readonly_password.get_secret_value(),
        )
    if oracle:
        return MemoryGraphStore(corpus.canonical)
    path = extracted_graph_path(corpus)
    if not path.exists():
        raise FileNotFoundError(f"no extracted graph at {path}; run `oagbench ingest graph` first")
    return MemoryGraphStore(Graph.load(path))


def build_retrievers(settings: Settings, corpus: Corpus, provider: LLMProvider,
                     pipelines: list[str]) -> dict[str, Retriever]:
    ontology = load_ontology()
    examples = few_shot_examples(corpus)
    built: dict[str, Retriever] = {}
    if {"A", "C"} & set(pipelines):
        embedder = build_embedder(settings)
        built["A"] = VectorRetriever(open_vector_store(settings, corpus.path, embedder), embedder)
    if {"B", "C"} & set(pipelines):
        built["B"] = GraphRetriever("B", open_graph_store(settings, corpus, oracle=False), ontology, provider,
                                    examples)
    if "C" in pipelines:
        built["C"] = HybridRetriever(built["A"], built["B"])
    if "B_oracle" in pipelines:
        built["B_oracle"] = GraphRetriever("B_oracle", open_graph_store(settings, corpus, oracle=True), ontology,
                                           provider, examples)
    return {p: built[p] for p in pipelines}


def run_benchmark(settings: Settings, corpus: Corpus, provider: LLMProvider | None, *, split: str,
                  pipelines: list[str], repeats: int, limit: int | None = None,
                  progress: Callable[[int, int], None] | None = None) -> Path:
    unknown = set(pipelines) - set(PIPELINES)
    if unknown:
        raise ValueError(f"unknown pipelines {sorted(unknown)}; choose from {PIPELINES}")
    mode = "real" if provider is not None else "fake"
    questions = corpus.questions(split)[:limit]
    started = datetime.now(UTC)
    run_id = f"{started:%Y%m%d-%H%M%S}-{mode}-{corpus.corpus_id}-{uuid.uuid4().hex[:4]}"
    out = settings.results_dir / run_id
    out.mkdir(parents=True, exist_ok=False)

    record = {
        "run_id": run_id,
        "mode": mode,
        "started_at": started.isoformat(),
        "corpus_id": corpus.corpus_id,
        "corpus_hash": corpus.manifest["corpus_hash"],
        "split": split,
        "pipelines": pipelines,
        "repeats": repeats,
        "questions": len(questions),
        "settings": settings.describe(),
        "prompt_hashes": {"answer": _hash(ANSWER_SYSTEM), "cypher_rules": _hash(CYPHER_RULES)},
        "oagbench_version": __version__,
        "git_commit": _git_commit(),
    }
    if mode == "fake":
        record["warning"] = "FAKE RESULTS: AI_PROVIDER=none, responses are simulated from gold answers."
        log.warning(record["warning"])
    if split != "test":
        record["warning_split"] = "Few-shot Cypher examples come from the dev split; dev scores are optimistic."
    (out / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")

    tasks = [(q, p, r) for r in range(1, repeats + 1) for q in questions for p in pipelines]
    seed = corpus.manifest["seed"]
    budget = settings.context_token_budget

    if provider is None:
        responses = [fake_response(corpus, q, p, r, seed, budget) for q, p, r in tasks]
        if progress:
            progress(len(tasks), len(tasks))
    else:
        retrievers = build_retrievers(settings, corpus, provider, pipelines)
        responses = _run_real(settings, provider, retrievers, tasks, progress)

    for n, resp in enumerate(responses, start=1):
        resp["response_id"] = f"{run_id}:{n:05d}"
        resp["run_id"] = run_id
    with (out / "responses.jsonl").open("w", encoding="utf-8") as fh:
        for resp in responses:
            fh.write(json.dumps(resp, default=str) + "\n")

    record["finished_at"] = datetime.now(UTC).isoformat()
    record["responses"] = len(responses)
    record["cost_usd"] = round(sum(r["cost_usd"] for r in responses), 4)
    record["stopped_by_budget"] = len(responses) < len(tasks)
    (out / "run.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    return out


def _run_real(settings: Settings, provider: LLMProvider, retrievers: dict[str, Retriever],
              tasks: list[tuple[GoldQuestion, str, int]],
              progress: Callable[[int, int], None] | None) -> list[dict[str, Any]]:
    spent = 0.0
    lock = threading.Lock()
    stop = threading.Event()
    results: list[dict[str, Any]] = []

    def one(q: GoldQuestion, pipeline: str, repeat: int) -> dict[str, Any] | None:
        nonlocal spent
        if stop.is_set():
            return None
        start = time.perf_counter()
        ctx = retrievers[pipeline].retrieve(q.question, settings.context_token_budget)
        ans = answer_question(provider, q.question, ctx)
        calls = ctx.llm_calls + ans.llm_calls
        cost = sum(c["cost_usd"] for c in calls)
        with lock:
            spent += cost
            if spent >= settings.run_budget_usd:
                stop.set()
        return {
            "qid": q.qid, "category": q.category, "pipeline": pipeline, "repeat": repeat, "mode": "real",
            "answer": ans.answer, "items": ans.items, "citations": ans.citations, "abstained": ans.abstained,
            "context": ctx.to_trace(), "llm_calls": calls,
            "latency_ms": {"retrieve": ctx.timings_ms.get("retrieve", 0), "answer": ans.latency_ms,
                           "total": int((time.perf_counter() - start) * 1000)},
            "cost_usd": round(cost, 6), "error": ans.error,
        }

    with ThreadPoolExecutor(max_workers=settings.llm_concurrency) as pool:
        futures = [pool.submit(one, *t) for t in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result is not None:
                results.append(result)
            if progress:
                progress(done, len(tasks))
    if stop.is_set():
        log.warning("run stopped at the budget of USD %.2f", settings.run_budget_usd)
    order = {(q.qid, p, r): i for i, (q, p, r) in enumerate(tasks)}
    return sorted(results, key=lambda r: order[(r["qid"], r["pipeline"], r["repeat"])])


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              timeout=5, check=False).stdout.strip() or "uncommitted"
    except OSError:
        return "unknown"
