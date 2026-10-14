import json

import pandas as pd

from oagbench.harness import run_benchmark
from oagbench.retrieval.vector import build_embedder, build_vector_index
from oagbench.scoring import score_run


def test_fake_run_is_labelled_and_scored(settings, corpus) -> None:
    out = run_benchmark(settings, corpus, None, split="test", pipelines=["A", "B", "C", "B_oracle"], repeats=2)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["mode"] == "fake"
    assert "FAKE" in run["warning"]
    responses = [json.loads(line) for line in (out / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(responses) == len(corpus.questions("test")) * 4 * 2
    assert all(r["mode"] == "fake" for r in responses)

    summary = score_run(out, corpus)
    overall = {c["pipeline"]: c["correct"] for c in summary["cells"] if c["category"] == "all"}
    assert set(overall) == {"A", "B", "C", "B_oracle"}
    assert overall["B_oracle"] > overall["A"]  # follows the simulated profiles
    assert len(pd.read_parquet(out / "scores.parquet")) == len(responses)


def test_fake_run_is_deterministic(settings, corpus) -> None:
    a = run_benchmark(settings, corpus, None, split="test", pipelines=["A"], repeats=1)
    b = run_benchmark(settings, corpus, None, split="test", pipelines=["A"], repeats=1)
    strip = lambda p: [{k: v for k, v in json.loads(line).items() if k not in {"response_id", "run_id"}}  # noqa: E731
                       for line in (p / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert strip(a) == strip(b)


def test_real_mode_with_stub_provider(settings, corpus, stub_provider) -> None:
    """Exercises the real-mode path (retrievers, shared answer step, tracing) with the memory backend."""
    build_vector_index(settings, corpus.path, corpus.documents, build_embedder(settings))
    corpus.canonical.save(corpus.path / "index" / "extracted-graph.json")  # stand-in for `ingest graph`

    out = run_benchmark(settings, corpus, stub_provider, split="test", pipelines=["A", "B", "C", "B_oracle"],
                        repeats=1, limit=3)
    responses = [json.loads(line) for line in (out / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(responses) == 12
    assert all(r["mode"] == "real" for r in responses)
    assert all(r["llm_calls"] and r["llm_calls"][-1]["purpose"] == "answer" for r in responses)
    # The memory backend cannot run Cypher, so graph pipelines use neighbourhood expansion and make no Cypher call.
    graph_calls = [c["purpose"] for r in responses if r["pipeline"] != "A" for c in r["llm_calls"]]
    assert "cypher" not in graph_calls
    # Every pipeline sends the same answer system prompt.
    assert len({c["system"] for c in stub_provider.calls}) == 1

    summary = score_run(out, corpus, stub_provider)
    assert summary["judged"] is True
    assert "judge_correctness" in summary["metrics"]


def test_run_stops_at_budget(settings, corpus, stub_provider) -> None:
    build_vector_index(settings, corpus.path, corpus.documents, build_embedder(settings))
    settings.run_budget_usd = 0.0  # stub calls are unpriced, so any spend >= 0 trips the guard
    settings.llm_concurrency = 1
    out = run_benchmark(settings, corpus, stub_provider, split="test", pipelines=["A"], repeats=1)
    run = json.loads((out / "run.json").read_text(encoding="utf-8"))
    assert run["stopped_by_budget"] is True
    assert run["responses"] < len(corpus.questions("test"))
