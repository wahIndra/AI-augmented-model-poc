"""Deterministic scoring against the gold answers, optional LLM judging, and aggregation with bootstrap CIs."""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from oagbench.generate import Corpus, GoldQuestion
from oagbench.graph import Graph, normalize
from oagbench.llm import LLMProvider

SUMMARY_PASS = 0.8
BOOTSTRAP_SAMPLES = 10_000


def load_responses(run_dir: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (run_dir / "responses.jsonl").read_text(encoding="utf-8").splitlines()
            if line]


def resolve_items(g: Graph, items: list[str], index: dict[str, str]) -> set[str]:
    """Map answer strings to canonical entity ids; unresolvable strings stay as normalised text."""
    resolved: set[str] = set()
    for item in items:
        key = normalize(item)
        if key in index:
            resolved.add(index[key])
            continue
        mentions = g.find_mentions(item, index)
        resolved.update(mentions or {f"text:{key}"})
    return resolved


def score_response(g: Graph, index: dict[str, str], q: GoldQuestion, resp: dict[str, Any]) -> dict[str, float]:
    items = resp["items"]
    abstained = bool(resp["abstained"]) or resp["answer"].strip().upper() == "NO_ANSWER"
    supporting = set(q.supporting_docs)
    cited = {d.strip() for c in resp["citations"] for d in str(c).split(",")}
    citation_recall = len(supporting & cited) / len(supporting) if supporting else float("nan")
    base = {"abstained": float(abstained), "citation_recall": citation_recall}

    if q.answer_type == "no_answer":
        ok = float(abstained)
        return base | {"correct": ok, "f1": ok, "precision": ok, "recall": ok}
    if abstained or (not items and not resp["answer"].strip()):
        return base | {"correct": 0.0, "f1": 0.0, "precision": 0.0, "recall": 0.0}

    if q.answer_type == "number":
        text = " ".join(items) or resp["answer"]
        found = re.findall(r"-?\d+", text)
        ok = float(bool(found) and int(found[0]) == int(q.answer[0]))
        return base | {"correct": ok, "f1": ok, "precision": ok, "recall": ok}
    if q.answer_type == "value":
        gold = normalize(q.answer[0])
        ok = float(any(normalize(i) == gold for i in items) or f" {gold} " in f" {normalize(resp['answer'])} ")
        return base | {"correct": ok, "f1": ok, "precision": ok, "recall": ok}
    if q.answer_type == "summary":
        mentioned = set(g.find_mentions(" | ".join([resp["answer"], *items]), index))
        coverage = len(mentioned & set(q.answer_ids)) / len(q.answer_ids)
        return base | {"correct": float(coverage >= SUMMARY_PASS), "f1": coverage, "precision": float("nan"),
                       "recall": coverage}

    predicted = resolve_items(g, items or [resp["answer"]], index)
    gold_ids = set(q.answer_ids)
    hits = len(predicted & gold_ids)
    precision = hits / len(predicted) if predicted else 0.0
    recall = hits / len(gold_ids) if gold_ids else 0.0
    f1 = 2 * precision * recall / (precision + recall) if hits else 0.0
    return base | {"correct": float(predicted == gold_ids), "f1": f1, "precision": precision, "recall": recall}


def score_run(run_dir: Path, corpus: Corpus, provider: LLMProvider | None = None, judge_model: str | None = None,
              concurrency: int = 4) -> dict[str, Any]:
    responses = load_responses(run_dir)
    gold = {q.qid: q for q in corpus.gold}
    g = corpus.canonical
    index = g.alias_index()
    rows = []
    for resp in responses:
        q = gold[resp["qid"]]
        rows.append({
            "response_id": resp["response_id"], "qid": q.qid, "category": q.category, "template": q.template,
            "pipeline": resp["pipeline"], "repeat": resp["repeat"], "mode": resp["mode"],
            **score_response(g, index, q, resp),
            "latency_ms": resp["latency_ms"]["total"], "cost_usd": resp["cost_usd"],
            "fallback_used": float(resp["context"].get("fallback_used", False)),
            "context_tokens": sum(i.get("tokens", 0) for i in resp["context"]["items"]),
        })

    if provider is not None:
        from oagbench.judge import judge_response

        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            judged = list(pool.map(lambda r: judge_response(provider, judge_model, gold[r["qid"]], r), responses))
        for row, j in zip(rows, judged, strict=True):
            row["judge_faithfulness"] = j.get("judge_faithfulness", float("nan"))
            row["judge_correctness"] = j.get("judge_correctness", float("nan"))
            row["judge_cost_usd"] = j.get("judge_call", {}).get("cost_usd", 0.0)

    df = pd.DataFrame(rows)
    df.to_parquet(run_dir / "scores.parquet", index=False)
    summary = aggregate(df)
    summary["run"] = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    summary["judged"] = provider is not None
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=1, default=_json_default), encoding="utf-8")
    return summary


def _json_default(value: Any) -> Any:
    if isinstance(value, np.floating | np.integer):
        return value.item()
    raise TypeError(type(value))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = values[~np.isnan(values)]
    if len(values) < 2:
        return (float("nan"), float("nan"))
    samples = rng.choice(values, size=(BOOTSTRAP_SAMPLES, len(values)), replace=True).mean(axis=1)
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def aggregate(df: pd.DataFrame, baseline: str = "A") -> dict[str, Any]:
    """Per pipeline x category means with 95% bootstrap CIs; repeats are averaged per question first."""
    rng = np.random.default_rng(0)
    metrics = [m for m in ("correct", "f1", "citation_recall", "judge_correctness", "judge_faithfulness")
               if m in df]
    per_q = df.groupby(["pipeline", "category", "qid"])[metrics].mean().reset_index()
    cells = []
    for (pipeline, category), part in [*per_q.groupby(["pipeline", "category"]),
                                       *((k, v) for k, v in per_q.assign(category="all").groupby(
                                           ["pipeline", "category"]))]:
        cell = {"pipeline": pipeline, "category": category, "questions": int(part["qid"].nunique())}
        for m in metrics:
            values = part[m].to_numpy(dtype=float)
            low, high = bootstrap_ci(values, rng)
            cell[m] = float(np.nanmean(values)) if (~np.isnan(values)).any() else float("nan")
            cell[f"{m}_ci"] = [low, high]
        cells.append(cell)

    diffs = []
    if baseline in set(df["pipeline"]):
        wide = per_q.pivot_table(index=["category", "qid"], columns="pipeline", values="correct").reset_index()
        for pipeline in sorted(set(df["pipeline"]) - {baseline}):
            for category, part in [*wide.groupby("category"), ("all", wide)]:
                d = (part[pipeline] - part[baseline]).to_numpy(dtype=float)
                low, high = bootstrap_ci(d, rng)
                diffs.append({"pipeline": pipeline, "baseline": baseline, "category": category,
                              "diff_correct": float(np.nanmean(d)), "ci": [low, high]})

    ops = []
    for pipeline, part in df.groupby("pipeline"):
        ops.append({
            "pipeline": pipeline,
            "responses": len(part),
            "latency_p50_ms": float(part["latency_ms"].quantile(0.5)),
            "latency_p95_ms": float(part["latency_ms"].quantile(0.95)),
            "cost_per_query_usd": float(part["cost_usd"].mean()),
            "cost_total_usd": float(part["cost_usd"].sum()),
            "fallback_rate": float(part["fallback_used"].mean()),
            "abstain_rate": float(part["abstained"].mean()),
            "context_tokens_mean": float(part["context_tokens"].mean()),
        })
    return {"cells": cells, "paired_vs_baseline": diffs, "operations": ops, "metrics": metrics}
