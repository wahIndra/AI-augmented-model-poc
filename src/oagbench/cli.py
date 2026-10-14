"""oagbench command line."""

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Annotated

import typer

from oagbench.config import ConfigError, StoreBackend, get_settings
from oagbench.generate import corpus_id, generate_corpus, load_corpus
from oagbench.llm import build_provider

app = typer.Typer(no_args_is_help=True, add_completion=False, help="OAG vs RAG benchmark")
ingest_app = typer.Typer(no_args_is_help=True, help="Build the indexes for a corpus")
app.add_typer(ingest_app, name="ingest")

CorpusOpt = Annotated[str, typer.Option("--corpus", "-c", help="Corpus id, e.g. small-s42")]
DEFAULT_CORPUS = "small-s42"


def _settings():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    return get_settings()


def _provider(settings):
    try:
        return build_provider(settings)
    except ConfigError as exc:
        typer.secho(f"Configuration error: {exc}", fg="red", err=True)
        raise typer.Exit(2) from None


def _corpus(settings, name: str):
    try:
        return load_corpus(settings.data_dir / name)
    except FileNotFoundError as exc:
        typer.secho(str(exc), fg="red", err=True)
        raise typer.Exit(1) from None


@app.command()
def status() -> None:
    """Show configuration (no secrets), mode, corpora and runs."""
    settings = _settings()
    _provider(settings)
    mode = "REAL (AI provider active)" if settings.ai_active else "FAKE (AI_PROVIDER=none, results are simulated)"
    typer.echo(f"Mode: {mode}")
    for key, value in settings.describe().items():
        typer.echo(f"  {key:22} {value}")
    corpora = sorted(p.name for p in settings.data_dir.glob("*") if (p / "manifest.json").exists())
    runs = sorted(p.name for p in settings.results_dir.glob("*") if (p / "run.json").exists())
    typer.echo(f"Corpora: {', '.join(corpora) or '-'}")
    typer.echo(f"Runs: {len(runs)}" + (f" (latest {runs[-1]})" if runs else ""))


@app.command()
def generate(profile: str = "small", seed: int = 42) -> None:
    """Generate the canonical graph, documents and gold questions (no AI calls)."""
    settings = _settings()
    path = generate_corpus(settings.data_dir, profile, seed)
    corpus = load_corpus(path)
    typer.echo(f"Corpus {corpus.corpus_id}: {len(corpus.documents)} documents, {len(corpus.gold)} questions "
               f"-> {path}")


@ingest_app.command("vector")
def ingest_vector(corpus: CorpusOpt = DEFAULT_CORPUS) -> None:
    """Chunk and embed documents (pipeline A). Local embeddings, no AI provider needed."""
    from oagbench.retrieval.vector import build_embedder, build_vector_index

    settings = _settings()
    c = _corpus(settings, corpus)
    stats = build_vector_index(settings, c.path, c.documents, build_embedder(settings))
    typer.echo(json.dumps(stats))


@ingest_app.command("graph")
def ingest_graph(corpus: CorpusOpt = DEFAULT_CORPUS, limit: int | None = None) -> None:
    """Extract the knowledge graph from documents with the AI provider (pipeline B)."""
    from oagbench.extract import compare_with_canonical, extract_graph
    from oagbench.harness import extracted_graph_path, open_graph_store
    from oagbench.ontology import load_ontology

    settings = _settings()
    provider = _provider(settings)
    if provider is None:
        typer.secho("Graph extraction needs an AI provider (AI_PROVIDER=none). In fake mode `oagbench run` "
                    "simulates pipeline B without an extracted graph.", fg="yellow", err=True)
        raise typer.Exit(1)
    c = _corpus(settings, corpus)
    graph, report = extract_graph(c.documents[:limit], provider, load_ontology(), settings.llm_concurrency)
    report.quality = compare_with_canonical(graph, c.canonical)
    path = extracted_graph_path(c)
    path.parent.mkdir(parents=True, exist_ok=True)
    graph.save(path)
    (path.parent / "extraction-report.json").write_text(
        json.dumps(report.summary() | {"merge_log": report.merges}, indent=1), encoding="utf-8")
    if settings.store_backend is StoreBackend.SERVICES:
        open_graph_store(settings, c, oracle=False).load(graph)
    typer.echo(json.dumps(report.summary(), indent=1))


@ingest_app.command("oracle")
def ingest_oracle(corpus: CorpusOpt = DEFAULT_CORPUS) -> None:
    """Load the canonical graph into the oracle Neo4j (services backend only; memory needs no step)."""
    from oagbench.harness import open_graph_store

    settings = _settings()
    if settings.store_backend is not StoreBackend.SERVICES:
        typer.echo("STORE_BACKEND=memory: the oracle graph is read straight from canonical.json.")
        return
    c = _corpus(settings, corpus)
    open_graph_store(settings, c, oracle=True).load(c.canonical)
    typer.echo(f"Loaded {len(c.canonical.entities)} entities, {len(c.canonical.relations)} relations.")


@app.command()
def run(
    corpus: CorpusOpt = DEFAULT_CORPUS,
    split: str = "test",
    pipelines: str = "A,B,C,B_oracle",
    repeats: int = 1,
    limit: Annotated[int | None, typer.Option(help="Only the first N questions (smoke tests)")] = None,
    score: Annotated[bool, typer.Option(help="Score the run when it finishes")] = True,
) -> None:
    """Run the benchmark. With AI_PROVIDER=none this writes clearly labelled FAKE results."""
    from oagbench.harness import run_benchmark

    settings = _settings()
    provider = _provider(settings)
    if provider is None:
        typer.secho("AI_PROVIDER=none: writing FAKE results (simulated from gold answers).", fg="yellow")
    c = _corpus(settings, corpus)
    with typer.progressbar(length=1, label="Running") as bar:
        def progress(done: int, total: int) -> None:
            bar.length = total
            bar.update(done - bar.pos)

        out = run_benchmark(settings, c, provider, split=split, pipelines=pipelines.split(","), repeats=repeats,
                            limit=limit, progress=progress)
    typer.echo(f"Run written to {out}")
    if score:
        _score(settings, out, judge=False)


@app.command("score")
def score_cmd(
    run_id: str,
    judge: Annotated[bool, typer.Option(help="Also run the LLM judge (real AI provider only)")] = False,
) -> None:
    """Score a run against the gold answers."""
    settings = _settings()
    _score(settings, settings.results_dir / run_id, judge)


def _score(settings, run_dir: Path, judge: bool) -> None:
    from oagbench.scoring import score_run

    record = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    provider = _provider(settings) if judge else None
    if judge and provider is None:
        typer.secho("--judge needs an AI provider; scoring deterministically only.", fg="yellow")
    summary = score_run(run_dir, _corpus(settings, record["corpus_id"]), provider, settings.judge_model or None,
                        settings.llm_concurrency)
    if record["mode"] == "fake":
        typer.secho("FAKE RESULTS: numbers below are simulated.", fg="yellow")
    typer.echo(f"{'pipeline':10} {'correct':>8} {'95% CI':>16} {'p50 ms':>8} {'$/query':>8}")
    ops = {o["pipeline"]: o for o in summary["operations"]}
    for cell in summary["cells"]:
        if cell["category"] == "all":
            low, high = cell["correct_ci"]
            o = ops[cell["pipeline"]]
            typer.echo(f"{cell['pipeline']:10} {cell['correct']:8.2f} {f'[{low:.2f}, {high:.2f}]':>16} "
                       f"{o['latency_p50_ms']:8.0f} {o['cost_per_query_usd']:8.4f}")


@app.command()
def demo(profile: str = "small", seed: int = 42) -> None:
    """Generate a corpus and produce a scored run in one go (fake unless an AI provider is configured)."""
    generate(profile, seed)
    ingest_vector(corpus_id(profile, seed))
    run(corpus=corpus_id(profile, seed), split="test", pipelines="A,B,C,B_oracle", repeats=1, limit=None,
        score=True)


@app.command()
def dashboard(port: int = 8501) -> None:
    """Open the results dashboard (binds to localhost only)."""
    app_path = Path(__file__).parent / "dashboard" / "app.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path), "--server.address", "127.0.0.1",
                    "--server.port", str(port), "--browser.gatherUsageStats", "false"], check=False)


if __name__ == "__main__":
    app()
