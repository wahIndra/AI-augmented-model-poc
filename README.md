# OAG vs RAG PoC

Benchmark of three retrieval architectures over one seeded, synthetic enterprise corpus (the fictional fintech
Kalimaya Digital):

- **A. Vector RAG**: chunks and embeddings, top-k similarity
- **B. OAG**: ontology-constrained knowledge graph extracted by an LLM, queried with text-to-Cypher
- **C. Hybrid**: A and B in parallel, fused context
- **B_oracle**: B on the ground-truth graph, to separate extraction errors from query errors

Design docs: [docs/index.md](docs/index.md) (also published in docuCenter).

## Quick start (no AI provider, no Docker)

```bash
uv sync --extra anthropic --extra openai --extra services --extra dashboard
cp .env.example .env          # AI_PROVIDER=none by default
uv run oagbench demo          # corpus + vector index + FAKE scored run
uv run oagbench dashboard     # http://127.0.0.1:8501
uv run pytest
```

With `AI_PROVIDER=none`, `oagbench run` writes **fake results**: answers simulated from the gold set, labelled
`fake` in the run id, the records, the CLI and the dashboard. Use them to build the harness and dashboard, never as
findings. Development costs nothing in API spend.

## Modes

| Setting | Development | Benchmark run |
| --- | --- | --- |
| `AI_PROVIDER` | `none` (fake results) | `anthropic` or `openai_compatible` |
| `STORE_BACKEND` | `memory` (no Cypher; B uses neighbourhood expansion) | `services` (Qdrant + Neo4j via `docker compose`) |
| `EMBEDDER` | `hash` (no download) | `sentence_transformers` (`uv sync --extra local-embeddings`) |

## Commands

| Command | AI calls | Purpose |
| --- | --- | --- |
| `oagbench status` | no | Show mode and configuration (no secrets) |
| `oagbench generate --profile small\|full --seed 42` | no | Canonical graph, documents, gold questions |
| `oagbench ingest vector -c full-s42` | no | Chunk and embed (pipeline A) |
| `oagbench ingest graph -c full-s42` | yes | LLM extraction into the knowledge graph (pipeline B) |
| `oagbench ingest oracle -c full-s42` | no | Load the ground-truth graph into the oracle Neo4j |
| `oagbench run -c full-s42 --repeats 3` | yes, or fake | Run the benchmark; stops at `RUN_BUDGET_USD` |
| `oagbench score <run-id> [--judge]` | only with `--judge` | Deterministic scores, optional LLM judge |
| `oagbench dashboard` | no | Results browser |

## Benchmark run (later, with a provider)

```bash
# .env: AI_PROVIDER=anthropic, ANTHROPIC_API_KEY=..., STORE_BACKEND=services,
#       EMBEDDER=sentence_transformers, NEO4J_PASSWORD=...
docker compose up -d qdrant neo4j neo4j-oracle
uv run oagbench generate --profile full
uv run oagbench ingest vector -c full-s42
uv run oagbench ingest graph -c full-s42
uv run oagbench ingest oracle -c full-s42
uv run oagbench run -c full-s42 --repeats 3
uv run oagbench score <run-id> --judge
```

Estimated cost of a full run: see [docs/phase-1/cost.md](docs/phase-1/cost.md).
