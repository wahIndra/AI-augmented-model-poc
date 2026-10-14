"""Results dashboard: `oagbench dashboard` (Streamlit, localhost only). Read-only over the results directory."""

import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from oagbench.config import get_settings

PIPELINE_ORDER = ["A", "B", "C", "B_oracle"]
PIPELINE_LABELS = {"A": "A · Vector RAG", "B": "B · OAG", "C": "C · Hybrid", "B_oracle": "B · OAG (oracle graph)"}
# Validated categorical palette (fixed order, never cycled); contrast relief is the table view below the chart.
PIPELINE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
CATEGORY_ORDER = ["direct", "multihop_2", "multihop_3plus", "constraint", "summary", "unanswerable"]

st.set_page_config(page_title="OAG vs RAG benchmark", layout="wide")
settings = get_settings()
results_dir = Path(settings.results_dir)


@st.cache_data
def load_run(run_dir: str) -> tuple[dict, dict | None, pd.DataFrame | None, list[dict]]:
    path = Path(run_dir)
    record = json.loads((path / "run.json").read_text(encoding="utf-8"))
    summary_path = path / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else None
    scores = pd.read_parquet(path / "scores.parquet") if (path / "scores.parquet").exists() else None
    responses = [json.loads(line) for line in (path / "responses.jsonl").read_text(encoding="utf-8").splitlines()
                 if line]
    return record, summary, scores, responses


@st.cache_data
def load_gold(corpus_id: str) -> dict[str, dict]:
    path = Path(settings.data_dir) / corpus_id / "gold.jsonl"
    return {q["qid"]: q for q in map(json.loads, path.read_text(encoding="utf-8").splitlines()) if q}


runs = sorted((p for p in results_dir.glob("*") if (p / "run.json").exists()), reverse=True)
st.title("OAG vs RAG benchmark")
if not runs:
    st.info("No runs yet. Run `uv run oagbench demo` to generate a corpus and a (fake) scored run.")
    st.stop()

with st.sidebar:
    run_dir = st.selectbox("Run", runs, format_func=lambda p: p.name)
record, summary, scores, responses = load_run(str(run_dir))

if record["mode"] == "fake":
    st.warning("**FAKE RESULTS.** AI_PROVIDER=none, so every answer is simulated from the gold set with fixed "
               "per-pipeline profiles. Use this run to check the harness and the dashboard, not to compare "
               "architectures.", icon="⚠️")
st.caption(f"Run `{record['run_id']}` · corpus `{record['corpus_id']}` · split `{record['split']}` · "
           f"{record['questions']} questions × {len(record['pipelines'])} pipelines × {record['repeats']} repeats · "
           f"provider `{record['settings']['ai_provider']}` / model `{record['settings']['model']}`")
if summary is None or scores is None:
    st.info(f"This run is not scored yet: `uv run oagbench score {record['run_id']}`")
    st.stop()

pipelines = [p for p in PIPELINE_ORDER if p in record["pipelines"]]
cells = pd.DataFrame(summary["cells"])
cells["low"] = cells["correct_ci"].str[0]
cells["high"] = cells["correct_ci"].str[1]
cells["pipeline_label"] = cells["pipeline"].map(PIPELINE_LABELS)
ops = pd.DataFrame(summary["operations"]).set_index("pipeline")

# --- KPI row --------------------------------------------------------------------------------------------------
st.subheader("Overall accuracy")
overall = cells[cells["category"] == "all"].set_index("pipeline")
for col, pipeline in zip(st.columns(len(pipelines)), pipelines, strict=True):
    row = overall.loc[pipeline]
    col.metric(PIPELINE_LABELS[pipeline], f"{row['correct']:.0%}",
               help=f"95% CI {row['low']:.0%}–{row['high']:.0%}; share of questions answered exactly right")
    col.caption(f"95% CI {row['low']:.0%}–{row['high']:.0%} · p50 {ops.loc[pipeline, 'latency_p50_ms']:.0f} ms · "
                f"${ops.loc[pipeline, 'cost_per_query_usd']:.4f}/query")

# --- accuracy by category ---------------------------------------------------------------------------------------
st.subheader("Accuracy by question category")
by_cat = cells[cells["category"] != "all"]
color = alt.Color("pipeline_label:N", title="Pipeline",
                  scale=alt.Scale(domain=[PIPELINE_LABELS[p] for p in pipelines],
                                  range=PIPELINE_COLORS[: len(pipelines)]),
                  legend=alt.Legend(orient="top"))
x = alt.X("category:N", sort=CATEGORY_ORDER, title=None, axis=alt.Axis(labelAngle=0))
offset = alt.XOffset("pipeline_label:N", sort=[PIPELINE_LABELS[p] for p in pipelines])
tooltip = [alt.Tooltip("pipeline_label:N", title="Pipeline"), alt.Tooltip("category:N", title="Category"),
           alt.Tooltip("correct:Q", title="Accuracy", format=".0%"),
           alt.Tooltip("low:Q", title="95% CI low", format=".0%"),
           alt.Tooltip("high:Q", title="95% CI high", format=".0%"),
           alt.Tooltip("questions:Q", title="Questions")]
bars = alt.Chart(by_cat).mark_bar(cornerRadiusTopLeft=2, cornerRadiusTopRight=2).encode(
    x=x, xOffset=offset, color=color, tooltip=tooltip,
    y=alt.Y("correct:Q", title="Accuracy", scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format="%", grid=True)),
)
errors = alt.Chart(by_cat).mark_rule(strokeWidth=1.2, color="#52514e").encode(
    x=x, xOffset=offset, y="low:Q", y2="high:Q", tooltip=tooltip,
)
st.altair_chart((bars + errors).properties(height=340).configure_axis(gridOpacity=0.4, domain=False),
                use_container_width=True)

with st.expander("Table view"):
    table = by_cat.pivot_table(index="category", columns="pipeline_label", values="correct").reindex(CATEGORY_ORDER)
    st.dataframe(table.style.format("{:.0%}"), use_container_width=True)

# --- paired comparison and operations ---------------------------------------------------------------------------
left, right = st.columns(2)
with left:
    st.subheader("Difference vs A (paired)")
    diffs = pd.DataFrame(summary["paired_vs_baseline"])
    if not diffs.empty:
        diffs["95% CI"] = diffs["ci"].map(lambda c: f"{c[0]:+.0%} to {c[1]:+.0%}")
        diffs["significant"] = diffs["ci"].map(lambda c: "yes" if c[0] > 0 or c[1] < 0 else "no")
        view = diffs.assign(diff=diffs["diff_correct"].map("{:+.0%}".format))
        st.dataframe(view[["pipeline", "category", "diff", "95% CI", "significant"]], hide_index=True,
                     use_container_width=True)
with right:
    st.subheader("Operations")
    ops_view = ops.reindex(pipelines).rename(index=PIPELINE_LABELS)[
        ["latency_p50_ms", "latency_p95_ms", "cost_per_query_usd", "fallback_rate", "abstain_rate",
         "context_tokens_mean"]]
    st.dataframe(ops_view.style.format({
        "latency_p50_ms": "{:.0f}", "latency_p95_ms": "{:.0f}", "cost_per_query_usd": "${:.4f}",
        "fallback_rate": "{:.0%}", "abstain_rate": "{:.0%}", "context_tokens_mean": "{:.0f}"}),
        use_container_width=True)
    if record["mode"] == "fake":
        st.caption("Latency and cost are simulated (token counts priced as Claude Opus 5.5).")

# --- question explorer ------------------------------------------------------------------------------------------
st.subheader("Question explorer")
gold = load_gold(record["corpus_id"])
per_q = scores.groupby(["qid", "pipeline"])["correct"].mean().unstack()
c1, c2 = st.columns([1, 2])
with c1:
    category = st.selectbox("Category", ["all", *CATEGORY_ORDER])
    only_disagreements = st.toggle("Only questions where pipelines disagree", value=True)
qids = [q for q in per_q.index if category == "all" or gold[q]["category"] == category]
if only_disagreements:
    qids = [q for q in qids if per_q.loc[q].nunique() > 1]
with c2:
    if not qids:
        st.info("No questions match the filters.")
        st.stop()
    qid = st.selectbox("Question", qids, format_func=lambda q: f"{q} · {gold[q]['question']}")

q = gold[qid]
st.markdown(f"**{q['question']}**  \nCategory `{q['category']}` · hops {q['hops']} · "
            f"gold: **{'no answer' if q['answer_type'] == 'no_answer' else ', '.join(q['answer'])}** · "
            f"supporting docs: {', '.join(q['supporting_docs']) or '-'}")
repeat = st.radio("Repeat", sorted({r["repeat"] for r in responses}), horizontal=True)
by_pipeline = {r["pipeline"]: r for r in responses if r["qid"] == qid and r["repeat"] == repeat}
score_rows = scores[(scores["qid"] == qid) & (scores["repeat"] == repeat)].set_index("pipeline")
for col, pipeline in zip(st.columns(len(pipelines)), pipelines, strict=True):
    r = by_pipeline.get(pipeline)
    with col:
        st.markdown(f"**{PIPELINE_LABELS[pipeline]}**")
        if r is None:
            st.caption("No response (stopped by budget?)")
            continue
        s = score_rows.loc[pipeline]
        verdict = "✅ correct" if s["correct"] == 1 else f"❌ wrong (F1 {s['f1']:.2f})"
        st.markdown(f"{verdict}  \n{r['answer']}")
        st.caption(f"Cited: {', '.join(r['citations']) or '-'} · {r['latency_ms']['total']} ms")
        if r["context"].get("cypher"):
            st.code(r["context"]["cypher"], language="cypher")
        if r["context"].get("fallback_used"):
            st.caption(f"Fell back to neighbourhood expansion. {r['context'].get('error', '')}")
        with st.expander(f"Context ({len(r['context']['items'])} items)"):
            for item in r["context"]["items"]:
                st.markdown(f"`{item['kind']}` · doc `{item['source_doc']}`")
                st.text(item["content"][:600])
