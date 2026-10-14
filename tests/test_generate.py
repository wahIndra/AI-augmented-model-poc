from collections import Counter
from pathlib import Path

from oagbench.generate import generate_corpus, load_corpus
from oagbench.generate.gold import QUOTAS, policy_violations


def test_generation_is_reproducible(tmp_path: Path) -> None:
    a = load_corpus(generate_corpus(tmp_path / "a", "small", 3))
    b = load_corpus(generate_corpus(tmp_path / "b", "small", 3))
    assert a.manifest["corpus_hash"] == b.manifest["corpus_hash"]
    assert [q.to_dict() for q in a.gold] == [q.to_dict() for q in b.gold]


def test_different_seeds_differ(tmp_path: Path) -> None:
    a = load_corpus(generate_corpus(tmp_path, "small", 1))
    b = load_corpus(generate_corpus(tmp_path, "small", 2))
    assert a.manifest["corpus_hash"] != b.manifest["corpus_hash"]


def test_quotas_and_split(corpus) -> None:
    counts = Counter(q.category for q in corpus.gold)
    assert counts == Counter(QUOTAS["small"])
    assert {q.split for q in corpus.gold} == {"dev", "test"}
    assert len({q.question for q in corpus.gold}) == len(corpus.gold)


def test_supporting_docs_exist(corpus) -> None:
    doc_ids = {d.doc_id for d in corpus.documents}
    for q in corpus.gold:
        assert set(q.supporting_docs) <= doc_ids, q.qid


def test_answers_are_stated_in_supporting_docs(corpus) -> None:
    """Every entity answer to a direct question appears in the documents that support it."""
    for q in corpus.gold:
        if q.category != "direct" or q.answer_type != "entity":
            continue
        text = " ".join(corpus.doc(d).text for d in q.supporting_docs)
        assert q.answer[0] in text, (q.qid, q.question, q.answer)


def test_superseded_region_uses_latest_fact(corpus) -> None:
    g = corpus.canonical
    migrated = [s.id for s in g.by_label("Service") if len(g.out(s.id, "DEPLOYED_IN")) > 1]
    assert migrated
    for svc in migrated:
        latest = max(g.out_relations(svc, "DEPLOYED_IN"), key=lambda r: r.valid_from)
        assert latest.source_doc.startswith("migration-")
        assert g.one_out(svc, "DEPLOYED_IN") == latest.dst


def test_dependencies_are_acyclic(corpus) -> None:
    g = corpus.canonical
    for s in g.by_label("Service"):
        assert s.id not in g.transitive_in(s.id, "DEPENDS_ON")


def test_policy_violations_are_consistent(corpus) -> None:
    g = corpus.canonical
    violators, _ = policy_violations(g, "pol-res-01")
    allowed = set(g.out("pol-res-01", "ALLOWS_REGION"))
    for svc in violators:
        regions = {g.one_out(svc, "DEPLOYED_IN")} | {g.one_out(d, "HOSTED_IN") for d in g.out(svc, "USES_DATASTORE")}
        assert regions - allowed
