import pytest

from oagbench.extract import ExtractionReport, compare_with_canonical, resolve
from oagbench.graph import normalize
from oagbench.ontology import load_ontology
from oagbench.retrieval.context import ContextItem, fit_budget
from oagbench.retrieval.graph import expand_neighbourhood
from oagbench.retrieval.graph_store import CypherRejected, check_read_only
from oagbench.retrieval.vector import HashEmbedder, MemoryVectorStore, VectorRetriever, chunk_documents


@pytest.mark.parametrize("query", [
    "MATCH (n) DETACH DELETE n",
    "MATCH (n) SET n.x = 1 RETURN n",
    "CREATE (n:Team {name: 'x'})",
    "MATCH (n) RETURN n; MATCH (m) DELETE m",
    "CALL dbms.security.listUsers()",
    "LOAD CSV FROM 'file:///x' AS row RETURN row",
])
def test_write_cypher_is_rejected(query: str) -> None:
    with pytest.raises(CypherRejected):
        check_read_only(query)


def test_read_cypher_passes_even_with_keywords_in_strings() -> None:
    q = "MATCH (s:Service) WHERE s.key = 'set create delete' RETURN s.name;"
    assert check_read_only(q) == q.rstrip(";")


def test_chunks_cover_every_document(corpus) -> None:
    chunks = chunk_documents(corpus.documents)
    assert {c.doc_id for c in chunks} == {d.doc_id for d in corpus.documents}


def test_vector_retrieval_finds_the_service_page(corpus) -> None:
    chunks = chunk_documents(corpus.documents)
    embedder = HashEmbedder()
    retriever = VectorRetriever(MemoryVectorStore(chunks, embedder.embed([c.text for c in chunks])), embedder)
    svc = corpus.canonical.by_label("Service")[0]
    ctx = retriever.retrieve(f"Which team owns {svc.name}?", budget=3000)
    assert svc.id in [i.source_doc for i in ctx.items[:5]]
    assert ctx.tokens <= 3000


def test_neighbourhood_expansion_contains_owner(corpus) -> None:
    g = corpus.canonical
    svc = g.by_label("Service")[0]
    owner = g.name_of(g.one_in(svc.id, "OWNS"))
    facts = expand_neighbourhood(g, f"Which team owns {svc.name}?")
    assert any("-[OWNS]->" in f.content and owner in f.content for f in facts)


def test_fit_budget_keeps_order_and_limit() -> None:
    items = [ContextItem("chunk", str(i), "x" * 400) for i in range(10)]  # 100 tokens each
    kept = fit_budget(items, 350)
    assert [i.source_doc for i in kept] == ["0", "1", "2"]


def test_entity_resolution_merges_aliases_and_validates_relations() -> None:
    ontology = load_ontology()
    raw = [
        ("svc-payments-api", {"entities": [
            {"label": "Service", "name": "Payments API", "aliases": ["payments-api"], "properties": [
                {"key": "tier", "value": "T1"}, {"key": "colour", "value": "blue"}]},
            {"label": "Team", "name": "Payments Core", "aliases": [], "properties": []},
        ], "relations": [
            {"type": "OWNS", "source": "Payments Core", "target": "Payments API", "valid_from": ""},
            {"type": "OWNS", "source": "Payments API", "target": "Payments Core", "valid_from": ""},
        ]}),
        ("inc-1", {"entities": [{"label": "Service", "name": "payments-api", "aliases": [], "properties": []}],
                   "relations": []}),
    ]
    report = ExtractionReport()
    g = resolve(raw, ontology, report)
    services = g.by_label("Service")
    assert len(services) == 1
    assert services[0].props == {"tier": "T1"}  # unknown property dropped
    assert set(services[0].source_docs) == {"svc-payments-api", "inc-1"}
    assert len(g.relations) == 1 and report.dropped_relations["domain/range"] == 1


def test_canonical_graph_compares_perfectly_with_itself(corpus) -> None:
    assert compare_with_canonical(corpus.canonical, corpus.canonical)["triple_f1"] == 1.0


def test_normalize() -> None:
    assert normalize("PG-Core_03") == normalize("pg core 03") == "pg core 03"
