"""Graph stores: in-memory (dev) and Neo4j (benchmark). Both expose an in-memory snapshot for anchoring and
neighbourhood expansion; only Neo4j runs Cypher."""

import re
from collections import defaultdict
from typing import Any, Protocol

from oagbench.graph import Graph, normalize
from oagbench.ontology import load_ontology

WRITE_CLAUSES = re.compile(
    r"\b(CREATE|MERGE|DELETE|DETACH|SET|REMOVE|DROP|LOAD\s+CSV|FOREACH|CALL\s+\{|CALL\s+(dbms|db\.create|apoc))\b",
    re.IGNORECASE,
)


class CypherRejected(ValueError):
    pass


def check_read_only(cypher: str) -> str:
    """Reject anything that could write. The read transaction and read-only user are further layers."""
    stripped = re.sub(r"//[^\n]*|'[^']*'|\"[^\"]*\"", "''", cypher)  # ignore comments and string literals
    if WRITE_CLAUSES.search(stripped):
        raise CypherRejected("query contains a write or procedure clause")
    if ";" in stripped.strip().rstrip(";"):
        raise CypherRejected("multiple statements are not allowed")
    return cypher.strip().rstrip(";")


class GraphStore(Protocol):
    supports_cypher: bool

    def snapshot(self) -> Graph: ...

    def run_cypher(self, cypher: str) -> list[dict[str, Any]]: ...


class MemoryGraphStore:
    supports_cypher = False

    def __init__(self, graph: Graph):
        self._graph = graph

    def snapshot(self) -> Graph:
        return self._graph

    def run_cypher(self, cypher: str) -> list[dict[str, Any]]:
        raise NotImplementedError("the memory backend has no Cypher engine; use STORE_BACKEND=services")


class Neo4jGraphStore:
    supports_cypher = True

    def __init__(self, uri: str, user: str, password: str, *, query_user: str = "", query_password: str = "",
                 timeout_s: float = 5.0, row_limit: int = 200):
        from neo4j import GraphDatabase

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        self._query_driver = (GraphDatabase.driver(uri, auth=(query_user, query_password))
                              if query_user else self._driver)
        self.timeout_s = timeout_s
        self.row_limit = row_limit
        self._snapshot: Graph | None = None

    def close(self) -> None:
        self._driver.close()
        if self._query_driver is not self._driver:
            self._query_driver.close()

    def load(self, graph: Graph) -> None:
        """Replace the database contents with `graph`."""
        ontology = load_ontology()
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n").consume()
            for label in ontology.classes:
                session.run(f"CREATE CONSTRAINT {label.lower()}_id IF NOT EXISTS "
                            f"FOR (n:{label}) REQUIRE n.id IS UNIQUE").consume()
            by_label: dict[str, list[dict]] = defaultdict(list)
            for e in graph.entities.values():
                keys = sorted({normalize(n) for n in e.names()})
                by_label[e.label].append({"id": e.id, "props": {
                    **{k: str(v) for k, v in e.props.items()}, "name": e.name, "key": normalize(e.name),
                    "keys": keys, "aliases": e.aliases, "source_docs": e.source_docs}})
            for label, rows in by_label.items():
                if label not in ontology.classes:
                    continue  # labels are interpolated, so only ontology labels are allowed
                session.run(f"UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n += row.props",
                            rows=rows).consume()
            by_type: dict[str, list[dict]] = defaultdict(list)
            for r in graph.relations:
                by_type[r.type].append({"src": r.src, "dst": r.dst, "doc": r.source_doc, "from": r.valid_from})
            for rel_type, rows in by_type.items():
                spec = ontology.relations.get(rel_type)
                if spec is None:
                    continue
                session.run(
                    f"UNWIND $rows AS row MATCH (a:{spec.source} {{id: row.src}}) "
                    f"MATCH (b:{spec.target} {{id: row.dst}}) "
                    f"MERGE (a)-[r:{rel_type} {{source_doc: row.doc}}]->(b) SET r.valid_from = row.from",
                    rows=rows,
                ).consume()
        self._snapshot = graph

    def snapshot(self) -> Graph:
        if self._snapshot is None:
            self._snapshot = self._read_graph()
        return self._snapshot

    def _read_graph(self) -> Graph:
        from oagbench.graph import Entity, Relation

        g = Graph()
        with self._driver.session() as session:
            for rec in session.run("MATCH (n) RETURN n, labels(n)[0] AS label"):
                props = dict(rec["n"])
                g.add_entity(Entity(
                    props.pop("id"), rec["label"], props.pop("name", ""),
                    aliases=list(props.pop("aliases", [])), source_docs=list(props.pop("source_docs", [])),
                    props={k: v for k, v in props.items() if k not in {"key", "keys"}},
                ))
            for rec in session.run("MATCH (a)-[r]->(b) RETURN a.id AS src, type(r) AS t, b.id AS dst, "
                                   "r.source_doc AS doc, r.valid_from AS vf"):
                g.add_relation(Relation(rec["t"], rec["src"], rec["dst"], rec["doc"] or "", rec["vf"] or ""))
        return g

    def run_cypher(self, cypher: str) -> list[dict[str, Any]]:
        from neo4j import Query

        query = Query(check_read_only(cypher), timeout=self.timeout_s)
        with self._query_driver.session() as session:
            # execute_read opens a READ transaction; the server refuses writes inside it.
            return session.execute_read(lambda tx: [r.data() for r in tx.run(query)][: self.row_limit])
