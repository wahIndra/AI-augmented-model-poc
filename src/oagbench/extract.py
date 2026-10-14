"""Ontology extraction line: LLM triple extraction per document, validation, entity resolution, graph build."""

import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

from oagbench.generate import Document
from oagbench.graph import Entity, Graph, Relation, normalize
from oagbench.llm import LLMError, LLMProvider
from oagbench.ontology import Ontology

EXTRACTION_RULES = """You extract a knowledge graph from one internal document of the company Kalimaya Digital.

Use only these classes, properties and relationships:
{schema}

Rules:
- Extract only facts the document states as current. Ignore rejected or hypothetical proposals.
- Entity `name`: the canonical name as written. Services: display name (for example "Payments API").
  Datastores: their identifier (for example "pg-core-01"). Teams: the team name without the word "team".
  People: full name. Regions: city name. Data classes: PII, PCI, Financial, Internal or Public.
  Incidents: their identifier (for example "INC-2025-001").
- `aliases`: every other way this document refers to the entity (backticked slugs, descriptive names).
- `properties`: only properties listed for the class, as key/value strings.
- Every relationship `source` and `target` must be the `name` of an entity you also list.
- `valid_from`: the effective date the document gives for that fact (YYYY-MM-DD), otherwise "".
- Relationship direction must follow the schema exactly."""


@dataclass
class ExtractionReport:
    documents: int = 0
    failed_documents: list[str] = field(default_factory=list)
    entities: int = 0
    relations: int = 0
    dropped_relations: Counter = field(default_factory=Counter)
    merges: list[str] = field(default_factory=list)
    llm_calls: list[dict] = field(default_factory=list)
    seconds: float = 0.0
    quality: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        return {
            "documents": self.documents,
            "failed_documents": self.failed_documents,
            "entities": self.entities,
            "relations": self.relations,
            "dropped_relations": dict(self.dropped_relations),
            "merges": len(self.merges),
            "llm_calls": len(self.llm_calls),
            "cost_usd": round(sum(c["cost_usd"] for c in self.llm_calls), 4),
            "seconds": round(self.seconds, 1),
            "quality_vs_canonical": self.quality,
        }


def extract_graph(docs: list[Document], provider: LLMProvider, ontology: Ontology,
                  concurrency: int = 4) -> tuple[Graph, ExtractionReport]:
    report = ExtractionReport(documents=len(docs))
    system = EXTRACTION_RULES.format(schema=ontology.schema_text())
    schema = ontology.extraction_schema()
    start = time.perf_counter()

    def run(doc: Document) -> tuple[Document, dict | None, dict | None, str]:
        prompt = f"Document id: {doc.doc_id}\n\n{doc.text}"
        for _ in range(2):
            try:
                result = provider.complete(system=system, prompt=prompt, schema=schema, max_tokens=16000)
                return doc, result.data, result.trace("extract"), ""
            except LLMError as exc:
                error = str(exc)
        return doc, None, None, error

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        outputs = list(pool.map(run, docs))

    raw = []
    for doc, data, trace, error in outputs:
        if trace:
            report.llm_calls.append(trace)
        if data is None:
            report.failed_documents.append(f"{doc.doc_id}: {error}")
        else:
            raw.append((doc.doc_id, data))
    graph = resolve(raw, ontology, report)
    report.entities, report.relations = len(graph.entities), len(graph.relations)
    report.seconds = time.perf_counter() - start
    return graph, report


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def resolve(raw: list[tuple[str, dict]], ontology: Ontology, report: ExtractionReport) -> Graph:
    """Merge per-document extractions: same label and a shared normalised name or alias means same entity."""
    parent: dict[tuple[str, str], tuple[str, str]] = {}

    def find(x: tuple[str, str]) -> tuple[str, str]:
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[str, str], b: tuple[str, str]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra
            report.merges.append(f"{rb[1]} -> {ra[1]} ({ra[0]})")

    mentions: list[tuple[str, dict]] = []
    for doc_id, data in raw:
        for ent in data.get("entities", []):
            if ent.get("label") not in ontology.classes or not str(ent.get("name", "")).strip():
                continue
            label = ent["label"]
            keys = [normalize(n) for n in [ent["name"], *ent.get("aliases", [])] if normalize(n)]
            for k in keys[1:]:
                union((label, keys[0]), (label, k))
            find((label, keys[0]))
            mentions.append((doc_id, ent))

    # Build one entity per cluster; the most frequent surface name becomes the display name.
    names: dict[tuple[str, str], Counter] = defaultdict(Counter)
    aliases: dict[tuple[str, str], set[str]] = defaultdict(set)
    props: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    docs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for doc_id, ent in mentions:
        root = find((ent["label"], normalize(ent["name"])))
        names[root][ent["name"].strip()] += 1
        aliases[root].update(a.strip() for a in ent.get("aliases", []) if a.strip())
        allowed = ontology.classes[ent["label"]]["properties"]
        for prop in ent.get("properties", []):
            if prop.get("key") in allowed and prop.get("value"):
                props[root].setdefault(prop["key"], str(prop["value"]))
        docs[root].add(doc_id)

    graph = Graph()
    ids: dict[tuple[str, str], str] = {}
    for root, counter in names.items():
        display = counter.most_common(1)[0][0]
        entity_id = f"{root[0].lower()}:{_slug(display)}"
        ids[root] = entity_id
        graph.add_entity(Entity(entity_id, root[0], display, props[root],
                                sorted(aliases[root] - {display}), sorted(docs[root])))

    for doc_id, data in raw:
        local = {normalize(e["name"]): e["label"] for e in data.get("entities", [])
                 if e.get("label") in ontology.classes and e.get("name")}
        for rel in data.get("relations", []):
            spec = ontology.relations.get(rel.get("type", ""))
            if spec is None:
                report.dropped_relations["unknown type"] += 1
                continue
            src_key, dst_key = normalize(rel.get("source", "")), normalize(rel.get("target", ""))
            if local.get(src_key) != spec.source or local.get(dst_key) != spec.target:
                report.dropped_relations["domain/range"] += 1
                continue
            src, dst = ids[find((spec.source, src_key))], ids[find((spec.target, dst_key))]
            graph.add_relation(Relation(spec.name, src, dst, doc_id, rel.get("valid_from", "") or ""))
    return graph


def compare_with_canonical(extracted: Graph, canonical: Graph) -> dict[str, float]:
    """Triple precision/recall after mapping extracted entities onto canonical ones by name and alias."""
    index = canonical.alias_index()

    def canonical_id(entity_id: str) -> str | None:
        entity = extracted.entities[entity_id]
        for name in entity.names():
            hit = index.get(normalize(name))
            if hit and canonical.entities[hit].label == entity.label:
                return hit
        return None

    gold = {(r.type, r.src, r.dst) for r in canonical.relations}
    mapped = set()
    for r in extracted.relations:
        src, dst = canonical_id(r.src), canonical_id(r.dst)
        mapped.add((r.type, src or f"?{r.src}", dst or f"?{r.dst}"))
    hits = len(gold & mapped)
    precision = hits / len(mapped) if mapped else 0.0
    recall = hits / len(gold) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"triple_precision": round(precision, 4), "triple_recall": round(recall, 4), "triple_f1": round(f1, 4)}
