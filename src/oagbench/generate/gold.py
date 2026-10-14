"""Gold question set computed from the canonical graph.

Each template yields candidate questions with an exact answer, the documents that support it, and a reference
Cypher query (used as few-shot examples for text-to-Cypher, from the dev split only).
"""

import random
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass, field

from oagbench.generate.canonical import SERVICE_DOMAINS, SERVICE_PARTS
from oagbench.graph import Graph, normalize

CATEGORIES = ["direct", "multihop_2", "multihop_3plus", "constraint", "summary", "unanswerable"]
QUOTAS = {
    "small": {"direct": 10, "multihop_2": 8, "multihop_3plus": 8, "constraint": 8, "summary": 4, "unanswerable": 4},
    "full": {"direct": 25, "multihop_2": 20, "multihop_3plus": 20, "constraint": 20, "summary": 10,
             "unanswerable": 10},
}
DEV_SHARE = 0.3
TIER_RANK = {"T1": 1, "T2": 2, "T3": 3}


@dataclass
class GoldQuestion:
    qid: str
    category: str
    template: str
    hops: int
    question: str
    answer_type: str  # entity | entities | value | number | summary | no_answer
    answer: list[str]  # display values; key points for summaries
    answer_ids: list[str]  # entity ids behind `answer` (empty for values and numbers)
    supporting_docs: list[str]
    cypher: str = ""
    split: str = "test"
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


Template = Callable[[Graph, random.Random], Iterator[GoldQuestion]]


def _q(category: str, template: str, hops: int, question: str, answer_type: str, ids: list[str], g: Graph,
       docs: list[str], cypher: str = "", values: list[str] | None = None) -> GoldQuestion:
    answer = values if values is not None else [g.name_of(i) for i in ids]
    return GoldQuestion("", category, template, hops, question, answer_type, answer, ids,
                        sorted(set(docs)), cypher.strip())


def _region_docs(g: Graph, svc: str) -> list[str]:
    return [r.source_doc for r in g.out_relations(svc, "DEPLOYED_IN")]


def _active_services(g: Graph) -> list[str]:
    return [s.id for s in g.by_label("Service") if s.props["lifecycle"] == "active"]


# --- direct ------------------------------------------------------------------------------------------------

def t_owner(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        team = g.one_in(s.id, "OWNS")
        yield _q("direct", "owner", 1, f"Which team owns {s.name}?", "entity", [team], g, [s.id],
                 f"MATCH (t:Team)-[:OWNS]->(s:Service {{key: '{normalize(s.name)}'}}) RETURN t.name AS team, s.source_docs AS docs")


def t_engine(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for d in g.by_label("Datastore"):
        name = rng.choice([d.id, f"the {d.aliases[0]}"])
        yield _q("direct", "engine", 1, f"Which database engine does {name} run?", "value", [], g, [d.id],
                 f"MATCH (d:Datastore {{key: '{normalize(d.name)}'}}) RETURN d.engine AS engine, d.source_docs AS docs",
                 values=[d.props["engine"]])


def t_region(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        region = g.one_out(s.id, "DEPLOYED_IN")
        q = _q("direct", "region", 1, f"In which region is {s.name} currently deployed?", "entity", [region], g,
               _region_docs(g, s.id),
               f"MATCH (s:Service {{key: '{normalize(s.name)}'}})-[r:DEPLOYED_IN]->(reg:Region) "
               "RETURN reg.name AS region, r.valid_from AS since, r.source_doc AS doc "
               "ORDER BY r.valid_from DESC LIMIT 1")
        if len(g.out(s.id, "DEPLOYED_IN")) > 1:
            q.notes.append("superseded-fact")
        yield q


def t_lead(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for t in g.by_label("Team"):
        lead = g.one_in(t.id, "LEADS")
        yield _q("direct", "lead", 1, f"Who leads the {t.name} team?", "entity", [lead], g, [t.id],
                 f"MATCH (p:Person)-[:LEADS]->(t:Team {{key: '{normalize(t.name)}'}}) RETURN p.name AS lead")


# --- two hops ----------------------------------------------------------------------------------------------

def t_lead_of_owner(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        team = g.one_in(s.id, "OWNS")
        lead = g.one_in(team, "LEADS")
        yield _q("multihop_2", "lead_of_owner", 2, f"Who leads the team that owns {s.name}?", "entity", [lead],
                 g, [s.id, team],
                 f"MATCH (p:Person)-[:LEADS]->(t:Team)-[:OWNS]->(s:Service {{key: '{normalize(s.name)}'}}) "
                 "RETURN p.name AS lead, t.name AS team")


def t_datastore_regions(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        stores = g.out(s.id, "USES_DATASTORE")
        regions = sorted({g.one_out(d, "HOSTED_IN") for d in stores})
        yield _q("multihop_2", "datastore_regions", 2,
                 f"In which regions are the datastores used by {s.name} hosted?", "entities", regions, g,
                 [s.id, *stores],
                 f"MATCH (s:Service {{key: '{normalize(s.name)}'}})-[:USES_DATASTORE]->(d:Datastore)-[:HOSTED_IN]->(r:Region) "
                 "RETURN DISTINCT r.name AS region, collect(d.id) AS datastores")


def t_division_of_owner(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        team = g.one_in(s.id, "OWNS")
        ou = g.one_in(team, "CONTAINS")
        yield _q("multihop_2", "division_of_owner", 2,
                 f"Which division does the team that owns {s.name} belong to?", "entity", [ou], g,
                 [s.id, "org-overview"],
                 f"MATCH (o:OrgUnit)-[:CONTAINS]->(:Team)-[:OWNS]->(s:Service {{key: '{normalize(s.name)}'}}) RETURN o.name AS division")


def t_dependency_owners(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        deps = g.out(s.id, "DEPENDS_ON")
        if not deps:
            continue
        teams = sorted({g.one_in(d, "OWNS") for d in deps})
        yield _q("multihop_2", "dependency_owners", 2,
                 f"Which teams own the services that {s.name} calls directly?", "entities", teams, g,
                 [s.id, *deps],
                 f"MATCH (s:Service {{key: '{normalize(s.name)}'}})-[:DEPENDS_ON]->(d:Service)<-[:OWNS]-(t:Team) "
                 "RETURN DISTINCT t.name AS team, collect(d.name) AS services")


# --- three or more hops ------------------------------------------------------------------------------------

def t_datastore_impact(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for d in g.by_label("Datastore"):
        users = set(g.inn(d.id, "USES_DATASTORE"))
        impacted = set(users)
        for u in users:
            impacted |= g.transitive_in(u, "DEPENDS_ON")
        if impacted == users:
            continue  # no dependency hop involved, so not a 3+ hop question
        teams = sorted({g.one_in(s, "OWNS") for s in impacted})
        yield _q("multihop_3plus", "datastore_impact", 3,
                 f"If {d.id} becomes unavailable, which teams own services that are affected, either because "
                 "they use it directly or because they depend, directly or indirectly, on a service that does?",
                 "entities", teams, g, sorted(impacted),
                 f"MATCH (d:Datastore {{key: '{normalize(d.name)}'}})<-[:USES_DATASTORE]-(:Service)<-[:DEPENDS_ON*0..]-(s:Service) "
                 "MATCH (t:Team)-[:OWNS]->(s) RETURN DISTINCT t.name AS team, collect(DISTINCT s.name) AS services")


def t_incident_division(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for i in g.by_label("Incident"):
        root = g.one_out(i.id, "ROOT_CAUSE")
        team = g.one_in(root, "OWNS")
        ou = g.one_in(team, "CONTAINS")
        yield _q("multihop_3plus", "incident_division", 3,
                 f"Which division owns the service that caused incident {i.name}?", "entity", [ou], g,
                 [i.id, root, "org-overview"],
                 f"MATCH (:Incident {{key: '{normalize(i.name)}'}})-[:ROOT_CAUSE]->(s:Service)<-[:OWNS]-(t:Team)<-[:CONTAINS]-(o:OrgUnit) "
                 "RETURN o.name AS division, t.name AS team, s.name AS service")


def t_incident_leads(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for i in g.by_label("Incident"):
        affected = g.out(i.id, "AFFECTED")
        teams = {g.one_in(s, "OWNS") for s in affected}
        leads = sorted({g.one_in(t, "LEADS") for t in teams})
        yield _q("multihop_3plus", "incident_leads", 3,
                 f"Who leads the teams that own the services affected by incident {i.name}?", "entities", leads, g,
                 [i.id, *affected, *teams],
                 f"MATCH (:Incident {{key: '{normalize(i.name)}'}})-[:AFFECTED]->(s:Service)<-[:OWNS]-(t:Team)<-[:LEADS]-(p:Person) "
                 "RETURN DISTINCT p.name AS lead, t.name AS team")


# --- constraints and aggregation -----------------------------------------------------------------------------

def policy_violations(g: Graph, policy_id: str) -> tuple[list[str], list[str]]:
    """Active services violating a policy, and the documents needed to show it."""
    policy = g.entities[policy_id]
    governed = set(g.out(policy_id, "GOVERNS"))
    allowed = set(g.out(policy_id, "ALLOWS_REGION"))
    min_tier = policy.props.get("min_tier")
    violators, docs = [], [policy_id]
    for s in _active_services(g):
        if not governed & set(g.out(s, "PROCESSES")):
            continue
        stores = g.out(s, "USES_DATASTORE")
        bad_region = g.one_out(s, "DEPLOYED_IN") not in allowed
        bad_store = any(g.one_out(d, "HOSTED_IN") not in allowed for d in stores)
        bad_tier = bool(min_tier) and TIER_RANK[g.entities[s].props["tier"]] > TIER_RANK[min_tier]
        if bad_region or bad_store or bad_tier:
            violators.append(s)
            docs += [*_region_docs(g, s), *stores]
    return sorted(violators), docs


def t_policy_violations(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for p in g.by_label("Policy"):
        violators, docs = policy_violations(g, p.id)
        if violators:
            yield _q("constraint", "policy_violations", 3, f"Which active services violate the {p.name} policy?",
                     "entities", violators, g, docs,
                     f"MATCH (p:Policy {{key: '{normalize(p.name)}'}})-[:GOVERNS]->(:DataClass)<-[:PROCESSES]-(s:Service {{lifecycle: 'active'}}) "
                     "MATCH (s)-[r:DEPLOYED_IN]->(reg:Region) WITH p, s, r, reg ORDER BY r.valid_from DESC "
                     "WITH p, s, head(collect(reg)) AS region "
                     "OPTIONAL MATCH (s)-[:USES_DATASTORE]->(:Datastore)-[:HOSTED_IN]->(dreg:Region) "
                     "WITH p, s, region, collect(DISTINCT dreg) AS dregions "
                     "WHERE NOT (p)-[:ALLOWS_REGION]->(region) "
                     "OR any(x IN dregions WHERE NOT (p)-[:ALLOWS_REGION]->(x)) "
                     "OR (p.min_tier IS NOT NULL AND s.tier > p.min_tier) "
                     "RETURN DISTINCT s.name AS service")


def t_tier1_count(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for t in g.by_label("Team"):
        owned = g.out(t.id, "OWNS")
        count = sum(1 for s in owned if g.entities[s].props["tier"] == "T1")
        yield _q("constraint", "tier1_count", 2, f"How many tier T1 services does the {t.name} team own?", "number",
                 [], g, owned,
                 f"MATCH (:Team {{key: '{normalize(t.name)}'}})-[:OWNS]->(s:Service {{tier: 'T1'}}) RETURN count(s) AS n",
                 values=[str(count)])


def t_classified_services(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for t in g.by_label("Team"):
        for dc in ("dc-pii", "dc-pci", "dc-financial"):
            hits = sorted(s for s in g.out(t.id, "OWNS") if dc in g.out(s, "PROCESSES"))
            if hits:
                yield _q("constraint", "classified_services", 2,
                         f"Which services owned by the {t.name} team process {g.name_of(dc)} data?", "entities", hits,
                         g, g.out(t.id, "OWNS"),
                         f"MATCH (:Team {{key: '{normalize(t.name)}'}})-[:OWNS]->(s:Service)-[:PROCESSES]->(:DataClass {{key: '{normalize(g.name_of(dc))}'}}) "
                         "RETURN s.name AS service")


def t_vendor_region(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for v in g.by_label("Vendor"):
        for r in g.by_label("Region"):
            hits = sorted(s for s in g.inn(v.id, "USES_VENDOR") if g.one_out(s, "DEPLOYED_IN") == r.id)
            if hits:
                yield _q("constraint", "vendor_region", 2,
                         f"Which services currently deployed in {r.name} use {v.name}?", "entities", hits, g,
                         [d for s in hits for d in _region_docs(g, s)],
                         f"MATCH (s:Service)-[:USES_VENDOR]->(:Vendor {{key: '{normalize(v.name)}'}}) "
                         "MATCH (s)-[rel:DEPLOYED_IN]->(reg:Region) WITH s, rel, reg ORDER BY rel.valid_from DESC "
                         f"WITH s, head(collect(reg)) AS region WHERE region.key = '{normalize(r.name)}' RETURN s.name AS service")


def t_depends_on_retiring(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    retiring = {s.id for s in g.by_label("Service") if s.props["lifecycle"] != "active"}
    hits = sorted(s for s in _active_services(g) if retiring & set(g.out(s, "DEPENDS_ON")))
    if hits:
        yield _q("constraint", "depends_on_retiring", 2,
                 "Which active services depend directly on a service that is deprecated or retired?", "entities",
                 hits, g, [*hits, *(retiring & {d for s in hits for d in g.out(s, "DEPENDS_ON")})],
                 "MATCH (s:Service {lifecycle: 'active'})-[:DEPENDS_ON]->(d:Service) "
                 "WHERE d.lifecycle IN ['deprecated', 'retired'] RETURN DISTINCT s.name AS service")


# --- summaries -----------------------------------------------------------------------------------------------

def t_team_overview(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for t in g.by_label("Team"):
        lead = g.one_in(t.id, "LEADS")
        owned = sorted(g.out(t.id, "OWNS"))
        yield _q("summary", "team_overview", 1,
                 f"Give a short overview of the {t.name} team: who leads it and which services it owns.", "summary",
                 [lead, *owned], g, [t.id, *owned])


def t_incident_history(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    for s in g.by_label("Service"):
        incidents = sorted(g.inn(s.id, "AFFECTED"))
        if len(incidents) >= 2:
            yield _q("summary", "incident_history", 1, f"Summarise the incidents that affected {s.name}.", "summary",
                     incidents, g, incidents)


# --- unanswerable --------------------------------------------------------------------------------------------

def t_unanswerable(g: Graph, rng: random.Random) -> Iterator[GoldQuestion]:
    existing = {e.aliases[0] for e in g.by_label("Service")}
    unused = [f"{d}-{p}" for d in SERVICE_DOMAINS for p in SERVICE_PARTS if f"{d}-{p}" not in existing]
    rng.shuffle(unused)
    for slug in unused[:6]:
        name = " ".join(w.capitalize() for w in slug.split("-"))
        yield _q("unanswerable", "unknown_service", 1, f"Which team owns the {name} service?", "no_answer", [], g, [],
                 values=[])
    for d in g.by_label("Datastore")[:6]:
        yield _q("unanswerable", "datastore_slo", 1, f"What is the availability SLO of {d.id}?", "no_answer", [], g,
                 [], values=[])
    for v in g.by_label("Vendor")[:6]:
        yield _q("unanswerable", "vendor_contract", 1,
                 f"What is the annual contract value Kalimaya pays {v.name}?", "no_answer", [], g, [], values=[])


TEMPLATES: dict[str, list[Template]] = {
    "direct": [t_owner, t_engine, t_region, t_lead],
    "multihop_2": [t_lead_of_owner, t_datastore_regions, t_division_of_owner, t_dependency_owners],
    "multihop_3plus": [t_datastore_impact, t_incident_division, t_incident_leads],
    "constraint": [t_policy_violations, t_tier1_count, t_classified_services, t_vendor_region,
                   t_depends_on_retiring],
    "summary": [t_team_overview, t_incident_history],
    "unanswerable": [t_unanswerable],
}


def build_gold(g: Graph, seed: int, profile: str) -> list[GoldQuestion]:
    rng = random.Random(seed + 2)
    questions: list[GoldQuestion] = []
    for category in CATEGORIES:
        pools = []
        for template in TEMPLATES[category]:
            pool = list(template(g, rng))
            # Superseded facts are the interesting region questions, so put them first.
            rng.shuffle(pool)
            pool.sort(key=lambda q: "superseded-fact" not in q.notes)
            pools.append(pool)
        picked = _round_robin(pools, QUOTAS[profile][category])
        rng.shuffle(picked)
        n_dev = round(len(picked) * DEV_SHARE)
        for i, q in enumerate(picked):
            q.split = "dev" if i < n_dev else "test"
        questions += picked
    for n, q in enumerate(questions, start=1):
        q.qid = f"q{n:03d}"
    return questions


def _round_robin(pools: list[list[GoldQuestion]], quota: int) -> list[GoldQuestion]:
    picked: list[GoldQuestion] = []
    seen: set[str] = set()
    while len(picked) < quota and any(pools):
        for pool in pools:
            while pool:
                q = pool.pop(0)
                if q.question not in seen:
                    seen.add(q.question)
                    picked.append(q)
                    break
            if len(picked) == quota:
                break
    return picked
