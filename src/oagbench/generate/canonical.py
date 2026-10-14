"""Builds the canonical (ground-truth) graph of the fictional company Kalimaya Digital from a seed.

Every relation records the document that states it (`source_doc`); document ids are derived from entity ids,
so the renderer and the gold set agree without coordination.
"""

import random
from dataclasses import dataclass

from oagbench.graph import Entity, Graph, Relation


@dataclass(frozen=True)
class Profile:
    teams: int
    services: int
    datastores: int
    incidents: int
    migrations: int
    rfcs: int


PROFILES = {
    "small": Profile(teams=6, services=25, datastores=12, incidents=10, migrations=2, rfcs=2),
    "full": Profile(teams=15, services=80, datastores=35, incidents=40, migrations=5, rfcs=4),
}

ORG_UNITS = [("ou-payments", "Payments Division"), ("ou-platform", "Platform Engineering"),
             ("ou-consumer", "Consumer Products"), ("ou-risk", "Risk and Compliance")]

TEAM_NAMES = ["Payments Core", "Ledger", "Risk Engine", "Identity", "Merchant Onboarding", "Notifications",
              "Data Platform", "Lending", "Fraud Analytics", "Care Tools", "Mobile Gateway", "Treasury",
              "Compliance Tech", "Search", "Growth"]

FIRST_NAMES = ["Arif", "Bunga", "Citra", "Dimas", "Eka", "Fajar", "Gilang", "Hana", "Indah", "Joko", "Kartika",
               "Lestari", "Made", "Nadia", "Oka", "Putri", "Raka", "Sari", "Tegar", "Utami", "Wulan", "Yoga"]
LAST_NAMES = ["Anggara", "Basuki", "Cahyadi", "Darmawan", "Effendi", "Gunawan", "Halim", "Irawan", "Kusuma",
              "Lubis", "Maharani", "Nugroho", "Pratama", "Rahayu", "Santoso", "Tanjung", "Wibowo", "Yuliani"]
ROLES = ["Backend Engineer", "Senior Backend Engineer", "SRE", "Data Engineer", "QA Engineer", "Mobile Engineer"]

SERVICE_DOMAINS = ["payments", "ledger", "wallet", "kyc", "payout", "refund", "invoice", "merchant", "loan",
                   "scoring", "fraud", "notify", "auth", "profile", "search", "promo", "fx", "settlement",
                   "dispute", "reporting", "billing", "card", "qris", "audit"]
SERVICE_PARTS = ["api", "worker", "gateway", "sync", "engine", "scheduler", "indexer", "adapter"]

REGIONS = [("region-jkt", "Jakarta", "Indonesia"), ("region-sby", "Surabaya", "Indonesia"),
           ("region-btm", "Batam", "Indonesia"), ("region-sgp", "Singapore", "Singapore"),
           ("region-fra", "Frankfurt", "Germany")]
REGION_WEIGHTS = [50, 15, 10, 18, 7]

DATA_CLASSES = ["PII", "PCI", "Financial", "Internal", "Public"]

VENDORS = [("vendor-nusapay", "Nusapay Gateway", "payments"), ("vendor-awanmail", "Awan Mail", "email"),
           ("vendor-cendana", "Cendana SMS", "messaging"), ("vendor-lintas", "Lintas KYC", "identity"),
           ("vendor-pelangi", "Pelangi Maps", "maps"), ("vendor-arunika", "Arunika Analytics", "analytics"),
           ("vendor-bayu", "Bayu CDN", "cdn"), ("vendor-kirana", "Kirana OCR", "document AI")]

ENGINES = [("postgres", "pg", "Postgres", ["15.6", "16.2"]), ("redis", "redis", "Redis", ["7.2"]),
           ("kafka", "kafka", "Kafka", ["3.6", "3.7"]), ("s3", "s3", "object storage", ["n/a"]),
           ("mongo", "mongo", "MongoDB", ["6.0", "7.0"])]
ENGINE_WEIGHTS = [45, 20, 15, 10, 10]
DATASTORE_PURPOSES = ["core", "events", "cache", "archive", "ops", "analytics", "session", "ledger"]

POLICIES = [
    ("pol-res-01", "Indonesian Data Residency", ["PII", "Financial"], ["region-jkt", "region-sby", "region-btm"],
     None, "Services processing PII or Financial data, and every datastore they use, must run in an "
           "Indonesian region (Jakarta, Surabaya or Batam)."),
    ("pol-pci-02", "Cardholder Data Standard", ["PCI"], ["region-jkt", "region-sgp"], "T1",
     "Services processing PCI data must be tier T1 and run only in Jakarta or Singapore, together with "
     "their datastores."),
    ("pol-pub-03", "Public Content Policy", ["Public"], [r[0] for r in REGIONS], None,
     "Public data may be hosted in any approved region."),
]

SEVERITIES = ["SEV1", "SEV2", "SEV3", "SEV4"]


def doc_id_for(entity_id: str) -> str:
    """Document that describes an entity. Service, team, datastore, policy and incident each get one."""
    return entity_id


def build_canonical(seed: int, profile: Profile) -> tuple[Graph, list[dict]]:
    """Return the canonical graph and the list of region migrations (superseded facts)."""
    rng = random.Random(seed)
    g = Graph()

    def rel(type_: str, src: str, dst: str, doc: str, valid_from: str = "") -> None:
        g.add_relation(Relation(type_, src, dst, source_doc=doc, valid_from=valid_from))

    for region_id, name, country in REGIONS:
        g.add_entity(Entity(region_id, "Region", name, {"country": country}))
    for dc in DATA_CLASSES:
        g.add_entity(Entity(f"dc-{dc.lower()}", "DataClass", dc))
    for vendor_id, name, category in VENDORS:
        g.add_entity(Entity(vendor_id, "Vendor", name, {"category": category}, source_docs=["vendors"]))
    for ou_id, name in ORG_UNITS:
        g.add_entity(Entity(ou_id, "OrgUnit", name, source_docs=["org-overview"]))

    # Teams and people
    used_people: set[str] = set()
    teams: list[str] = []
    for i, team_name in enumerate(TEAM_NAMES[: profile.teams]):
        team_id = "team-" + team_name.lower().replace(" ", "-")
        teams.append(team_id)
        g.add_entity(Entity(team_id, "Team", team_name, {
            "slack_channel": "#" + team_id.removeprefix("team-"),
            "oncall_rotation": rng.choice(["weekly", "bi-weekly", "follow-the-sun"]),
        }, aliases=[f"{team_name} team"], source_docs=[doc_id_for(team_id)]))
        rel("CONTAINS", ORG_UNITS[i % len(ORG_UNITS)][0], team_id, "org-overview")
        members = []
        for _ in range(rng.randint(3, 5)):
            person = _unique_person(rng, used_people)
            person_id = "person-" + person.lower().replace(" ", "-")
            g.add_entity(Entity(person_id, "Person", person, {"role": rng.choice(ROLES)},
                                source_docs=[doc_id_for(team_id)]))
            rel("HAS_MEMBER", team_id, person_id, doc_id_for(team_id))
            members.append(person_id)
        lead = members[0]
        g.entities[lead].props["role"] = "Engineering Manager"
        rel("LEADS", lead, team_id, doc_id_for(team_id))

    # Datastores
    datastores: list[str] = []
    counters: dict[str, int] = {}
    for _ in range(profile.datastores):
        engine, prefix, display, versions = rng.choices(ENGINES, ENGINE_WEIGHTS)[0]
        purpose = rng.choice(DATASTORE_PURPOSES)
        key = f"{prefix}-{purpose}"
        counters[key] = counters.get(key, 0) + 1
        ds_id = f"{key}-{counters[key]:02d}"
        datastores.append(ds_id)
        alias = f"{purpose} {display} cluster {counters[key]}"
        g.add_entity(Entity(ds_id, "Datastore", ds_id, {"engine": engine, "version": rng.choice(versions)},
                            aliases=[alias], source_docs=[doc_id_for(ds_id)]))
        region = rng.choices(REGIONS, REGION_WEIGHTS)[0][0]
        rel("HOSTED_IN", ds_id, region, doc_id_for(ds_id))

    # Services. Index order is a layering: a service only depends on services with a higher index,
    # so DEPENDS_ON is acyclic and has long chains for multi-hop questions.
    names = [f"{d}-{p}" for d in SERVICE_DOMAINS for p in SERVICE_PARTS]
    rng.shuffle(names)
    services = [f"svc-{n}" for n in names[: profile.services]]
    for idx, svc_id in enumerate(services):
        slug = svc_id.removeprefix("svc-")
        display = " ".join(w.upper() if w in {"api", "kyc", "fx", "qris"} else w.capitalize()
                           for w in slug.split("-"))
        tier = rng.choices(["T1", "T2", "T3"], [30, 45, 25])[0]
        lifecycle = rng.choices(["active", "deprecated", "retired"], [85, 10, 5])[0]
        slo = {"T1": "99.95%", "T2": "99.9%", "T3": "99.5%"}[tier]
        doc = doc_id_for(svc_id)
        g.add_entity(Entity(svc_id, "Service", display,
                            {"tier": tier, "slo_availability": slo, "lifecycle": lifecycle},
                            aliases=[slug], source_docs=[doc]))
        rel("OWNS", rng.choice(teams), svc_id, doc)
        rel("DEPLOYED_IN", svc_id, rng.choices(REGIONS, REGION_WEIGHTS)[0][0], doc, "2024-01-15")
        lower = services[idx + 1 :]
        for dep in rng.sample(lower, k=min(len(lower), rng.choices([0, 1, 2, 3], [20, 40, 30, 10])[0])):
            rel("DEPENDS_ON", svc_id, dep, doc)
        for ds in rng.sample(datastores, k=rng.choices([1, 2], [65, 35])[0]):
            rel("USES_DATASTORE", svc_id, ds, doc)
        for dc in rng.sample(DATA_CLASSES, k=rng.choices([0, 1, 2], [25, 50, 25])[0]):
            rel("PROCESSES", svc_id, f"dc-{dc.lower()}", doc)
        if rng.random() < 0.35:
            rel("USES_VENDOR", svc_id, rng.choice(VENDORS)[0], doc)

    for pol_id, name, classes, regions, min_tier, rule in POLICIES:
        props = {"rule_text": rule}
        if min_tier:
            props["min_tier"] = min_tier
        g.add_entity(Entity(pol_id, "Policy", name, props, aliases=[pol_id.upper()], source_docs=[pol_id]))
        for dc in classes:
            rel("GOVERNS", pol_id, f"dc-{dc.lower()}", pol_id)
        for region in regions:
            rel("ALLOWS_REGION", pol_id, region, pol_id)

    # Incidents: affected services are the root cause plus some of its transitive dependents.
    for n in range(1, profile.incidents + 1):
        inc_id = f"inc-2025-{n:03d}"
        root = rng.choice(services)
        dependents = sorted(g.transitive_in(root, "DEPENDS_ON"))
        affected = [root, *rng.sample(dependents, k=min(len(dependents), rng.randint(0, 3)))]
        month, day = rng.randint(1, 12), rng.randint(1, 28)
        owner = g.one_in(root, "OWNS")
        resolver = owner if rng.random() < 0.8 else rng.choice(teams)
        g.add_entity(Entity(inc_id, "Incident", inc_id.upper(), {
            "occurred_on": f"2025-{month:02d}-{day:02d}",
            "severity": rng.choices(SEVERITIES, [10, 30, 40, 20])[0],
            "duration_min": str(rng.choice([12, 25, 40, 75, 120, 210])),
        }, source_docs=[inc_id]))
        rel("ROOT_CAUSE", inc_id, root, inc_id)
        for svc in affected:
            rel("AFFECTED", inc_id, svc, inc_id)
        rel("RESOLVED_BY", inc_id, resolver, inc_id)

    # Superseded facts: some services moved region. The service page keeps the old region (dated 2024),
    # a later migration notice states the new one. The canonical answer is the newest fact.
    migrations = []
    for n, svc_id in enumerate(rng.sample(services, k=profile.migrations), start=1):
        old = g.one_out(svc_id, "DEPLOYED_IN")
        new = rng.choice([r[0] for r in REGIONS if r[0] != old])
        doc = f"migration-{n:02d}"
        date = f"2025-{rng.randint(2, 11):02d}-01"
        rel("DEPLOYED_IN", svc_id, new, doc, date)
        g.entities[svc_id].source_docs.append(doc)
        migrations.append({"doc_id": doc, "service": svc_id, "old": old, "new": new, "date": date})

    return g, migrations


def _unique_person(rng: random.Random, used: set[str]) -> str:
    while True:
        name = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if name not in used:
            used.add(name)
            return name
