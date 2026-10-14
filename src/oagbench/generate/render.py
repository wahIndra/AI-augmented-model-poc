"""Renders the canonical graph into Markdown documents.

Facts are spread across document types on purpose (a service page names its owning team, but only the team page
names the lead, and only the org overview names the org unit) so multi-hop questions must cross documents.
Entities are referred to by varying names (id, display name, alias) to exercise entity resolution.
"""

import random
from dataclasses import dataclass

from oagbench.graph import Graph


@dataclass(frozen=True)
class Document:
    doc_id: str
    doc_type: str
    title: str
    text: str


def render_corpus(g: Graph, migrations: list[dict], seed: int, rfcs: int) -> list[Document]:
    rng = random.Random(seed + 1)
    docs = [_org_overview(g), _vendors(g)]
    docs += [_team_page(g, t.id, rng) for t in g.by_label("Team")]
    docs += [_service_page(g, s.id, rng) for s in g.by_label("Service")]
    docs += [_datastore_runbook(g, d.id, rng) for d in g.by_label("Datastore")]
    docs += [_policy(g, p.id) for p in g.by_label("Policy")]
    docs += [_postmortem(g, i.id, rng) for i in g.by_label("Incident")]
    docs += [_migration(g, m) for m in migrations]
    docs += _rfcs(g, rng, rfcs)
    return docs


def _ref(g: Graph, entity_id: str, rng: random.Random) -> str:
    """Refer to an entity by one of its names."""
    entity = g.entities[entity_id]
    return rng.choice(entity.names()) if entity.label in {"Service", "Datastore"} else entity.name


def _join(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _org_overview(g: Graph) -> Document:
    lines = ["# Kalimaya Digital: organisation overview", "",
             "Kalimaya Digital is organised into divisions. Each division groups several engineering teams.", ""]
    for ou in g.by_label("OrgUnit"):
        teams = [g.name_of(t) for t in g.out(ou.id, "CONTAINS")]
        if teams:
            lines += [f"## {ou.name}", "", f"The {ou.name} contains the following teams: {_join(teams)}.", ""]
    return Document("org-overview", "org", "Organisation overview", "\n".join(lines))


def _vendors(g: Graph) -> Document:
    lines = ["# Approved vendors", "", "| Vendor | Category |", "| --- | --- |"]
    lines += [f"| {v.name} | {v.props['category']} |" for v in g.by_label("Vendor")]
    return Document("vendors", "vendor", "Approved vendors", "\n".join(lines))


def _team_page(g: Graph, team_id: str, rng: random.Random) -> Document:
    team = g.entities[team_id]
    lead = g.one_in(team_id, "LEADS")
    members = [m for m in g.out(team_id, "HAS_MEMBER") if m != lead]
    intro = rng.choice([
        f"The {team.name} team is led by {g.name_of(lead)}.",
        f"{g.name_of(lead)} is the engineering manager of the {team.name} team.",
        f"{team.name} reports to {g.name_of(lead)}, who leads the team.",
    ])
    lines = [f"# {team.name} team", "", intro, "", "## Members", ""]
    lines += [f"- {g.name_of(m)} ({g.entities[m].props['role']})" for m in members]
    lines += ["", "## Ways of working", "",
              f"- Slack: {team.props['slack_channel']}",
              f"- On-call rotation: {team.props['oncall_rotation']}",
              "- Planning happens every second Monday; requests go through the team channel."]
    return Document(team_id, "team", f"{team.name} team", "\n".join(lines))


def _service_page(g: Graph, svc_id: str, rng: random.Random) -> Document:
    svc = g.entities[svc_id]
    p = svc.props
    owner = g.one_in(svc_id, "OWNS")
    # The service page is dated 2024 and states the original region; migrations are documented separately.
    original_region = next(r.dst for r in g.out_relations(svc_id, "DEPLOYED_IN") if r.source_doc == svc_id)
    owner_line = rng.choice([
        f"{svc.name} is owned by the {g.name_of(owner)} team.",
        f"Owner: {g.name_of(owner)}.",
        f"The {g.name_of(owner)} team maintains this service.",
    ])
    lines = [f"# {svc.name} (`{svc.aliases[0]}`)", "", "_Last reviewed: 2024-01-15_", "", owner_line, "",
             "## Operations", "",
             f"- Tier: {p['tier']} (availability SLO {p['slo_availability']})",
             f"- Lifecycle: {p['lifecycle']}",
             f"- Deployed in the {g.name_of(original_region)} region."]
    deps = g.out(svc_id, "DEPENDS_ON")
    stores = g.out(svc_id, "USES_DATASTORE")
    lines += ["", "## Dependencies", ""]
    if deps:
        lines.append(rng.choice(["Calls at runtime: ", "Upstream services it depends on: "])
                     + _join([_ref(g, d, rng) for d in deps]) + ".")
    else:
        lines.append("No runtime dependencies on other internal services.")
    lines.append("Datastores: " + _join([_ref(g, d, rng) for d in stores]) + ".")
    vendors = g.out(svc_id, "USES_VENDOR")
    if vendors:
        lines.append("External vendors: " + _join([g.name_of(v) for v in vendors]) + ".")
    classes = [g.name_of(c) for c in g.out(svc_id, "PROCESSES")]
    lines += ["", "## Data handled", ""]
    lines.append(f"This service processes {_join(classes)} data." if classes else "No classified data.")
    return Document(svc_id, "service", svc.name, "\n".join(lines))


def _datastore_runbook(g: Graph, ds_id: str, rng: random.Random) -> Document:
    ds = g.entities[ds_id]
    region = g.one_out(ds_id, "HOSTED_IN")
    lines = [f"# Runbook: {ds_id}", "",
             f"Also known as the {ds.aliases[0]}.", "",
             f"- Engine: {ds.props['engine']} {ds.props['version']}",
             f"- Hosted in: {g.name_of(region)}",
             "", "## Failover", "",
             rng.choice(["Promote the standby replica, then update the service discovery record.",
                         "Fail over with the platform CLI; expect up to two minutes of write unavailability.",
                         "Restore from the latest snapshot if the replica is unhealthy."]),
             "", "## Contacts", "", "Page the on-call engineer of any service that uses this datastore."]
    return Document(ds_id, "runbook", f"Runbook {ds_id}", "\n".join(lines))


def _policy(g: Graph, pol_id: str) -> Document:
    pol = g.entities[pol_id]
    classes = [g.name_of(c) for c in g.out(pol_id, "GOVERNS")]
    regions = [g.name_of(r) for r in g.out(pol_id, "ALLOWS_REGION")]
    lines = [f"# {pol.name} ({pol_id.upper()})", "", pol.props["rule_text"], "",
             f"- Governs data classes: {_join(classes)}",
             f"- Allowed regions: {_join(regions)}"]
    if "min_tier" in pol.props:
        lines.append(f"- Minimum service tier: {pol.props['min_tier']}")
    return Document(pol_id, "policy", pol.name, "\n".join(lines))


def _postmortem(g: Graph, inc_id: str, rng: random.Random) -> Document:
    inc = g.entities[inc_id]
    p = inc.props
    root = g.one_out(inc_id, "ROOT_CAUSE")
    affected = [a for a in g.out(inc_id, "AFFECTED") if a != root]
    resolver = g.one_out(inc_id, "RESOLVED_BY")
    lines = [f"# Postmortem {inc.name}", "",
             f"- Date: {p['occurred_on']}", f"- Severity: {p['severity']}", f"- Duration: {p['duration_min']} minutes",
             "", "## Summary", "",
             rng.choice([f"A faulty deployment of {_ref(g, root, rng)} caused elevated error rates.",
                         f"The root cause was resource exhaustion in {_ref(g, root, rng)}.",
                         f"{_ref(g, root, rng)} failed after a configuration change."])]
    if affected:
        lines.append("Downstream impact was observed on " + _join([_ref(g, a, rng) for a in affected]) + ".")
    lines += ["", "## Resolution", "",
              f"The {g.name_of(resolver)} team mitigated the incident and owns the follow-ups."]
    return Document(inc_id, "postmortem", f"Postmortem {inc.name}", "\n".join(lines))


def _migration(g: Graph, m: dict) -> Document:
    svc = g.entities[m["service"]]
    lines = [f"# Migration notice: {svc.name}", "", f"_Effective {m['date']}_", "",
             f"{svc.name} (`{svc.aliases[0]}`) has moved from {g.name_of(m['old'])} to {g.name_of(m['new'])}. "
             f"From {m['date']} it is deployed in the {g.name_of(m['new'])} region. "
             "The service page will be updated in the next review cycle."]
    return Document(m["doc_id"], "migration", f"Migration notice {svc.name}", "\n".join(lines))


def _rfcs(g: Graph, rng: random.Random, count: int) -> list[Document]:
    """Distractors: proposals that mention dependencies which do not exist."""
    services = [s.id for s in g.by_label("Service")]
    docs = []
    for n in range(1, count + 1):
        a, b = rng.sample(services, 2)
        lines = [f"# RFC {n:03d}: route {g.name_of(a)} through {g.name_of(b)}", "", "_Status: rejected_", "",
                 f"This proposal suggested that {g.name_of(a)} should call {g.name_of(b)} directly. "
                 "The architecture review rejected it; the dependency was never built."]
        docs.append(Document(f"rfc-{n:03d}", "rfc", f"RFC {n:03d}", "\n".join(lines)))
    return docs
