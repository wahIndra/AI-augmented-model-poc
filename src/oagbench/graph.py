"""In-memory property graph. Holds the canonical graph, the extracted graph, and backs the memory store."""

import json
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


def normalize(text: str) -> str:
    """Lowercase; every run of non-alphanumeric characters becomes one space: 'PG-Core_03?' -> 'pg core 03'."""
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


@dataclass
class Entity:
    id: str
    label: str
    name: str
    props: dict[str, Any] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    source_docs: list[str] = field(default_factory=list)

    def names(self) -> list[str]:
        return [self.name, self.id, *self.aliases]


@dataclass(frozen=True)
class Relation:
    type: str
    src: str
    dst: str
    source_doc: str = ""
    valid_from: str = ""


class Graph:
    def __init__(self, entities: Iterable[Entity] = (), relations: Iterable[Relation] = ()):
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self._out: dict[tuple[str, str], list[Relation]] = defaultdict(list)
        self._in: dict[tuple[str, str], list[Relation]] = defaultdict(list)
        for entity in entities:
            self.add_entity(entity)
        for relation in relations:
            self.add_relation(relation)

    def add_entity(self, entity: Entity) -> None:
        self.entities[entity.id] = entity

    def add_relation(self, relation: Relation) -> None:
        if relation in self._out[(relation.src, relation.type)]:
            return
        self.relations.append(relation)
        self._out[(relation.src, relation.type)].append(relation)
        self._in[(relation.dst, relation.type)].append(relation)

    # --- navigation -------------------------------------------------------------------------------

    def by_label(self, label: str) -> list[Entity]:
        return sorted((e for e in self.entities.values() if e.label == label), key=lambda e: e.id)

    def out_relations(self, node_id: str, rel_type: str) -> list[Relation]:
        return list(self._out.get((node_id, rel_type), []))

    def out(self, node_id: str, rel_type: str) -> list[str]:
        return [r.dst for r in self._out.get((node_id, rel_type), [])]

    def inn(self, node_id: str, rel_type: str) -> list[str]:
        return [r.src for r in self._in.get((node_id, rel_type), [])]

    def one_out(self, node_id: str, rel_type: str) -> str | None:
        """Target of a functional relation; the latest `valid_from` wins when facts were superseded."""
        rels = self._out.get((node_id, rel_type), [])
        if not rels:
            return None
        return max(rels, key=lambda r: r.valid_from).dst

    def one_in(self, node_id: str, rel_type: str) -> str | None:
        rels = self._in.get((node_id, rel_type), [])
        return max(rels, key=lambda r: r.valid_from).src if rels else None

    def relations_of(self, node_id: str) -> list[Relation]:
        found = [r for (src, _), rels in self._out.items() if src == node_id for r in rels]
        found += [r for (dst, _), rels in self._in.items() if dst == node_id for r in rels]
        return found

    def transitive_in(self, node_id: str, rel_type: str) -> set[str]:
        """All nodes that reach `node_id` through one or more `rel_type` edges."""
        seen: set[str] = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for src in self.inn(current, rel_type):
                if src not in seen:
                    seen.add(src)
                    frontier.append(src)
        return seen

    def name_of(self, node_id: str) -> str:
        entity = self.entities.get(node_id)
        return entity.name if entity else node_id

    # --- aliases ---------------------------------------------------------------------------------

    def alias_index(self) -> dict[str, str]:
        """Normalised name/id/alias -> entity id. Longer keys are preferred when matching text."""
        index: dict[str, str] = {}
        for entity in self.entities.values():
            for name in entity.names():
                key = normalize(name)
                if key:
                    index.setdefault(key, entity.id)
        return index

    def find_mentions(self, text: str, index: dict[str, str] | None = None) -> list[str]:
        """Entity ids mentioned in free text, longest alias first, without overlapping matches."""
        index = index or self.alias_index()
        haystack = f" {normalize(text)} "
        found: list[str] = []
        for key in sorted(index, key=len, reverse=True):
            needle = f" {key} "
            if needle in haystack:
                entity_id = index[key]
                if entity_id not in found:
                    found.append(entity_id)
                haystack = haystack.replace(needle, " | ")
        return found

    # --- persistence -----------------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "entities": [asdict(e) for e in sorted(self.entities.values(), key=lambda e: e.id)],
            "relations": [asdict(r) for r in self.relations],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Graph":
        return cls(
            entities=(Entity(**e) for e in data["entities"]),
            relations=(Relation(**r) for r in data["relations"]),
        )

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1, sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Graph":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
