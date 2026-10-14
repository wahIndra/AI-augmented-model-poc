"""Loads the ontology and derives the extraction schema and the schema text for text-to-Cypher."""

from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

import yaml


@dataclass(frozen=True)
class RelationType:
    name: str
    source: str
    target: str
    functional: bool = False  # at most one target per source
    inverse_functional: bool = False  # at most one source per target
    transitive: bool = False
    description: str = ""


@dataclass(frozen=True)
class Ontology:
    name: str
    classes: dict[str, dict[str, Any]]
    relations: dict[str, RelationType]

    def schema_text(self) -> str:
        """Compact schema description for prompts."""
        lines = ["Node labels (every node has `id`, `name`, `aliases` (list), `source_docs` (list)):"]
        for label, spec in self.classes.items():
            props = ", ".join(_prop_text(k, v) for k, v in spec["properties"].items()) or "-"
            lines.append(f"- {label}: {spec['description']} Properties: {props}")
        lines.append("Relationships (every relationship has `source_doc` and `valid_from` (ISO date or '')):")
        for rel in self.relations.values():
            notes = [n for n, on in (("transitive", rel.transitive), ("functional", rel.functional)) if on]
            suffix = f" [{', '.join(notes)}]" if notes else ""
            lines.append(f"- (:{rel.source})-[:{rel.name}]->(:{rel.target}){suffix}")
        return "\n".join(lines)

    def extraction_schema(self) -> dict[str, Any]:
        """JSON Schema for LLM triple extraction. Strict-mode compatible (closed objects, all required)."""
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["entities", "relations"],
            "properties": {
                "entities": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["label", "name", "aliases", "properties"],
                        "properties": {
                            "label": {"type": "string", "enum": list(self.classes)},
                            "name": {"type": "string"},
                            "aliases": {"type": "array", "items": {"type": "string"}},
                            "properties": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": ["key", "value"],
                                    "properties": {"key": {"type": "string"}, "value": {"type": "string"}},
                                },
                            },
                        },
                    },
                },
                "relations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["type", "source", "target", "valid_from"],
                        "properties": {
                            "type": {"type": "string", "enum": list(self.relations)},
                            "source": {"type": "string"},
                            "target": {"type": "string"},
                            "valid_from": {"type": "string"},
                        },
                    },
                },
            },
        }


def _prop_text(name: str, spec: Any) -> str:
    if isinstance(spec, dict) and "enum" in spec:
        return f"{name} ({'|'.join(spec['enum'])})"
    return name


@cache
def load_ontology() -> Ontology:
    raw = yaml.safe_load(resources.files(__package__).joinpath("kalimaya.yaml").read_text(encoding="utf-8"))
    relations = {
        name: RelationType(
            name=name,
            source=spec["from"],
            target=spec["to"],
            functional=spec.get("functional", False),
            inverse_functional=spec.get("inverse_functional", False),
            transitive=spec.get("transitive", False),
            description=spec.get("description", ""),
        )
        for name, spec in raw["relations"].items()
    }
    return Ontology(name=raw["name"], classes=raw["classes"], relations=relations)
