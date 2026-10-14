"""Corpus generation and loading."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from oagbench import __version__
from oagbench.generate.canonical import PROFILES, build_canonical
from oagbench.generate.gold import GoldQuestion, build_gold
from oagbench.generate.render import Document, render_corpus
from oagbench.graph import Graph


def corpus_id(profile: str, seed: int) -> str:
    return f"{profile}-s{seed}"


def generate_corpus(data_dir: Path, profile: str, seed: int) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from {sorted(PROFILES)}")
    graph, migrations = build_canonical(seed, PROFILES[profile])
    docs = render_corpus(graph, migrations, seed, PROFILES[profile].rfcs)
    gold = build_gold(graph, seed, profile)

    out = data_dir / corpus_id(profile, seed)
    (out / "docs").mkdir(parents=True, exist_ok=True)
    for stale in (out / "docs").glob("*.md"):
        stale.unlink()
    for doc in docs:
        (out / "docs" / f"{doc.doc_id}.md").write_text(doc.text + "\n", encoding="utf-8")
    graph.save(out / "canonical.json")
    (out / "gold.jsonl").write_text("".join(json.dumps(q.to_dict()) + "\n" for q in gold), encoding="utf-8")

    digest = hashlib.sha256()
    for doc in docs:
        digest.update(doc.doc_id.encode() + b"\0" + doc.text.encode())
    manifest = {
        "corpus_id": corpus_id(profile, seed),
        "corpus_hash": digest.hexdigest()[:16],
        "seed": seed,
        "profile": profile,
        "generator_version": __version__,
        "documents": [{"doc_id": d.doc_id, "doc_type": d.doc_type, "title": d.title} for d in docs],
        "questions": len(gold),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return out


@dataclass
class Corpus:
    path: Path
    manifest: dict
    documents: list[Document]
    gold: list[GoldQuestion]
    canonical: Graph

    @property
    def corpus_id(self) -> str:
        return self.manifest["corpus_id"]

    def doc(self, doc_id: str) -> Document | None:
        return next((d for d in self.documents if d.doc_id == doc_id), None)

    def questions(self, split: str) -> list[GoldQuestion]:
        return [q for q in self.gold if split == "all" or q.split == split]


def load_corpus(path: Path) -> Corpus:
    if not (path / "manifest.json").exists():
        raise FileNotFoundError(f"no corpus at {path}; run `oagbench generate` first")
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    documents = [
        Document(m["doc_id"], m["doc_type"], m["title"],
                 (path / "docs" / f"{m['doc_id']}.md").read_text(encoding="utf-8"))
        for m in manifest["documents"]
    ]
    gold = [GoldQuestion(**json.loads(line))
            for line in (path / "gold.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return Corpus(path, manifest, documents, gold, Graph.load(path / "canonical.json"))


__all__ = ["Corpus", "Document", "GoldQuestion", "corpus_id", "generate_corpus", "load_corpus"]
