"""Pipeline A: recursive chunking, embeddings, top-k similarity search."""

import hashlib
import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from oagbench.config import EmbedderKind, Settings, StoreBackend
from oagbench.generate import Document
from oagbench.retrieval.context import Context, ContextItem, approx_tokens, fit_budget

CHUNK_TOKENS = 512
OVERLAP_TOKENS = 64


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    doc_type: str
    text: str


def chunk_documents(docs: list[Document], max_tokens: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[Chunk]:
    """Split on headings, then paragraphs, then sentences; pack pieces up to `max_tokens` with overlap."""
    chunks = []
    for doc in docs:
        pieces = [p for section in re.split(r"\n(?=#{1,3} )", doc.text) for p in _split(section, max_tokens)]
        current = ""
        n = 0
        for piece in pieces:
            if current and approx_tokens(current + "\n\n" + piece) > max_tokens:
                chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.doc_type, current))
                n += 1
                current = current[-overlap * 4 :] + "\n\n" + piece
            else:
                current = f"{current}\n\n{piece}" if current else piece
        if current:
            chunks.append(Chunk(f"{doc.doc_id}#{n}", doc.doc_id, doc.doc_type, current))
    return chunks


def _split(text: str, max_tokens: int) -> list[str]:
    if approx_tokens(text) <= max_tokens:
        return [text.strip()]
    for separator in ("\n\n", "\n", ". "):
        parts = [p for p in text.split(separator) if p.strip()]
        if len(parts) > 1:
            return [s for p in parts for s in _split(p, max_tokens)]
    return [text[i : i + max_tokens * 4] for i in range(0, len(text), max_tokens * 4)]


# --- embeddings -------------------------------------------------------------------------------------------------


class Embedder(Protocol):
    name: str

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    """Deterministic feature-hashing embedder (unigrams + bigrams). No download; a weak, dev-only baseline."""

    name = "hash-1024"

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9]+", text.lower())
            for feature in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]:
                h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
                out[row, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(norms == 0, 1, norms)


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True, batch_size=32), dtype=np.float32)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder is EmbedderKind.SENTENCE_TRANSFORMERS:
        return SentenceTransformerEmbedder(settings.embedding_model)
    return HashEmbedder()


# --- stores -------------------------------------------------------------------------------------------------------


class VectorStore(Protocol):
    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]: ...


class MemoryVectorStore:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        self.chunks = chunks
        self.vectors = vectors

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        scores = self.vectors @ vector
        top = np.argsort(-scores)[:k]
        return [(self.chunks[i], float(scores[i])) for i in top]

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "vectors.npy", self.vectors)
        (directory / "chunks.jsonl").write_text("".join(json.dumps(asdict(c)) + "\n" for c in self.chunks),
                                                encoding="utf-8")

    @classmethod
    def load(cls, directory: Path) -> "MemoryVectorStore":
        chunks = [Chunk(**json.loads(line)) for line in
                  (directory / "chunks.jsonl").read_text(encoding="utf-8").splitlines() if line]
        return cls(chunks, np.load(directory / "vectors.npy"))


class QdrantVectorStore:
    def __init__(self, url: str, collection: str):
        from qdrant_client import QdrantClient

        self._client = QdrantClient(url=url)
        self.collection = collection

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray) -> None:
        from qdrant_client import models

        if self._client.collection_exists(self.collection):
            self._client.delete_collection(self.collection)
        self._client.create_collection(
            self.collection, vectors_config=models.VectorParams(size=vectors.shape[1], distance=models.Distance.COSINE)
        )
        self._client.upsert(self.collection, points=[
            models.PointStruct(id=i, vector=vectors[i].tolist(), payload=asdict(c)) for i, c in enumerate(chunks)
        ])

    def search(self, vector: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        hits = self._client.query_points(self.collection, query=vector.tolist(), limit=k).points
        return [(Chunk(**hit.payload), float(hit.score)) for hit in hits]


def index_dir(corpus_path: Path, embedder: Embedder) -> Path:
    return corpus_path / "index" / f"vectors-{embedder.name.replace('/', '_')}"


def build_vector_index(settings: Settings, corpus_path: Path, docs: list[Document], embedder: Embedder) -> dict:
    start = time.perf_counter()
    chunks = chunk_documents(docs)
    vectors = embedder.embed([c.text for c in chunks])
    if settings.store_backend is StoreBackend.SERVICES:
        QdrantVectorStore(settings.qdrant_url, collection_name(corpus_path, embedder)).upsert(chunks, vectors)
    else:
        MemoryVectorStore(chunks, vectors).save(index_dir(corpus_path, embedder))
    return {"chunks": len(chunks), "embedder": embedder.name, "seconds": round(time.perf_counter() - start, 2)}


def open_vector_store(settings: Settings, corpus_path: Path, embedder: Embedder) -> VectorStore:
    if settings.store_backend is StoreBackend.SERVICES:
        return QdrantVectorStore(settings.qdrant_url, collection_name(corpus_path, embedder))
    directory = index_dir(corpus_path, embedder)
    if not (directory / "vectors.npy").exists():
        raise FileNotFoundError(f"no vector index at {directory}; run `oagbench ingest vector` first")
    return MemoryVectorStore.load(directory)


def collection_name(corpus_path: Path, embedder: Embedder) -> str:
    return f"chunks_{corpus_path.name}_{re.sub(r'[^a-z0-9]+', '_', embedder.name.lower())}"


class VectorRetriever:
    name = "A"

    def __init__(self, store: VectorStore, embedder: Embedder, max_k: int = 20):
        self.store = store
        self.embedder = embedder
        self.max_k = max_k

    def retrieve(self, question: str, budget: int) -> Context:
        start = time.perf_counter()
        hits = self.store.search(self.embedder.embed([question])[0], self.max_k)
        items = [ContextItem("chunk", c.doc_id, c.text, score) for c, score in hits]
        return Context(items=fit_budget(items, budget),
                       timings_ms={"retrieve": int((time.perf_counter() - start) * 1000)})
