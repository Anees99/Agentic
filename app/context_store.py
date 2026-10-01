"""Hybrid context layer: ChromaDB vector store + NetworkX knowledge graph.

Architectural rationale
-----------------------
Pure vector RAG fails on *relational* enterprise questions ("which app runs on
Server_A?") because embeddings capture similarity, not structure.  Pure graphs
fail on fuzzy natural-language questions because entities never match exactly.
Production retrieval systems therefore run **both** and fuse the results —
that is what this module provides as a single read surface.

Design decisions:

* **Write once, read many.**  Ingestion (:mod:`scripts.data_ingestion`) owns
  writes; this module only ever opens stores read-only-ish (Chroma's persistent
  client is idempotent for reads).  This keeps API workers from racing with
  re-ingestion jobs.
* **Graceful degradation.**  If either backend is missing/empty, retrieval
  still returns whatever the other backend found plus an explicit provenance
  note, instead of raising.  A partially-broken memory should degrade answer
  quality, not availability.
* **Deterministic offline embeddings.**  ChromaDB's default embedding function
  downloads ``all-MiniLM-L6-v2`` on first use.  For CI/offline demos we inject
  a tiny hash-based embedding so ingestion and querying remain *mutually
  consistent* (same function both sides) without network access.  Set
  ``USE_DEFAULT_EMBEDDINGS=true`` to get real semantic search.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = 256


def deterministic_embedding(text: str) -> list[float]:
    """Produce a stable pseudo-embedding from repeated SHA-256 expansion.

    Not semantically meaningful — it exists so that *offline ingestion and
    offline querying share one identical embedding space*.  Both sides must
    use this exact function or cosine similarity becomes noise.  Swap in a
    real model by setting ``USE_DEFAULT_EMBEDDINGS=true``.
    """
    digest = b""
    counter = 0
    while len(digest) < EMBEDDING_DIMENSIONS * 4:  # 4 bytes per float32 slot
        seed = f"{text}:{counter}".encode("utf-8")
        digest += hashlib.sha256(seed).digest()
        counter += 1
    values = [b / 255.0 for b in digest[: EMBEDDING_DIMENSIONS * 4 : 4]]
    norm = sum(v * v for v in values) ** 0.5 or 1.0
    return [v / norm for v in values]


class HashEmbeddingFunction:
    """ChromaDB ``EmbeddingFunction`` adapter over :func:`deterministic_embedding`.

    Implements the full chromadb>=1.x protocol surface (``name()``,
    ``get_config()``, ``build_from_config()``, ``embed_query()``) so the
    collection configuration round-trips through persistence without falling
    back to the default ONNX model — which would require a network download
    on first use.  We deliberately do *not* subclass the Protocol base: its
    ``__init_subclass__`` machinery re-wraps ``__call__`` in ways that break
    plain duck-typed adapters, while Chroma's runtime validation only needs
    these methods to exist.
    """

    _EF_NAME = "deterministic-hash-v0"

    def __call__(self, input: list[str]) -> list[list[float]]:  # noqa: A002 - Chroma API
        return [deterministic_embedding(t) for t in input]

    def embed_query(self, input: list[str]) -> list[list[float]]:  # noqa: A002
        """Query-side embedding; symmetric model, so identical to documents."""
        return self.__call__(input)

    def name(self) -> str:
        return self._EF_NAME

    def get_config(self) -> dict[str, Any]:
        return {"dimensions": EMBEDDING_DIMENSIONS}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> "HashEmbeddingFunction":
        # Stateless function: nothing to reconstruct from config.
        return HashEmbeddingFunction()


# Backwards-compatible private alias used by the ingestion script.
_HashEmbeddingFunction = HashEmbeddingFunction


@dataclass
class RetrievedChunk:
    """One unit of fused retrieval output with full provenance.

    Provenance matters: the Coder prompt labels each fact's origin so the
    Evaluator can later distinguish grounded claims from hallucinations.
    """

    source: str          # "chroma:<doc_id>" or "kg:<edge>"
    text: str
    score: float | None = None


@dataclass
class HybridContext:
    """Aggregated retrieval result across both memory backends."""

    chunks: list[RetrievedChunk] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_prompt_block(self) -> str:
        """Render the fused context as a labelled block for LLM prompts."""
        if not self.chunks:
            return "(no context retrieved)"
        lines = [f"[{c.source}] {c.text}" for c in self.chunks]
        return "\n".join(lines)


class HybridContextStore:
    """Unified read facade over ChromaDB (unstructured) and NetworkX KG (structured)."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

        self._client = chromadb.PersistentClient(
            path=str(self._settings.chroma_persist_dir),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        use_default_ef = _env_truthy("USE_DEFAULT_EMBEDDINGS")
        ef: Any = None if use_default_ef else _HashEmbeddingFunction()
        # get_or_create keeps readers robust against a not-yet-run ingestion;
        # queries then simply return zero hits (handled gracefully below).
        self._collection = self._client.get_or_create_collection(
            name=self._settings.chroma_collection_name,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
        self._graph: nx.MultiDiGraph | None = None

    # ------------------------------------------------------------------
    # Knowledge graph side
    # ------------------------------------------------------------------
    def _load_graph(self) -> nx.MultiDiGraph | None:
        """Lazily load the pickled graph produced by the ingestion script."""
        if self._graph is None:
            path = self._settings.knowledge_graph_path
            if path.exists():
                with path.open("rb") as fh:
                    self._graph = pickle.load(fh)
                logger.info("Knowledge graph loaded: %d nodes", self._graph.number_of_nodes())
            else:
                logger.warning("Knowledge graph file missing at %s — run ingestion.", path)
        return self._graph

    def query_graph(self, entity: str, max_hops: int = 2) -> list[RetrievedChunk]:
        """Traverse the KG around every known entity mentioned in ``entity``.

        Accepting the whole user query (not just a single noun) is deliberate:
        retrieval must work end-to-end without a separate NER stage.  We scan
        all graph node names for case-insensitive substring matches, then
        expand each hit's neighbourhood up to ``max_hops`` in both directions —
        mirroring how an SRE mentally walks "Server_A → its apps → their DBs".
        Edges are rendered as subject-predicate-object triples so the LLM sees
        machine-checkable structure rather than prose.
        """
        graph = self._load_graph()
        if graph is None:
            return []
        haystack = entity.lower().replace("-", "_")
        matches = [n for n in graph.nodes if n.lower() in haystack]
        if not matches:
            return []
        chunks: list[RetrievedChunk] = []
        seen_edges: set[tuple[str, str, str]] = set()
        frontier = set(matches)
        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node in frontier:
                for _, target, key, data in graph.out_edges(node, keys=True, data=True):
                    triple = (node, data.get("relation", key), target)
                    if triple not in seen_edges:
                        seen_edges.add(triple)
                        chunks.append(
                            RetrievedChunk(
                                source=f"kg:{triple[0]}->{triple[2]}",
                                text=f"{triple[0]} {triple[1]} {triple[2]}"
                                + (f" ({data['detail']})" if "detail" in data else ""),
                            )
                        )
                    next_frontier.add(target)
                for source, _, key, data in graph.in_edges(node, keys=True, data=True):
                    triple = (source, data.get("relation", key), node)
                    if triple not in seen_edges:
                        seen_edges.add(triple)
                        chunks.append(
                            RetrievedChunk(
                                source=f"kg:{triple[0]}->{triple[2]}",
                                text=f"{triple[0]} {triple[1]} {triple[2]}",
                            )
                        )
                    next_frontier.add(source)
            frontier = next_frontier - set(matches) - {
                n for c in chunks for n in c.source.removeprefix("kg:").split("->")
            }
            if not frontier:
                break
        return chunks

    # ------------------------------------------------------------------
    # Vector side
    # ------------------------------------------------------------------
    def query_vector(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """Semantic search over ingested documents; empty-safe."""
        if self._collection.count() == 0:
            return []
        res = self._collection.query(query_texts=[query], n_results=top_k)
        docs = res["documents"][0] if res["documents"] else []
        ids = res["ids"][0] if res["ids"] else []
        dists = (res["distances"][0] if res["distances"] else [None] * len(docs))
        return [
            RetrievedChunk(source=f"chroma:{doc_id}", text=doc, score=dist)
            for doc_id, doc, dist in zip(ids, docs, dists)
        ]

    # ------------------------------------------------------------------
    # Fusion
    # ------------------------------------------------------------------
    def retrieve(self, query: str, top_k: int = 3) -> HybridContext:
        """Run both backends and fuse into one ordered context object.

        Ordering policy: graph facts first (they are exact and cheap to verify),
        then vector passages (rich but fuzzy).  The Evaluator relies on this
        split when checking grounding.
        """
        context = HybridContext()
        try:
            kg_chunks = self.query_graph(query)
        except Exception:  # pragma: no cover - defensive: memory must not crash agents
            logger.exception("KG retrieval failed")
            context.notes.append("knowledge-graph backend unavailable")
            kg_chunks = []
        try:
            vec_chunks = self.query_vector(query, top_k=top_k)
        except Exception:  # pragma: no cover
            logger.exception("Vector retrieval failed")
            context.notes.append("vector backend unavailable")
            vec_chunks = []
        context.chunks = kg_chunks + vec_chunks
        if not kg_chunks:
            context.notes.append("no knowledge-graph entities matched the query")
        return context


def _env_truthy(name: str) -> bool:
    import os

    return os.getenv(name, "").strip().lower() in {"1", "true", "yes"}


_store_singleton: HybridContextStore | None = None


def get_context_store(settings: Settings | None = None) -> HybridContextStore:
    """Return the process-wide :class:`HybridContextStore` (lazy singleton).

    Chroma's PersistentClient is heavyweight to construct (file locks), so we
    build it once per process and share it across all graph nodes/workers.
    """
    global _store_singleton
    if _store_singleton is None or settings is not None:
        _store_singleton = HybridContextStore(settings)
    return _store_singleton
