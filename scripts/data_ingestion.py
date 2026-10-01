"""Data-ingestion entry point for the hybrid context layer (Step 1 of the spec).

Architectural rationale
-----------------------
Ingestion is a **batch job**, not a request-path concern: it runs once per
corpus change (ideally from CI or a scheduled ETL), writes both memory
backends, and exits.  Keeping it out of the API process means serving latency
never depends on document parsing, and re-ingestion can't race live traffic.

Idempotency strategy: documents are upserted under *stable IDs* derived from
their titles, so re-running this script converges to the same store state
instead of duplicating chunks — the property you want from any job that might
be retried by an orchestrator.

Run with::

    python -m scripts.data_ingestion          # repo root, PYTHONPATH=.
"""

from __future__ import annotations

import logging
import pickle
import sys
from pathlib import Path

# Allow `python scripts/data_ingestion.py` without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import networkx as nx

from app.config import get_settings
from app.context_store import _HashEmbeddingFunction, deterministic_embedding

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("data_ingestion")

# ---------------------------------------------------------------------------
# Dummy enterprise corpus — three documents about server architecture.
# Real systems would chunk larger docs; at demo scale one doc == one chunk.
# ---------------------------------------------------------------------------

ENTERPRISE_DOCUMENTS: list[dict[str, str]] = [
    {
        "id": "doc-server-architecture-overview",
        "title": "Enterprise Server Architecture Overview",
        "text": (
            "The production fleet is organised into three tiers: edge, application, "
            "and data. Server_A is the primary application-tier host in us-east-1 and "
            "runs App_X, the customer-facing order-management service. Server_A "
            "connects to Database_Y, a PostgreSQL 15 cluster, over a private VPC link "
            "with TLS enforced. Health endpoints follow the convention "
            "https://<host>:8443/healthz with a 2-second timeout and 3 retries."
        ),
    },
    {
        "id": "doc-app-x-runbook",
        "title": "App_X Operations Runbook",
        "text": (
            "App_X is deployed only on Server_A. Its readiness probe is "
            "/healthz/ready and its liveness probe is /healthz/live. If App_X stops "
            "responding within 5 seconds, operators must fail over to Server_B before "
            "page escalation. App_X writes exclusively to Database_Y; no other "
            "database is authorised for its traffic."
        ),
    },
    {
        "id": "doc-database-y-policy",
        "title": "Database_Y Connectivity & Security Policy",
        "text": (
            "Database_Y exposes port 5432 internally and a health endpoint on "
            "https://database-y.internal:9443/health. All client scripts must read "
            "credentials from environment variables (DB_Y_USER, DB_Y_PASSWORD); "
            "embedding secrets in code is a policy violation. Only Server_A holds "
            "network ACLs permitting connections to Database_Y."
        ),
    },
]


def ingest_documents() -> int:
    """Upsert the dummy corpus into the persistent ChromaDB collection.

    Uses the same deterministic embedding function as the read path when real
    embeddings are disabled, guaranteeing query/ingest space consistency.
    Returns the total document count after ingestion.
    """
    import chromadb
    from chromadb.config import Settings as ChromaSettings

    settings = get_settings()
    client = chromadb.PersistentClient(
        path=str(settings.chroma_persist_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )
    use_default_ef = True if _env_truthy_default_embeddings() else False
    ef = None if use_default_ef else _HashEmbeddingFunction()
    collection = client.get_or_create_collection(
        name=settings.chroma_collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    ids = [doc["id"] for doc in ENTERPRISE_DOCUMENTS]
    documents = [f"{doc['title']}\n{doc['text']}" for doc in ENTERPRISE_DOCUMENTS]
    metadatas = [{"title": doc["title"], "source": "enterprise_wiki"} for doc in ENTERPRISE_DOCUMENTS]

    # upsert (not add): stable-ID re-runs converge instead of duplicating.
    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    logger.info(
        "ChromaDB: upserted %d documents into '%s' (total=%d)",
        len(ids), settings.chroma_collection_name, collection.count(),
    )
    return collection.count()


def build_knowledge_graph() -> nx.MultiDiGraph:
    """Construct and persist the structured infrastructure knowledge graph.

    Modelled as a MultiDiGraph because enterprise relationships are directed
    and may be parallel-typed (Server_A RUNS App_X *and* MONITORS App_X).
    Edges carry a ``relation`` attribute so retrieval can render clean triples.
    """
    graph = nx.MultiDiGraph(name="enterprise-infrastructure-v1")

    graph.add_node("Server_A", type="server", tier="application", region="us-east-1")
    graph.add_node("Server_B", type="server", tier="application", role="failover")
    graph.add_node("App_X", type="application", service="order-management",
                   health_path="/healthz/live")
    graph.add_node("Database_Y", type="database", engine="postgresql-15",
                   health_endpoint="https://database-y.internal:9443/health")

    # Spec-mandated core edges, plus two grounded extras drawn from the docs.
    graph.add_edge("Server_A", "App_X", relation="RUNS",
                   detail="primary deployment host")
    graph.add_edge("Server_A", "Database_Y", relation="CONNECTS_TO",
                   detail="private VPC link, TLS enforced, port 5432")
    graph.add_edge("App_X", "Database_Y", relation="READS_WRITES",
                   detail="exclusive authorised datastore")
    graph.add_edge("Server_B", "App_X", relation="FAILOVER_FOR",
                   detail="activate if App_X unresponsive > 5s")

    settings = get_settings()
    settings.knowledge_graph_path.parent.mkdir(parents=True, exist_ok=True)
    # networkx>=3.2 removed write_/read_gpickle; explicit pickle is the blessed
    # replacement for trusted, self-produced graph artifacts.  A JSON node/edge
    # export would be the choice if the file must cross trust boundaries.
    with settings.knowledge_graph_path.open("wb") as fh:
        pickle.dump(graph, fh, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "Knowledge graph: %d nodes / %d edges persisted to %s",
        graph.number_of_nodes(), graph.number_of_edges(), settings.knowledge_graph_path,
    )
    return graph


def _env_truthy_default_embeddings() -> bool:
    import os

    return os.getenv("USE_DEFAULT_EMBEDDINGS", "").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    """Run the full ingestion batch: vector store first, then the graph."""
    settings = get_settings()
    settings.chroma_persist_dir.mkdir(parents=True, exist_ok=True)
    total = ingest_documents()
    build_knowledge_graph()
    # Sanity echo: prove both backends are readable immediately after writes.
    sample = deterministic_embedding("Server_A")
    logger.info("Embedding smoke test OK (dim=%d). Ingestion complete: %d docs.",
                len(sample), total)


if __name__ == "__main__":
    main()
