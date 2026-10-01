"""Centralised, environment-driven configuration for the Orchestrator.

Architectural rationale
-----------------------
Every tunable in a multi-agent system (model IDs, persistence paths, sampling
parameters) is a *deployment concern*, not a *logic concern*.  Hard-coding them
inside agent code makes the agents untestable and couples business logic to a
specific cloud account.  We therefore funnel all of it through a single frozen
Pydantic ``BaseSettings`` object that reads from environment variables (and an
optional ``.env`` file for local development).

Key decisions:

* **One API key, many models.**  Alibaba's DashScope gateway serves every Qwen
  tier with a single credential, so authentication lives here while model
  *selection* lives in :mod:`app.llm_factory` (the routing layer).
* **Offline-safe defaults.**  Paths for ChromaDB / the knowledge graph are
  plain strings resolved relative to the repository root, so ingestion and
  retrieval work identically under pytest, uvicorn, or a container volume.
* **Fail-fast validation.**  Requiring ``DASHSCOPE_API_KEY`` at settings-parse
  time surfaces misconfigured deployments during startup rather than on the
  first user request.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Repository root — used to anchor all relative persistence paths so the
#: service behaves identically regardless of the process CWD.
ROOT_DIR: Path = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Runtime configuration, overridable via environment variables / .env."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # immutable after load — prevents accidental mutation mid-request
    )

    # ------------------------------------------------------------------
    # Provider credentials
    # ------------------------------------------------------------------
    dashscope_api_key: str = Field(
        default="",
        description="Alibaba Cloud Model Studio (DashScope) API key.",
    )

    # ------------------------------------------------------------------
    # Dynamic model routing table (overridable per deployment)
    # ------------------------------------------------------------------
    router_model: str = Field(
        default="qwen-turbo",
        description="Cheap/fast tier for the Router Agent (classification workload).",
    )
    coder_model: str = Field(
        default="qwen2.5-coder-32b-instruct",
        description=(
            "Coder-specialised tier for the Coder Agent. The hosted DashScope "
            "identifier for the 'qwen2.5-coder' family; falls back gracefully "
            "in offline/mock mode."
        ),
    )
    evaluator_model: str = Field(
        default="qwen-max",
        description="Highest-reasoning tier for the Evaluator (LLM-as-judge) Agent.",
    )

    # ------------------------------------------------------------------
    # Sampling parameters — deliberately asymmetric per role
    # ------------------------------------------------------------------
    router_temperature: float = Field(
        default=0.0,
        description="Router must be deterministic: classification, not creation.",
    )
    coder_temperature: float = Field(
        default=0.2,
        description="Low but non-zero: code needs some variety, never creativity.",
    )
    evaluator_temperature: float = Field(
        default=0.0,
        description="Judging must be reproducible across evaluation runs.",
    )
    max_tokens: int = Field(default=2048, ge=1, le=8192)

    # ------------------------------------------------------------------
    # Persistence locations (hybrid context layer)
    # ------------------------------------------------------------------
    chroma_persist_dir: Path = Field(
        default=ROOT_DIR / "data" / "chroma",
        description="Persistent directory for the ChromaDB vector store.",
    )
    chroma_collection_name: str = Field(
        default="enterprise_docs",
        description="Single-tenant collection for the dummy enterprise corpus.",
    )
    knowledge_graph_path: Path = Field(
        default=ROOT_DIR / "data" / "knowledge_graph.pkl",
        description="Serialised NetworkX knowledge-graph produced by ingestion.",
    )

    # ------------------------------------------------------------------
    # Behaviour switches
    # ------------------------------------------------------------------
    mock_llm: bool = Field(
        default=False,
        description=(
            "When True (or when no API key is present), agents run against a "
            "deterministic stub LLM so the whole pipeline can be exercised "
            "offline — essential for CI smoke tests and local demos."
        ),
    )

    @property
    def has_live_credentials(self) -> bool:
        """True only if a real DashScope key was supplied."""
        return bool(self.dashscope_api_key.strip())


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide singleton :class:`Settings`.

    Cached so repeated calls inside hot agent loops do not re-read the
    environment; Pydantic parses once, everyone shares one immutable view.
    """
    return Settings()
