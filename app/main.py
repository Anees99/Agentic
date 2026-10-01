"""FastAPI production wrapper around the compiled LangGraph orchestrator.

Architectural rationale
-----------------------
The HTTP layer is intentionally *thin*: validation, mapping, and error
translation — no business logic.  Everything substantive lives in the graph so
the API surface and the eval harness exercise identical code paths.

Operational decisions:

* **Sync graph runs are dispatched to a threadpool.**  LangGraph's ``invoke``
  is blocking (LLM HTTP calls dominate latency); FastAPI automatically runs
  ``def`` endpoints off the event loop, keeping it responsive for health
  checks and concurrent requests without async-rewriting the whole agent stack.
* **Explicit request/response models** give us free OpenAPI docs and reject
  malformed payloads at the edge (pydantic) before any tokens are spent.
* **Error taxonomy:** LLM provider failures → 502 with a machine-readable code;
  anything else → 500 with a logged exception id.  Never leak stack traces.
* **/health reports mock-mode** so operators can immediately tell whether
  they're looking at a real or stubbed pipeline.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.evaluator import run_eval_pipeline
from app.llm_factory import LLMCallError, TaskType, get_llm_router
from app.orchestrator import run_orchestrator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("orchestrator.api")

app = FastAPI(
    title="Enterprise Agentic Knowledge & Code Orchestrator",
    description=(
        "Multi-agent system (Router / Retriever / Coder / Evaluator) over "
        "hybrid ChromaDB + NetworkX memory with Qwen dynamic model routing."
    ),
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / response contracts
# ---------------------------------------------------------------------------


class AgentRequest(BaseModel):
    """Inbound payload for a single orchestrated request."""

    query: str = Field(
        min_length=3,
        max_length=4000,
        description="Natural-language user request (question or code task).",
        examples=["What applications run on Server_A?"],
    )
    include_trace: bool = Field(
        default=False,
        description="Return the per-agent reasoning trace (verbose; internal use).",
    )


class EvaluationSummary(BaseModel):
    """Condensed judge verdict surfaced alongside the answer."""

    score: int | None = Field(default=None, description="LLM-as-judge score 1-5.")
    verdict: str | None = None
    feedback: str | None = None
    security_risks: list[str] = Field(default_factory=list)


class AgentResponse(BaseModel):
    """Flattened pipeline outcome — the public contract of the service."""

    request_id: str
    query: str
    router_decision: str
    final_output: str
    generated_code: str | None = None
    evaluation: EvaluationSummary
    trace: list[dict[str, Any]] | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post("/run-agent", response_model=AgentResponse)
def run_agent(request: AgentRequest) -> AgentResponse:
    """Execute the full LangGraph pipeline for one user query.

    Flow: Router → (Retriever [→ Coder → Evaluator]) → Synthesizer.
    The endpoint returns the artifact *and* the judge's verdict so clients can
    implement their own gating (e.g. refuse to auto-execute non-passing code).
    """
    request_id = uuid.uuid4().hex[:12]
    logger.info("[%s] query=%.100s", request_id, request.query)
    try:
        outcome = run_orchestrator(request.query)
    except LLMCallError as exc:
        # Provider outage is upstream's fault: 502, retryable by the client.
        logger.exception("[%s] LLM provider failure", request_id)
        raise HTTPException(
            status_code=502,
            detail={"code": "llm_provider_error", "message": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover - safety net
        logger.exception("[%s] unexpected pipeline failure", request_id)
        raise HTTPException(status_code=500, detail={"code": "internal_error"}) from exc

    evaluation = outcome.get("evaluation_result") or {}
    return AgentResponse(
        request_id=request_id,
        query=outcome["query"],
        router_decision=outcome["router_decision"],
        final_output=outcome["final_output"],
        generated_code=outcome["generated_code"] or None,
        evaluation=EvaluationSummary(
            score=evaluation.get("score"),
            verdict=evaluation.get("verdict"),
            feedback=evaluation.get("feedback"),
            security_risks=evaluation.get("security_risks", []) or [],
        ),
        trace=outcome["trace"] if request.include_trace else None,
    )


@app.post("/evaluate")
def evaluate() -> dict[str, Any]:
    """Run the 3-query LLM-as-judge regression suite and return the report.

    Exposed over POST because it mutates nothing but is expensive (multiple
    live LLM calls) — caching GET semantics would invite crawlers to DoS us.
    """
    try:
        return run_eval_pipeline()
    except LLMCallError as exc:
        raise HTTPException(status_code=502, detail={"code": "llm_provider_error"}) from exc


@app.get("/health")
def health() -> dict[str, Any]:
    """Liveness + configuration probe (no LLM calls, safe for k8s probes)."""
    settings = get_settings()
    router = get_llm_router()
    return {
        "status": "ok",
        "mock_llm": router.is_mock,
        "model_routing": {
            "router": router.model_for(TaskType.ROUTING),
            "coder": router.model_for(TaskType.CODING),
            "evaluator": router.model_for(TaskType.EVALUATION),
        },
        "chroma_dir": str(settings.chroma_persist_dir),
        "knowledge_graph": str(settings.knowledge_graph_path),
    }


if __name__ == "__main__":  # pragma: no cover - dev convenience
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
