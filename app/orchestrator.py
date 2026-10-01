"""LangGraph orchestration: topology, conditional edges, and the compiled app.

Architectural rationale
-----------------------
This module owns *control flow only*.  Agents (app/agents.py) own reasoning;
the graph owns sequencing.  The separation means routing changes — e.g. adding
a human-in-the-loop interrupt before code execution — never touch agent code.

Topology (as required by the spec):

    START → router ─┬─ "retrieval_only"   → retriever ────────────→ synthesize → END
                    ├─ "code_generation"  → retriever → coder → evaluator → synthesize → END
                    └─ "direct_answer"    → synthesize → END

Two non-obvious decisions:

* **Code path always retrieves first.**  Generating infrastructure code without
  grounding is where hallucinated hostnames are born; retrieval is cheap
  (local Chroma + in-memory graph) so making it unconditional costs nothing
  and buys verifiable provenance for the Evaluator.
* **The judge gates synthesis, not loops.**  We deliberately do NOT wire an
  evaluator→coder retry cycle in v1: unbounded self-reflection loops are the
  classic way multi-agent systems blow their latency/cost budgets.  The verdict
  rides along in state so a future revision loop can be added behind a
  max-iterations counter without changing this topology's contract.

The graph is compiled once at import time into ``orchestrator_app`` — LangGraph
compilation is expensive and the resulting object is thread-safe for ``invoke``.
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, START, StateGraph

from langchain_core.messages import BaseMessage

from app.agents import (
    coder_node,
    evaluator_node,
    retriever_node,
    router_node,
    synthesize_node,
)
from app.state import AgentState, RouterDecision, initial_state

logger = logging.getLogger(__name__)


def route_after_router(state: AgentState) -> str:
    """Conditional-edge function: map the Router's decision onto a target node.

    Must return exactly one of the keys declared in the ``path_map`` below.
    Unknown/empty decisions degrade to ``synthesizer`` with a notice rather
    than raising — availability over purity at the edge level, since the
    Router node already applies its own fail-safe.
    """
    decision = state.get("router_decision", "")
    if decision == RouterDecision.RETRIEVAL_ONLY.value:
        return "retriever"
    if decision == RouterDecision.CODE_GENERATION.value:
        return "retriever_then_coder"
    if decision == RouterDecision.DIRECT_ANSWER.value:
        return "synthesize"
    logger.warning("Unrecognised router_decision=%r; short-circuiting to synthesizer", decision)
    return "synthesize"


def build_graph() -> StateGraph:
    """Assemble the uncompiled StateGraph (exposed separately for testing)."""
    graph = StateGraph(AgentState)

    # Nodes — thin wrappers keep signatures uniform for LangGraph.
    graph.add_node("router", router_node)
    graph.add_node("retriever", lambda state: retriever_node(state))
    graph.add_node("coder", coder_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("synthesizer", synthesize_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges(
        "router",
        route_after_router,
        {
            # Question answering: memory only.
            "retriever": "retriever",
            # Code generation: memory → code → judge (spec-mandated chain).
            "retriever_then_coder": "retriever",
            # Trivial/no-tool path.
            "synthesize": "synthesizer",
        },
    )
    # After retrieval the graph branches on whether code was requested.
    graph.add_conditional_edges(
        "retriever",
        lambda state: (
            "coder"
            if state.get("router_decision") == RouterDecision.CODE_GENERATION.value
            else "synthesizer"
        ),
        {"coder": "coder", "synthesizer": "synthesizer"},
    )
    graph.add_edge("coder", "evaluator")
    graph.add_edge("evaluator", "synthesizer")
    graph.add_edge("synthesizer", END)
    return graph


#: Compiled singleton — import this, don't rebuild per request.
orchestrator_app = build_graph().compile()


def _serialise_message(message: Any) -> dict[str, str]:
    """Normalise one trace entry from either message representation.

    ``add_messages`` may hand back LangChain ``BaseMessage`` objects *or* raw
    dicts depending on coercion; the API contract must be stable regardless,
    so we flatten both into ``{agent, role, content}`` here — the single place
    framework types leak toward the transport layer.
    """
    if isinstance(message, BaseMessage):
        return {
            "agent": str(message.name or message.type),
            "role": message.type,
            "content": str(message.content),
        }
    if isinstance(message, dict):
        return {
            "agent": str(message.get("agent", message.get("role", "?"))),
            "role": str(message.get("role", "?")),
            "content": str(message.get("content", "")),
        }
    return {"agent": "?", "role": "?", "content": str(message)}


def run_orchestrator(user_query: str) -> dict[str, Any]:
    """Execute one full pipeline run and return a flattened result payload.

    This is the single entry point shared by FastAPI and the evaluation
    harness, guaranteeing both exercise *identical* graph behaviour (no
    drift between "tested" and "served" paths — a classic prod incident).
    """
    final_state: AgentState = orchestrator_app.invoke(initial_state(user_query))
    evaluation = final_state.get("evaluation_result") or {}
    return {
        "query": user_query,
        "router_decision": final_state.get("router_decision", ""),
        "retrieved_context": final_state.get("retrieved_context", ""),
        "generated_code": final_state.get("generated_code", ""),
        "evaluation_result": evaluation,
        "eval_score": evaluation.get("score"),
        "final_output": final_state.get("final_output", ""),
        "trace": [_serialise_message(m) for m in final_state.get("messages", [])],
    }
