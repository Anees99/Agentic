"""Shared state schema and enums for the LangGraph orchestrator.

Architectural rationale
-----------------------
In LangGraph, the ``TypedDict`` state *is* the contract between agents: nodes
never call each other directly, they only exchange typed deltas through this
schema.  Making it explicit and small is what keeps a multi-agent graph
debuggable — you can dump state at any superstep and see the entire pipeline's
memory in one JSON object.

Reducer choices matter:

* ``messages`` uses ``add_messages`` (append semantics) so every agent's trace
  accumulates instead of overwriting — this is our audit log.
* All other keys use last-write-wins (default), because each represents a
  single artifact (one context block, one script, one verdict) produced by
  exactly one node per run.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, NotRequired, TypedDict

from langgraph.graph.message import add_messages


class RouterDecision(str, Enum):
    """Closed vocabulary emitted by the Router Agent.

    A string enum (not free text) lets the conditional edge compare against a
    finite set; anything unrecognisable maps to ``RETRIEVAL_ONLY`` upstream,
    which is the fail-safe branch.
    """

    RETRIEVAL_ONLY = "retrieval_only"        # answer from hybrid memory
    CODE_GENERATION = "code_generation"      # memory → code → judge
    DIRECT_ANSWER = "direct_answer"          # no tools needed


class Message(TypedDict, total=False):
    """Minimal chat-message shape used across nodes and the API layer.

    We avoid depending on langchain message classes inside state so the graph
    stays serialisable for checkpointing and trivially JSON-renderable for the
    FastAPI response.  (Note: LangGraph's ``add_messages`` reducer coerces
    these dicts into ``BaseMessage`` objects at runtime; consumers must handle
    both shapes — see ``app.agents._msg_content``.)
    """

    role: str          # "user" | "assistant"
    content: str
    agent: NotRequired[str]  # producer tag: router / retriever / coder / evaluator / api


class AgentState(TypedDict, total=False):
    """The single source of truth flowing through the StateGraph.

    Fields map 1:1 to the requirements spec:

    * ``messages``           — conversation + per-agent trace (append-only).
    * ``router_decision``    — the routing flag consumed by the conditional edge.
    * ``retrieved_context``  — fused ChromaDB + KG prompt block (Retriever output).
    * ``generated_code``     — Python script produced by the Coder.
    * ``evaluation_result``  — structured LLM-as-judge verdict (Evaluator output).
    * ``final_output``       — terminal, user-facing payload assembled by the graph.
    """

    messages: Annotated[list[Message], add_messages]
    router_decision: str
    retrieved_context: str
    generated_code: str
    evaluation_result: dict[str, Any]
    final_output: str


def initial_state(user_query: str) -> AgentState:
    """Build a well-formed empty state seeded with the user turn.

    Centralising construction here prevents subtle bugs where different entry
    points (API, eval harness, tests) seed state inconsistently.
    """
    return {
        "messages": [{"role": "user", "content": user_query, "agent": "api"}],
        "router_decision": "",
        "retrieved_context": "",
        "generated_code": "",
        "evaluation_result": {},
        "final_output": "",
    }
