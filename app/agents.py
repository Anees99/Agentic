"""The four agents of the orchestrator, implemented as pure LangGraph node functions.

Architectural rationale
-----------------------
Each agent here is a *plain function* ``(state) -> state_delta`` rather than a
conversational "assistant object".  Three reasons:

1. **Composability** — LangGraph nodes that only read/write declared state keys
   can be reordered, retried, or checkpointed without touching agent internals.
2. **Testability** — every node takes its collaborators (LLM router, context
   store) via dependency injection, so unit tests substitute the offline stub.
3. **Auditability** — each node appends an explicit trace message to
   ``state["messages"]``; the full reasoning chain ships in the API response
   for compliance review.

Prompt-engineering decisions worth noting:

* The Router is forced into strict JSON with a closed enum — classification
  prompts must never free-form, or downstream conditional edges break.
* The Coder receives *labelled provenance* context blocks and is instructed to
  cite them; this gives the Evaluator something concrete to ground-check.
* The Evaluator's rubric is decomposed into three independent axes (security /
  hallucination / logic) before the single 1–5 score, because composite
  scores emitted without decomposition correlate poorly with human judges.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage

from app.context_store import HybridContextStore, get_context_store
from app.llm_factory import LLMCallError, LLMRouter, TaskType, get_llm_router
from app.state import AgentState, RouterDecision

logger = logging.getLogger(__name__)


def _msg_content(messages: list[Any], index: int) -> str:
    """Safely read ``messages[index].content`` across message representations.

    LangGraph's ``add_messages`` reducer coerces plain dicts into LangChain
    ``BaseMessage`` objects, so downstream nodes must handle both shapes plus
    negative indices on possibly-empty lists — defensive access at a trust
    boundary between the framework and our state contract.
    """
    if not messages:
        return ""
    message = messages[index]
    if isinstance(message, BaseMessage):
        return str(message.content)
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(message)

# ---------------------------------------------------------------------------
# Prompt templates (module-level constants: easy to version / A-B test)
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are the Router Agent of an enterprise knowledge & code \
orchestrator. Classify the user's request into exactly one decision.

Rules:
- "retrieval_only": the user asks a factual/operational question answerable from \
internal documentation or the infrastructure knowledge graph. No code is requested.
- "code_generation": the user explicitly wants a Python script/function written, \
generated, or implemented (context retrieval still happens first).
- "direct_answer": a trivial greeting/meta question needing no tools.

Respond with ONLY this JSON object, no prose, no markdown fences:
{"decision": "<retrieval_only|code_generation|direct_answer>", "reasoning": "<one sentence>"}
"""

CODER_SYSTEM_PROMPT = """You are the Coder Agent. Write ONE complete, runnable Python \
script that satisfies the user's request, grounded strictly in the provided CONTEXT.

Requirements:
- Use standard library or widely-available packages only; no network side effects.
- NEVER hard-code secrets, credentials, hostnames, or IPs; accept them via \
environment variables or function arguments.
- Include type hints, a module docstring, and a `main()` guard.
- Prefer defensive coding: validate inputs, handle errors explicitly.
- If CONTEXT contains facts (lines starting with [kg:...] or [chroma:...]), treat \
them as authoritative; do not invent contradicting details.

Output the script inside a single ```python fenced block. No other commentary.
"""

EVALUATOR_SYSTEM_PROMPT = """You are the Evaluator Agent (LLM-as-a-Judge). Critically review \
the Coder Agent's output before it reaches the user. Judge along three independent axes:

1. security_risks: dangerous calls (os.system, eval, exec, shell=True, hardcoded \
secrets, unvalidated input, path traversal).
2. hallucinations: claims/entities inconsistent with the provided CONTEXT.
3. logic_errors: bugs, wrong assumptions, non-runnable code.

Then produce a holistic integer score 1-5 (5 = production-ready, 1 = dangerous/nonsense) \
and verdict "pass" iff score >= 3 AND security_risks is empty.

Respond with ONLY this JSON object, no markdown fences:
{"score": <int 1-5>, "verdict": "<pass|revise|reject>", "security_risks": [...], \
"hallucinations": [...], "logic_errors": [...], "feedback": "<two sentences max>"}
"""

RETRIEVER_SUMMARY_SYSTEM_PROMPT = """You are the Retriever Agent's summariser. Condense the \
raw hybrid-retrieval context into at most 8 bullet points that preserve every entity name, \
relationship, and numeric detail exactly as given. Output bullets only."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_python_block(text: str) -> str:
    """Pull the first ```python fence out of a coder response.

    Falls back to the raw text: some models omit fences under pressure, and a
    slightly noisy script is more useful downstream than raising.
    """
    match = re.search(r"```(?:python)?\s*\n(.*?)```", text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()


def _safe_json(content: str) -> dict[str, Any]:
    """Parse model JSON tolerating fences/prose; returns {} on total failure.

    Nodes convert parse failures into *state*, not exceptions: a router that
    emits garbage should degrade to a safe default path, not 500 the request.
    """
    try:
        match = re.search(r"\{.*\}", content, re.DOTALL)
        return json.loads(match.group(0)) if match else {}
    except (json.JSONDecodeError, TypeError):
        logger.warning("Model returned non-JSON payload: %.120r", content)
        return {}


# ---------------------------------------------------------------------------
# Node 1 — Router
# ---------------------------------------------------------------------------

def router_node(state: AgentState) -> dict[str, Any]:
    """Classify intent and set the routing flag consumed by the conditional edge.

    Failure policy: if qwen-turbo returns malformed JSON we default to
    ``retrieval_only`` — the cheapest, safest path (it cannot execute anything
    and still produces a grounded answer).
    """
    llm = get_llm_router()
    query = _msg_content(state["messages"], -1)
    try:
        result = llm.invoke(TaskType.ROUTING, ROUTER_SYSTEM_PROMPT, f"User request: {query}")
        payload = _safe_json(result.content)
        raw_decision = payload.get("decision", "")
    except LLMCallError:
        logger.exception("Router LLM unavailable; defaulting to retrieval_only")
        raw_decision, reasoning = "retrieval_only", "router LLM error → safe fallback"
    else:
        reasoning = payload.get("reasoning", "")

    try:
        decision = RouterDecision(raw_decision)
    except ValueError:
        decision = RouterDecision.RETRIEVAL_ONLY

    return {
        "router_decision": decision.value,
        "messages": [
            {"role": "assistant", "agent": "router",
             "content": f"decision={decision.value} reason={reasoning}"}
        ],
    }


# ---------------------------------------------------------------------------
# Node 2 — Retriever
# ---------------------------------------------------------------------------

def retriever_node(
    state: AgentState,
    store: HybridContextStore | None = None,
) -> dict[str, Any]:
    """Fuse ChromaDB + KG results into a single labelled context string.

    Retrieval is deliberately run for BOTH paths (question answering and code
    generation): grounding code in real infrastructure facts is precisely how
    we keep the Coder from hallucinating hostnames.  The optional LLM summary
    pass compresses long contexts to protect the Coder's attention budget;
    when the summariser fails we ship the raw fused block (quality degrades,
    availability does not).
    """
    store = store or get_context_store()
    query = _msg_content(state["messages"], 0)
    context = store.retrieve(query)

    raw_block = context.as_prompt_block()
    final_block = raw_block
    if len(raw_block) > 4000 and context.chunks:
        try:
            llm = get_llm_router()
            summary = llm.invoke(
                TaskType.RETRIEVAL,
                RETRIEVER_SUMMARY_SYSTEM_PROMPT,
                f"CONTEXT:\n{raw_block}",
            )
            final_block = summary.content
        except LLMCallError:
            logger.warning("Context summarisation failed; using raw block")

    header = "\n".join(f"(note: {n})" for n in context.notes)
    return {
        "retrieved_context": f"{final_block}\n{header}".strip(),
        "messages": [
            {"role": "assistant", "agent": "retriever",
             "content": f"fused {len(context.chunks)} chunks "
                        f"(kg+vector) into context"}
        ],
    }


# ---------------------------------------------------------------------------
# Node 3 — Coder
# ---------------------------------------------------------------------------

def coder_node(state: AgentState) -> dict[str, Any]:
    """Generate one grounded Python script from query + retrieved context."""
    llm = get_llm_router()
    query = _msg_content(state["messages"], 0)
    user_prompt = (
        f"USER REQUEST:\n{query}\n\n"
        f"AUTHORITATIVE CONTEXT (hybrid retrieval; kg:=knowledge graph, "
        f"chroma:=documents):\n{state['retrieved_context'] or '(none)'}"
    )
    result = llm.invoke(TaskType.CODING, CODER_SYSTEM_PROMPT, user_prompt)
    code = _extract_python_block(result.content)
    return {
        "generated_code": code,
        "messages": [
            {"role": "assistant", "agent": "coder",
             "content": f"generated {len(code.splitlines())} lines via "
                        f"{llm.model_for(TaskType.CODING)}"}
        ],
    }


# ---------------------------------------------------------------------------
# Node 4 — Evaluator (LLM-as-Judge)
# ---------------------------------------------------------------------------

def evaluator_node(state: AgentState) -> dict[str, Any]:
    """Judge the generated code against security/hallucination/logic axes.

    Parsing failure is treated as ``score=1, verdict=reject``: an *unjudgeable*
    artifact must never silently reach the user — failing closed is the correct
    safety posture for a gate component.
    """
    llm = get_llm_router()
    query = _msg_content(state["messages"], 0)
    user_prompt = (
        f"ORIGINAL USER REQUEST:\n{query}\n\n"
        f"CONTEXT THE CODE WAS GROUNDED IN:\n{state['retrieved_context'] or '(none)'}\n\n"
        f"CANDIDATE CODE:\n```python\n{state['generated_code']}\n```"
    )
    result = llm.invoke(TaskType.EVALUATION, EVALUATOR_SYSTEM_PROMPT, user_prompt)
    payload = _safe_json(result.content)

    score = payload.get("score")
    if not isinstance(score, int) or not 1 <= score <= 5:
        logger.warning("Evaluator produced unusable score; failing closed")
        payload.setdefault("score", 1)
        payload.setdefault("verdict", "reject")
        payload.setdefault("feedback", "evaluation output unparseable — rejected defensively")
    payload.setdefault("model", result.model)

    return {
        "evaluation_result": payload,
        "messages": [
            {"role": "assistant", "agent": "evaluator",
             "content": f"score={payload.get('score')} verdict={payload.get('verdict')}"}
        ],
    }


# ---------------------------------------------------------------------------
# Node 5 — Synthesiser (terminal formatting, no LLM call)
# ---------------------------------------------------------------------------

def synthesize_node(state: AgentState) -> dict[str, Any]:
    """Assemble ``final_output`` deterministically from whatever state exists.

    Kept LLM-free on purpose: the terminal step of a pipeline should be a pure
    formatter so the response contract never depends on model behaviour.
    """
    verdict = (state.get("evaluation_result") or {}).get("verdict", "n/a")
    score = (state.get("evaluation_result") or {}).get("score", "n/a")
    parts: list[str] = []
    if state.get("generated_code"):
        parts.append(
            f"## Generated code (judge verdict: {verdict}, score: {score}/5)\n"
            f"```python\n{state['generated_code']}\n```"
        )
    elif state.get("retrieved_context"):
        parts.append(
            "## Grounded answer context (retrieval-only path)\n" + state["retrieved_context"]
        )
    else:
        parts.append("No context retrieved; please rephrase the request.")
    return {"final_output": "\n\n".join(parts)}
