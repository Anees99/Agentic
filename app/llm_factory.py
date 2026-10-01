"""LLM factory and dynamic model-routing layer.

Architectural rationale
-----------------------
The single most important cost/latency lever in a multi-agent system is *not
letting every agent talk to the most expensive model*.  This module implements
that as a centralised **routing table**: agents never name a model directly,
they declare an intent (:class:`TaskType`) and the factory resolves it to a
concrete Qwen deployment with role-appropriate sampling parameters.

Why this shape:

* **Open/Closed for routing changes.**  Swapping ``qwen-turbo`` for a newer
  fast tier (or an A/B variant) is a one-line change to the table — or a pure
  environment override via :mod:`app.config` — with zero edits inside agents.
* **Provider isolation.**  All DashScope-specific knowledge lives here.  The
  rest of the codebase only sees the small :class:`ChatResult` contract, so
  migrating providers later touches exactly one file.
* **Offline determinism.**  When no API key is configured (CI, demos), the
  factory transparently returns a rule-based stub that speaks the same
  protocol.  Agents are therefore testable without network access, which is
  what makes the evaluation pipeline reproducible.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any

from app.config import Settings, get_settings

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    """Intent categories used by the router table.

    Members are string-valued so they serialise cleanly into logs, LangGraph
    state, and telemetry without custom encoders.
    """

    ROUTING = "routing"      # classification / intent analysis  -> qwen-turbo
    CODING = "coding"        # code generation                   -> qwen2.5-coder
    EVALUATION = "evaluation"  # judging / safety critique        -> qwen-max
    RETRIEVAL = "retrieval"  # query rewriting (cheap)           -> qwen-turbo


@dataclass(frozen=True)
class ChatResult:
    """Provider-agnostic response envelope.

    Keeping a tiny immutable dataclass instead of raw SDK dicts means every
    downstream consumer (agents, evaluator, API) has one stable contract even
    if DashScope's response schema evolves.
    """

    content: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0

    def json(self) -> dict[str, Any]:
        """Best-effort extraction of a JSON object from the model output.

        LLMs frequently wrap JSON in markdown fences or add prose; we scan for
        the outermost braces rather than trusting ``json.loads(content)``.
        Raises ``ValueError`` if nothing parseable is found so callers can
        decide their own retry/fallback policy.
        """
        match = re.search(r"\{.*\}", self.content, re.DOTALL)
        if not match:
            raise ValueError(f"No JSON object found in model output: {self.content!r}")
        return json.loads(match.group(0))


# ---------------------------------------------------------------------------
# Live client (DashScope) and offline stub share one interface
# ---------------------------------------------------------------------------


class DashScopeClient:
    """Thin, synchronous wrapper around the DashScope Generation SDK.

    Design notes:

    * One shared SDK client per process (created lazily) — connection pooling
      is handled by the SDK; recreating it per call would add latency.
    * ``tenacity`` retries cover transient 429/5xx blips, which are common
      enough under bursty agent traffic that a single failure should not fail
      an entire graph run.
    * Errors beyond the retry budget are raised as :class:`LLMCallError`, a
      narrow exception the API layer maps to HTTP 502.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        try:
            import dashscope  # imported lazily: keeps mock mode dependency-free
            from dashscope import Generation

            dashscope.api_key = settings.dashscope_api_key
            self._generation = Generation
        except ImportError as exc:  # pragma: no cover - environment guard
            raise RuntimeError("dashscope package is required for live LLM calls") from exc

        from tenacity import (
            retry,
            retry_if_exception_type,
            stop_after_attempt,
            wait_exponential,
        )

        # Bound the blast radius: max 3 attempts, exponential backoff 1–8 s.
        self._call_with_retry = retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=1, min=1, max=8),
            retry=retry_if_exception_type((ConnectionError, TimeoutError)),
            reraise=True,
        )(self._raw_call)

    def _raw_call(
        self, model: str, system_prompt: str, user_prompt: str, temperature: float
    ) -> ChatResult:
        response = self._generation.call(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=self._settings.max_tokens,
            result_format="message",
        )
        status = getattr(response, "status_code", 200)
        if status != 200:
            code = getattr(response, "code", "unknown")
            message = getattr(response, "message", str(response))
            raise LLMCallError(f"DashScope error {status} ({code}): {message}")

        usage = getattr(response, "usage", None) or {}
        content = response.output.choices[0].message.content
        return ChatResult(
            content=content,
            model=model,
            input_tokens=int(usage.get("input_tokens", 0) or 0),
            output_tokens=int(usage.get("output_tokens", 0) or 0),
        )

    def chat(
        self, model: str, system_prompt: str, user_prompt: str, temperature: float
    ) -> ChatResult:
        logger.debug("LLM call → model=%s temp=%.2f", model, temperature)
        return self._call_with_retry(model, system_prompt, user_prompt, temperature)


class MockChatClient:
    """Deterministic offline stand-in implementing the same ``chat`` contract.

    It does not fake intelligence — it fakes *protocol*.  Responses are simple
    rule-based strings good enough to drive the full LangGraph topology and
    the evaluation pipeline end-to-end without network access.  Every response
    is tagged with the requested model so tests can assert on routing.
    """

    def chat(
        self, model: str, system_prompt: str, user_prompt: str, temperature: float
    ) -> ChatResult:
        lowered = user_prompt.lower()
        if "classify" in system_prompt.lower():
            needs_code = any(
                kw in lowered for kw in ("write", "script", "code", "generate", "implement")
            )
            decision = "code_generation" if needs_code else "retrieval_only"
            content = json.dumps(
                {"decision": decision, "reasoning": "mock classifier"}
            )
        elif "judge" in system_prompt.lower() or "evaluat" in system_prompt.lower():
            content = json.dumps(
                {
                    "score": 4,
                    "verdict": "pass",
                    "security_risks": [],
                    "hallucinations": [],
                    "logic_errors": [],
                    "feedback": "Mock evaluation: output appears coherent.",
                }
            )
        elif "```python" in system_prompt.lower() or "python script" in system_prompt.lower():
            content = (
                "```python\n"
                "# Mock-generated script (offline mode)\n"
                "def main() -> None:\n"
                '    print("hello from mock coder")\n'
                "\nif __name__ == '__main__':\n"
                "    main()\n"
                "```"
            )
        else:
            content = "[mock response] " + user_prompt[:120]
        return ChatResult(content=content, model=model)


class LLMCallError(RuntimeError):
    """Raised when a live LLM call fails after the retry budget is exhausted."""


# ---------------------------------------------------------------------------
# The factory
# ---------------------------------------------------------------------------


class LLMRouter:
    """Centralised model router: maps :class:`TaskType` → client + model + temperature.

    Usage::

        router = get_llm_router()
        result = router.invoke(TaskType.CODING, system_prompt, user_prompt)

    The router is intentionally stateless apart from its injected client, so
    it is safe to share across LangGraph nodes and FastAPI workers.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        # Routing table — the heart of the cost/quality trade-off.
        self._table: dict[TaskType, tuple[str, float]] = {
            TaskType.ROUTING: (self._settings.router_model, self._settings.router_temperature),
            TaskType.RETRIEVAL: (self._settings.router_model, self._settings.router_temperature),
            TaskType.CODING: (self._settings.coder_model, self._settings.coder_temperature),
            TaskType.EVALUATION: (
                self._settings.evaluator_model,
                self._settings.evaluator_temperature,
            ),
        }
        use_mock = self._settings.mock_llm or not self._settings.has_live_credentials
        self._client: DashScopeClient | MockChatClient = (
            MockChatClient() if use_mock else DashScopeClient(self._settings)
        )
        if use_mock:
            logger.warning(
                "DASHSCOPE_API_KEY not configured (or MOCK_LLM=true): "
                "running with deterministic offline stub LLM."
            )

    @property
    def is_mock(self) -> bool:
        """Expose mock-mode for health endpoints / tests."""
        return isinstance(self._client, MockChatClient)

    def model_for(self, task: TaskType) -> str:
        """Return the model ID currently serving a task type (for observability)."""
        return self._table[task][0]

    def invoke(
        self, task: TaskType, system_prompt: str, user_prompt: str
    ) -> ChatResult:
        """Route a prompt pair to the correct Qwen tier and return the result."""
        model, temperature = self._table[task]
        return self._client.chat(model, system_prompt, user_prompt, temperature)


_router_singleton: LLMRouter | None = None


def get_llm_router(settings: Settings | None = None) -> LLMRouter:
    """Process-wide accessor for the :class:`LLMRouter` singleton.

    Lazily constructed so importing the package never touches the network;
    pass ``settings`` explicitly in tests to inject overrides.
    """
    global _router_singleton
    if _router_singleton is None or settings is not None:
        _router_singleton = LLMRouter(settings)
    return _router_singleton
