"""Enterprise Agentic Knowledge & Code Orchestrator.

A production-grade reference implementation of a multi-agent system combining:

* **Dynamic model routing** across Alibaba Qwen tiers (qwen-turbo / qwen2.5-coder / qwen-max)
* **Hybrid retrieval** — ChromaDB vector store + NetworkX knowledge graph
* **LangGraph orchestration** — Router → Retriever → Coder → Evaluator topology
* **LLM-as-a-Judge evaluation** and a **FastAPI** serving layer

Module map::

    app/config.py         env-driven settings (single source of tunables)
    app/state.py          TypedDict state schema shared by all nodes
    app/llm_factory.py    model-routing table + DashScope client (+ offline stub)
    app/context_store.py  hybrid ChromaDB + KG read facade
    app/agents.py         the four agent node functions + prompts
    app/orchestrator.py   StateGraph wiring and compiled app
    app/evaluator.py      run_eval_pipeline() regression harness
    app/main.py           FastAPI endpoints (/run-agent, /evaluate, /health)
"""

__version__ = "1.0.0"
