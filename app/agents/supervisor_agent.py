"""
SupervisorAgent — LangGraph StateGraph orchestrator.

Flow
----
  parse_query → plan_tasks → save_job → dispatch_pipeline → return_job_id

The graph runs fully async inside the FastAPI process.
It calls Gemini to produce a structured task plan, persists it to MongoDB,
then fires a Celery chain and returns the job_id immediately — the caller
does NOT wait for the pipeline to complete.

Celery chain per search query (parallel across queries):
    scanner_task → evaluator_task → extractor_task → processor_task

Usage
-----
    from app.agents.supervisor_agent import SupervisorAgent

    agent = SupervisorAgent()
    result = await agent.run(query="latest advances in RAG", session_id="sess-123")
    # result = { "job_id": "...", "plan": {...}, "task_count": 3 }
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TypedDict

from celery import chain
from langchain_core.messages import HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, StateGraph

from app.agents.evaluator_agent import evaluator_task
from app.agents.extractor_agent import extractor_task
from app.agents.processor_agent import processor_task
from app.agents.scanner_agent import scanner_task
from app.config import settings
from app.utils.job_manager import (
    broadcast_progress,
    create_agent_job,
    create_job,
)

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME = "SupervisorAgent"
PIPELINE_AGENTS = ["ScannerAgent", "EvaluatorAgent", "ExtractorAgent", "ProcessorAgent"]

# How many parallel chains to launch (one per search query)
MAX_PARALLEL_CHAINS = 5

# ── LangGraph state ────────────────────────────────────────────────

class SupervisorState(TypedDict):
    """Mutable state threaded through every LangGraph node."""
    # inputs
    query: str
    session_id: str
    # generated
    job_id: str
    plan: Dict[str, Any]          # Gemini's structured plan
    subtasks: List[str]           # human-readable subtask descriptions
    search_queries: List[str]     # queries to hand to the scanner
    expected_doc_count: int
    complexity: str               # "low" | "medium" | "high"
    # tracking
    status: str                   # "planning" | "dispatched" | "error"
    error: Optional[str]
    task_count: int               # number of Celery chains launched
    dispatched_at: Optional[str]


# ── Gemini client factory ──────────────────────────────────────────

def _get_gemini_model() -> ChatGoogleGenerativeAI:
    """Return a LangChain ChatGoogleGenerativeAI instance."""
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=settings.temperature,
        max_output_tokens=settings.max_tokens,
    )


# ── Node implementations ───────────────────────────────────────────

async def node_parse_query(state: SupervisorState) -> SupervisorState:
    """Validate the incoming query and generate a UUID job_id."""
    query = (state.get("query") or "").strip()
    if not query:
        return {**state, "status": "error", "error": "Empty query received"}
    return {
        **state,
        "job_id": str(uuid.uuid4()),
        "status": "planning",
        "error": None,
    }


async def node_plan_tasks(state: SupervisorState) -> SupervisorState:
    """
    Call Gemini to decompose the query into a structured task plan.

    Expected JSON response
    ----------------------
    {
        "subtasks": ["Find recent papers on RAG", "Look for benchmark results", ...],
        "search_queries": ["RAG retrieval augmented generation 2024", ...],
        "expected_doc_count": 8,
        "complexity": "medium"
    }
    """
    if state.get("status") == "error":
        return state

    prompt = _build_planning_prompt(state["query"])

    try:
        model = _get_gemini_model()
        response = await model.ainvoke([HumanMessage(content=prompt)])
        raw = response.content.strip()
        plan = _parse_json_response(raw)
        _validate_plan(plan)

        logger.info(
            f"📋 Plan generated | job_id={state['job_id']} "
            f"| queries={len(plan['search_queries'])} "
            f"| complexity={plan['complexity']}"
        )

        return {
            **state,
            "plan": plan,
            "subtasks": plan["subtasks"],
            "search_queries": plan["search_queries"][:MAX_PARALLEL_CHAINS],
            "expected_doc_count": plan.get("expected_doc_count", 5),
            "complexity": plan.get("complexity", "medium"),
            "status": "planned",
        }

    except Exception as exc:
        logger.error(f"Gemini planning failed: {exc}")
        # Graceful fallback — treat the whole query as one search
        fallback_plan = {
            "subtasks": [f"Search and process documents for: {state['query']}"],
            "search_queries": [state["query"]],
            "expected_doc_count": 5,
            "complexity": "low",
        }
        return {
            **state,
            "plan": fallback_plan,
            "subtasks": fallback_plan["subtasks"],
            "search_queries": fallback_plan["search_queries"],
            "expected_doc_count": 5,
            "complexity": "low",
            "status": "planned",
            "error": f"Planning degraded to fallback: {exc}",
        }


async def node_save_job(state: SupervisorState) -> SupervisorState:
    """
    Persist the parent supervisor job + per-agent sub-jobs to MongoDB
    before any Celery task is fired — so the frontend can immediately
    render pending progress bars.
    """
    if state.get("status") == "error":
        return state

    job_id = state["job_id"]
    session_id = state["session_id"]

    await create_job(
        job_id=job_id,
        session_id=session_id,
        query=state["query"],
        total_tasks=len(state["search_queries"]) * len(PIPELINE_AGENTS),
        agent_names=PIPELINE_AGENTS,
        input_data={
            "plan": state["plan"],
            "complexity": state["complexity"],
            "expected_doc_count": state["expected_doc_count"],
        },
    )

    # One sub-job record per agent so the frontend can track each bar
    for agent in PIPELINE_AGENTS:
        await create_agent_job(
            job_id=job_id,
            session_id=session_id,
            agent_name=agent,
            input_data={"search_queries": state["search_queries"]},
        )

    # Push initial "pending" state to any open WS connections
    await broadcast_progress(job_id)

    logger.info(f"💾 Job saved | job_id={job_id}")
    return {**state, "status": "saved"}


async def node_dispatch_pipeline(state: SupervisorState) -> SupervisorState:
    """
    Fire one Celery chain per search query in parallel (non-blocking).

    Chain per query:
        scanner_task → evaluator_task → extractor_task → processor_task

    Each task receives (job_id, session_id, <previous_result>) so workers
    can update MongoDB progress and broadcast over WebSocket.
    """
    if state.get("status") == "error":
        return state

    job_id = state["job_id"]
    session_id = state["session_id"]
    chains_launched = 0

    for sq in state["search_queries"]:
        try:
            # Build the Celery chain — each .s() is a Celery signature.
            # When chained, the return value of task N is passed as the
            # FIRST positional argument to task N+1 (after self if bind=True).
            pipeline = chain(
                scanner_task.s(job_id, session_id, state["query"], [sq]),
                evaluator_task.s(job_id, session_id),
                extractor_task.s(job_id, session_id),
                processor_task.s(job_id, session_id),
            )
            # apply_async is non-blocking — returns an AsyncResult immediately
            pipeline.apply_async()
            chains_launched += 1
            logger.info(f"🚀 Chain dispatched | job_id={job_id} | query='{sq}'")

        except Exception as exc:
            logger.error(f"Failed to dispatch chain for query '{sq}': {exc}")

    return {
        **state,
        "status": "dispatched",
        "task_count": chains_launched,
        "dispatched_at": datetime.now(timezone.utc).isoformat(),
    }


# ── Router ─────────────────────────────────────────────────────────

def _route_after_parse(state: SupervisorState) -> str:
    return "error_end" if state.get("status") == "error" else "plan_tasks"


def _route_after_plan(state: SupervisorState) -> str:
    return "error_end" if state.get("status") == "error" else "save_job"


def _route_after_save(state: SupervisorState) -> str:
    return "error_end" if state.get("status") == "error" else "dispatch_pipeline"


# ── Graph assembly ─────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    graph = StateGraph(SupervisorState)

    # Nodes
    graph.add_node("parse_query",       node_parse_query)
    graph.add_node("plan_tasks",        node_plan_tasks)
    graph.add_node("save_job",          node_save_job)
    graph.add_node("dispatch_pipeline", node_dispatch_pipeline)
    graph.add_node("error_end",         lambda s: {**s, "status": "error"})

    # Entry
    graph.set_entry_point("parse_query")

    # Conditional edges
    graph.add_conditional_edges("parse_query", _route_after_parse,
                                {"plan_tasks": "plan_tasks", "error_end": "error_end"})
    graph.add_conditional_edges("plan_tasks",  _route_after_plan,
                                {"save_job": "save_job", "error_end": "error_end"})
    graph.add_conditional_edges("save_job",    _route_after_save,
                                {"dispatch_pipeline": "dispatch_pipeline", "error_end": "error_end"})

    # Terminal edges
    graph.add_edge("dispatch_pipeline", END)
    graph.add_edge("error_end",         END)

    return graph.compile()


# ── Public class ───────────────────────────────────────────────────

class SupervisorAgent:
    """
    Async wrapper around the LangGraph supervisor graph.

    The compiled graph is built once at construction time and reused
    across requests (thread-safe, stateless between invocations).
    """

    def __init__(self) -> None:
        self._graph = _build_graph()
        logger.info("✅ SupervisorAgent graph compiled")

    async def run(self, query: str, session_id: str) -> Dict[str, Any]:
        """
        Execute the full supervisor flow and return immediately after
        Celery chains are dispatched.

        Returns
        -------
        {
            "job_id":     str,
            "status":     "dispatched" | "error",
            "task_count": int,          # chains launched
            "plan":       { subtasks, search_queries, expected_doc_count, complexity },
            "error":      str | None,
        }
        """
        initial_state: SupervisorState = {
            "query": query,
            "session_id": session_id,
            "job_id": "",
            "plan": {},
            "subtasks": [],
            "search_queries": [],
            "expected_doc_count": 0,
            "complexity": "medium",
            "status": "init",
            "error": None,
            "task_count": 0,
            "dispatched_at": None,
        }

        final_state = await self._graph.ainvoke(initial_state)

        return {
            "job_id":      final_state.get("job_id", ""),
            "session_id":  session_id,
            "status":      final_state.get("status", "error"),
            "task_count":  final_state.get("task_count", 0),
            "plan": {
                "subtasks":           final_state.get("subtasks", []),
                "search_queries":     final_state.get("search_queries", []),
                "expected_doc_count": final_state.get("expected_doc_count", 0),
                "complexity":         final_state.get("complexity", "medium"),
            },
            "error":       final_state.get("error"),
            "dispatched_at": final_state.get("dispatched_at"),
        }


# ── Prompt builder ─────────────────────────────────────────────────

def _build_planning_prompt(query: str) -> str:
    return f"""You are the SupervisorAgent of a multi-agent document intelligence system.

Your job is to analyse a user's research query and produce a structured task plan
that tells the downstream agents what to search for and how many documents to expect.

User query: "{query}"

Return ONLY valid JSON — no markdown fences, no explanation — in exactly this shape:

{{
  "subtasks": [
    "Brief description of subtask 1",
    "Brief description of subtask 2",
    "Brief description of subtask 3"
  ],
  "search_queries": [
    "specific web search query 1",
    "specific web search query 2",
    "specific web search query 3"
  ],
  "expected_doc_count": <integer between 3 and 20>,
  "complexity": "<one of: low | medium | high>"
}}

Rules:
- "subtasks" should be 2-5 concise action descriptions of what agents will do.
- "search_queries" should be 1-5 precise, distinct web-search strings derived
  from the user query. Favour recent, authoritative sources.
- "expected_doc_count" is your estimate of how many relevant documents exist.
- "complexity" reflects how broad or deep the research is:
    low    = single well-defined topic, few sources needed
    medium = multi-faceted topic, moderate sources
    high   = broad, comparative, or cutting-edge topic requiring many sources
- Do NOT include any text outside the JSON object.
"""


# ── JSON parsing helpers ───────────────────────────────────────────

def _parse_json_response(raw: str) -> Dict[str, Any]:
    """
    Parse Gemini's response to JSON.
    Handles cases where the model wraps the output in markdown fences.
    """
    # Strip optional markdown code fences
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Try to extract the first {...} block
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse JSON from Gemini response: {raw[:200]}")


def _validate_plan(plan: Dict[str, Any]) -> None:
    """Raise ValueError if required keys are missing or malformed."""
    required = {"subtasks", "search_queries", "expected_doc_count", "complexity"}
    missing = required - set(plan.keys())
    if missing:
        raise ValueError(f"Plan missing keys: {missing}")
    if not isinstance(plan["subtasks"], list) or not plan["subtasks"]:
        raise ValueError("'subtasks' must be a non-empty list")
    if not isinstance(plan["search_queries"], list) or not plan["search_queries"]:
        raise ValueError("'search_queries' must be a non-empty list")
    if plan["complexity"] not in ("low", "medium", "high"):
        raise ValueError(f"Invalid complexity: {plan['complexity']}")


# ── Module-level singleton (optional convenience) ──────────────────
supervisor_agent = SupervisorAgent()
