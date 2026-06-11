"""
QueryAgent — Agentic RAG using LangGraph (Step 9).

The agent decides dynamically whether retrieval is needed and how to
query — it is not a hardcoded retrieve-then-generate pipeline.

LangGraph Flow
--------------
                    ┌─────────────────────────────────────┐
                    │         QueryAgent StateGraph        │
                    └─────────────────────────────────────┘
                                      │
                               should_retrieve
                              /              \\
                      DIRECT_ANSWER      NEED_RETRIEVAL
                            │                  │
                     generate_answer  generate_search_query
                            │                  │
                      save_log &         vector_search
                      return                   │
                                        evaluate_chunks
                                       /             \\
                               SUFFICIENT           RETRY
                                   │           (once only)
                            generate_answer  generate_search_query
                                   │                  │
                             save_log &         vector_search
                             return                    │
                                             evaluate_chunks
                                                       │
                                              generate_answer
                                                       │
                                                 save_log &
                                                   return

State fields
------------
query            : original user question
session_id       : browser session
search_query     : potentially rewritten query for vector search
chunks           : raw retrieved chunk docs from MongoDB
filtered_chunks  : chunks that passed quality threshold
answer           : final generated answer
sources          : SourceReference list for the frontend
confidence       : float 0-1 overall answer confidence
retrieval_needed : bool — LLM decision
retrieval_done   : bool — guards against infinite retry loops
retry_count      : int  — max 1 retry allowed
latency_ms       : int
error            : str | None

Usage
-----
    from app.agents.query_agent import QueryAgent

    agent = QueryAgent()
    result = await agent.run(
        query="What are the key components of a RAG system?",
        session_id="sess-abc123",
    )
    # result.answer, result.sources, result.confidence, result.latency_ms
"""
from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langgraph.graph import END, StateGraph

from app.config import settings
from app.database.db import db_manager
from app.models.query_logs import QueryLogCreate, SourceReference
from app.utils.llm import get_llm

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME          = "QueryAgent"
TOP_K_RETRIEVE      = 10     # chunks to fetch from Atlas
TOP_K_CONTEXT       = 5      # best chunks to include in the LLM prompt
MIN_SIMILARITY      = 0.75   # Atlas $vectorSearch score threshold
CHUNK_QUALITY_MIN   = 6.0    # LLM chunk quality score (0-10) to keep
MAX_RETRIES         = 1      # one reformulation allowed


# ── LangGraph state ────────────────────────────────────────────────

class QueryState(TypedDict):
    # inputs
    query: str
    session_id: str
    # retrieval
    search_query: str
    retrieval_needed: bool
    retrieval_done: bool
    retry_count: int
    chunks: List[Dict[str, Any]]         # raw Atlas results
    filtered_chunks: List[Dict[str, Any]] # quality-filtered chunks
    # generation
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    # meta
    latency_ms: int
    start_ts: float
    error: Optional[str]
    routing: str  # "need_retrieval" | "direct_answer" | "sufficient" | "retry"


# ── Embedding helper ───────────────────────────────────────────────

def _get_embeddings() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,           # models/gemini-embedding-001
        google_api_key=settings.google_api_key,
        task_type="retrieval_query",              # query-side encoding
    )


async def _embed_query(text: str) -> List[float]:
    """Async embed a single query string."""
    embedder = _get_embeddings()
    loop = __import__("asyncio").get_event_loop()
    vector = await loop.run_in_executor(None, embedder.embed_query, text)
    return vector


# ── Atlas Vector Search ────────────────────────────────────────────

async def _vector_search(
    query_vector: List[float],
    session_id: str,
    top_k: int = TOP_K_RETRIEVE,
) -> List[Dict[str, Any]]:
    """
    Run MongoDB Atlas $vectorSearch on the chunks collection.

    Filters to chunks belonging to the caller's session, returns
    top_k results with Atlas-assigned relevance score.

    Returns list of dicts: {content, source_url, title, document_id,
                             chunk_index, score, metadata}
    """
    pipeline = [
        {
            "$vectorSearch": {
                "index": settings.vector_index_name,
                "path": "embedding",
                "queryVector": query_vector,
                "numCandidates": top_k * 10,   # wider search for better recall
                "limit": top_k,
                # Note: session_id filter requires it to be declared as a token
                # field in the Atlas Vector Search index definition.
                # We post-filter instead to avoid index configuration requirements.
            }
        },
        {
            "$project": {
                "_id": 1,
                "content": 1,
                "source_url": 1,
                "title": 1,
                "document_id": 1,
                "chunk_index": 1,
                "metadata": 1,
                "session_id": 1,
                "score": {"$meta": "vectorSearchScore"},
            }
        },
        # Post-filter: minimum similarity + session scope
        {"$match": {"score": {"$gte": MIN_SIMILARITY}, "session_id": session_id}},
    ]

    try:
        cursor = db_manager.chunks.aggregate(pipeline)
        results = await cursor.to_list(length=top_k)
        return [
            {
                "chunk_id": str(r["_id"]),
                "content": r.get("content", ""),
                "source_url": r.get("source_url", ""),
                "title": r.get("title", ""),
                "document_id": str(r.get("document_id", "")),
                "chunk_index": r.get("chunk_index", 0),
                "score": round(r.get("score", 0.0), 4),
                "metadata": r.get("metadata", {}),
            }
            for r in results
        ]
    except Exception as exc:
        logger.error(f"Vector search failed: {exc}")
        return []


# ── JSON parse helper ──────────────────────────────────────────────

def _parse_json(raw: str) -> Dict[str, Any]:
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise ValueError(f"Cannot parse JSON: {cleaned[:200]}")


# ── Node 1: should_retrieve ────────────────────────────────────────

async def node_should_retrieve(state: QueryState) -> QueryState:
    """
    LLM decides: does this question need document retrieval, or can
    it be answered from general knowledge?

    Returns routing = "need_retrieval" | "direct_answer"
    """
    prompt = f"""You are a routing agent for a document intelligence system.

A user has asked: "{state['query']}"

Decide whether answering this question requires searching the user's personal
document store (uploaded/indexed documents), or whether it can be answered
from general knowledge alone.

Rules:
- If the question references "my documents", "the documents", "what was stored",
  "what did I upload", specific project names, or anything suggesting private
  knowledge → RETRIEVAL NEEDED
- If the question is general knowledge (definitions, explanations, public facts) → DIRECT

Return ONLY valid JSON:
{{"needs_retrieval": true/false, "reason": "one sentence"}}"""

    try:
        llm = get_llm(temperature=0.0, max_tokens=100)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        result = _parse_json(response.content)
        needs = bool(result.get("needs_retrieval", True))
        reason = result.get("reason", "")
        logger.info(f"  Retrieval decision: {'YES' if needs else 'NO'} — {reason}")
    except Exception as exc:
        logger.warning(f"Routing decision failed, defaulting to retrieval: {exc}")
        needs = True

    return {
        **state,
        "retrieval_needed": needs,
        "routing": "need_retrieval" if needs else "direct_answer",
    }


# ── Node 2: generate_search_query ─────────────────────────────────

async def node_generate_search_query(state: QueryState) -> QueryState:
    """
    Rewrite the user question into a search query optimised for
    vector similarity retrieval.

    On retry (retry_count > 0): reformulate with broader/different terms.
    """
    is_retry = state["retry_count"] > 0
    retry_note = (
        "\n\nPrevious retrieval found insufficient context. "
        "Generate a DIFFERENT, broader search query using synonyms or related terms."
        if is_retry else ""
    )

    prompt = f"""Convert the following user question into a concise search query
optimised for semantic vector search in a document knowledge base.

User question: "{state['query']}"{retry_note}

Rules:
- Extract the core concepts and entities
- Use noun phrases, not full sentences  
- Include relevant synonyms or related terms
- Keep it under 20 words

Return ONLY the search query string, no explanation, no quotes."""

    try:
        llm = get_llm(temperature=0.2, max_tokens=60)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        search_q = response.content.strip().strip('"').strip("'")
        if not search_q:
            search_q = state["query"]
    except Exception as exc:
        logger.warning(f"Query generation failed, using raw query: {exc}")
        search_q = state["query"]

    logger.info(
        f"  Search query {'(retry)' if is_retry else ''}: {search_q[:80]}"
    )
    return {**state, "search_query": search_q}


# ── Node 3: vector_search ──────────────────────────────────────────

async def node_vector_search(state: QueryState) -> QueryState:
    """
    Embed the search query and run MongoDB Atlas $vectorSearch.
    Stores raw results (before quality filtering) in state.chunks.
    """
    search_q = state.get("search_query") or state["query"]

    try:
        vector = await _embed_query(search_q)
        chunks = await _vector_search(
            query_vector=vector,
            session_id=state["session_id"],
            top_k=TOP_K_RETRIEVE,
        )
        logger.info(
            f"  Vector search: '{search_q[:60]}' → {len(chunks)} chunks "
            f"(threshold={MIN_SIMILARITY})"
        )
    except Exception as exc:
        logger.error(f"Vector search error: {exc}")
        chunks = []

    return {**state, "chunks": chunks, "retrieval_done": True}


# ── Node 4: evaluate_chunks ────────────────────────────────────────

async def node_evaluate_chunks(state: QueryState) -> QueryState:
    """
    LLM scores each retrieved chunk for relevance to the query.
    Keeps only chunks scoring >= CHUNK_QUALITY_MIN (6/10).
    Selects top TOP_K_CONTEXT (5) for the answer prompt.

    Sets routing = "sufficient" | "retry"
    """
    chunks = state.get("chunks", [])

    if not chunks:
        logger.info("  No chunks retrieved — routing to retry or direct answer")
        # Only retry if we haven't already done so
        if state["retry_count"] < MAX_RETRIES:
            return {**state, "filtered_chunks": [], "routing": "retry",
                    "retry_count": state["retry_count"] + 1}
        else:
            return {**state, "filtered_chunks": [], "routing": "direct_answer"}

    # Build compact chunk list for scoring
    chunk_summaries = "\n".join(
        f"{i+1}. [{c['source_url'][:60]}] {c['content'][:300]}"
        for i, c in enumerate(chunks[:10])
    )

    prompt = f"""You are evaluating retrieved document chunks for a question.

Question: "{state['query']}"

Retrieved chunks (each prefixed with its source URL):
---
{chunk_summaries}
---

Rate each chunk's relevance to the question (0-10).
Return ONLY a JSON array of integers, one per chunk, in order:
[score1, score2, ...]"""

    try:
        llm = get_llm(temperature=0.0, max_tokens=100)
        response = await llm.ainvoke([HumanMessage(content=prompt)])
        raw = re.sub(r"```(?:json)?\s*", "", response.content.strip()).strip()
        # Extract array
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
        scores = json.loads(m.group()) if m else []
        scores = [float(s) for s in scores]
    except Exception as exc:
        logger.warning(f"Chunk evaluation failed, keeping all: {exc}")
        scores = [7.0] * len(chunks)

    # Pair chunks with scores and filter
    scored = []
    for i, c in enumerate(chunks):
        score = scores[i] if i < len(scores) else 5.0
        c = dict(c)
        c["quality_score"] = score
        scored.append(c)

    # Sort by quality_score desc, then take top TOP_K_CONTEXT
    scored.sort(key=lambda x: x["quality_score"], reverse=True)
    filtered = [c for c in scored if c["quality_score"] >= CHUNK_QUALITY_MIN]

    if not filtered:
        logger.info(f"  No chunks passed quality threshold ({CHUNK_QUALITY_MIN}/10)")
        if state["retry_count"] < MAX_RETRIES:
            routing = "retry"
            retry_count = state["retry_count"] + 1
        else:
            routing = "direct_answer"
            retry_count = state["retry_count"]
    else:
        filtered = filtered[:TOP_K_CONTEXT]
        routing = "sufficient"
        retry_count = state["retry_count"]

    logger.info(
        f"  Chunk eval: {len(chunks)} in → {len(filtered)} passed → "
        f"routing={routing} retry_count={retry_count}"
    )
    return {
        **state,
        "filtered_chunks": filtered,
        "retry_count": retry_count,
        "routing": routing,
    }


# ── Node 5: generate_answer ────────────────────────────────────────

async def node_generate_answer(state: QueryState) -> QueryState:
    """
    Generate the final answer using retrieved context (or general knowledge
    if no chunks were found).

    Also builds the sources list and confidence score.
    """
    chunks   = state.get("filtered_chunks", [])
    has_ctx  = len(chunks) > 0

    # Shared identity constraint applied to every invocation
    _IDENTITY = (
        "You are VoiceDoc Intelligence's QueryAgent. "
        "You ONLY answer questions about documents processed in this session via RAG. "
        "You are NOT a general-purpose assistant. "
        "Do not offer general capabilities, translation, games, or unrelated help."
    )

    # Build context block
    if has_ctx:
        context_block = "\n\n---\n\n".join(
            f"Source {i+1}: {c['source_url']}\n"
            f"Title: {c.get('title', 'Unknown')}\n\n"
            f"{c['content']}"
            for i, c in enumerate(chunks)
        )
        system_msg = (
            f"{_IDENTITY}\n\n"
            "Answer the question based ONLY on the provided document sources below. "
            "Cite sources by number [1], [2], etc. "
            "If the sources don't contain enough information to answer fully, say so explicitly — "
            "do not supplement with general knowledge."
        )
        user_msg = (
            f"Question: {state['query']}\n\n"
            f"Sources:\n{context_block}\n\n"
            "Provide a comprehensive, well-structured answer with citations."
        )
    else:
        system_msg = (
            f"{_IDENTITY}\n\n"
            "No relevant documents were found in this session's store for the user's question. "
            "You MUST respond with exactly this message and nothing else:\n"
            "\"No documents have been processed yet. "
            "Try submitting a voice command to scan and process documents first.\""
        )
        user_msg = state["query"]

    try:
        llm = get_llm(temperature=0.3, max_tokens=1024)
        response = await llm.ainvoke([
            SystemMessage(content=system_msg),
            HumanMessage(content=user_msg),
        ])
        answer = response.content.strip()
    except Exception as exc:
        logger.error(f"Answer generation failed: {exc}")
        answer = f"I encountered an error generating the answer: {exc}"

    # Build sources list
    sources = [
        {
            "chunk_id":       c["chunk_id"],
            "document_id":    c["document_id"],
            "source_url":     c["source_url"],
            "title":          c.get("title", ""),
            "relevance_score": round(c.get("score", 0.0), 4),
            "excerpt":        c["content"][:200],
        }
        for c in chunks
    ]

    # Confidence: based on number + quality of chunks
    if chunks:
        avg_score = sum(c.get("score", 0.5) for c in chunks) / len(chunks)
        confidence = round(min(0.95, avg_score * (len(chunks) / TOP_K_CONTEXT)), 2)
    else:
        confidence = 0.3

    elapsed = int((time.time() - state["start_ts"]) * 1000)

    logger.info(
        f"  Answer generated | {len(answer)} chars | "
        f"sources={len(sources)} | confidence={confidence} | {elapsed}ms"
    )

    return {
        **state,
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "latency_ms": elapsed,
    }


# ── Node 6: save_log ───────────────────────────────────────────────

async def node_save_log(state: QueryState) -> QueryState:
    """Persist the query + answer to MongoDB query_logs collection."""
    try:
        sources_pydantic = [
            SourceReference(
                chunk_id=s["chunk_id"],
                document_id=s["document_id"],
                source_url=s["source_url"],
                title=s.get("title"),
                relevance_score=max(0.0, min(1.0, s.get("relevance_score", 0.0))),
                excerpt=s.get("excerpt"),
            )
            for s in state.get("sources", [])
        ]

        log = QueryLogCreate(
            session_id=state["session_id"],
            query=state["query"],
            answer=state.get("answer"),
            sources=sources_pydantic,
            latency_ms=state.get("latency_ms"),
            model_used=settings.gemini_model,
            metadata={
                "search_query":   state.get("search_query", ""),
                "retrieval_needed": state.get("retrieval_needed", True),
                "chunks_retrieved": len(state.get("chunks", [])),
                "chunks_used":    len(state.get("filtered_chunks", [])),
                "confidence":     state.get("confidence", 0.0),
                "retry_count":    state.get("retry_count", 0),
            },
        )
        await db_manager.query_logs.insert_one(log.model_dump())
        logger.debug("  Query log saved")
    except Exception as exc:
        logger.warning(f"Failed to save query log: {exc}")

    return state


# ── Routers ────────────────────────────────────────────────────────

def _route_should_retrieve(state: QueryState) -> str:
    return state.get("routing", "need_retrieval")


def _route_after_evaluate(state: QueryState) -> str:
    return state.get("routing", "sufficient")


# ── Graph assembly ─────────────────────────────────────────────────

def _build_graph() -> StateGraph:
    g = StateGraph(QueryState)

    # Add nodes
    g.add_node("should_retrieve",       node_should_retrieve)
    g.add_node("generate_search_query", node_generate_search_query)
    g.add_node("vector_search",         node_vector_search)
    g.add_node("evaluate_chunks",       node_evaluate_chunks)
    g.add_node("generate_answer",       node_generate_answer)
    g.add_node("save_log",              node_save_log)

    # Entry point
    g.set_entry_point("should_retrieve")

    # Routing after should_retrieve
    g.add_conditional_edges(
        "should_retrieve",
        _route_should_retrieve,
        {
            "need_retrieval": "generate_search_query",
            "direct_answer":  "generate_answer",
        },
    )

    # Retrieval path
    g.add_edge("generate_search_query", "vector_search")
    g.add_edge("vector_search",         "evaluate_chunks")

    # Routing after evaluate_chunks
    g.add_conditional_edges(
        "evaluate_chunks",
        _route_after_evaluate,
        {
            "sufficient":    "generate_answer",
            "retry":         "generate_search_query",  # reformulate once
            "direct_answer": "generate_answer",        # no context found
        },
    )

    # Both paths converge at generate_answer → save_log → END
    g.add_edge("generate_answer", "save_log")
    g.add_edge("save_log",        END)

    return g.compile()


# ── Public API ─────────────────────────────────────────────────────

class QueryResult:
    """Structured return type from QueryAgent.run()."""
    __slots__ = ("answer", "sources", "confidence", "latency_ms",
                 "chunks_used", "retrieval_needed", "search_query", "error")

    def __init__(self, state: QueryState) -> None:
        self.answer           = state.get("answer", "")
        self.sources          = state.get("sources", [])
        self.confidence       = state.get("confidence", 0.0)
        self.latency_ms       = state.get("latency_ms", 0)
        self.chunks_used      = len(state.get("filtered_chunks", []))
        self.retrieval_needed = state.get("retrieval_needed", True)
        self.search_query     = state.get("search_query", "")
        self.error            = state.get("error")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "answer":           self.answer,
            "sources":          self.sources,
            "confidence":       self.confidence,
            "latency_ms":       self.latency_ms,
            "chunks_used":      self.chunks_used,
            "retrieval_needed": self.retrieval_needed,
            "search_query":     self.search_query,
            "error":            self.error,
        }


class QueryAgent:
    """
    Agentic RAG agent using LangGraph.

    The compiled graph is built once and reused across requests —
    each call to run() creates a fresh, isolated state.
    """

    def __init__(self) -> None:
        self._graph = _build_graph()
        logger.info("✅ QueryAgent graph compiled")

    async def run(self, query: str, session_id: str) -> QueryResult:
        """
        Run the agentic RAG pipeline.

        Parameters
        ----------
        query      : user's natural language question
        session_id : browser session (limits retrieval scope)

        Returns
        -------
        QueryResult with answer, sources, confidence, latency_ms
        """
        start_ts = time.time()

        # ── Pre-check: skip LangGraph entirely if no chunks exist ──
        # This avoids wasting LLM calls and prevents Gemini from acting
        # as a general assistant when the user hasn't processed any docs yet.
        try:
            chunk_count = await db_manager.chunks.count_documents(
                {"session_id": session_id}
            )
            if chunk_count == 0:
                logger.info(
                    f"QueryAgent pre-check: no chunks for session={session_id}, "
                    "returning canned response"
                )
                elapsed = int((time.time() - start_ts) * 1000)
                canned = (
                    "No documents have been processed yet. "
                    "Try submitting a voice command to scan and process documents first."
                )
                # Still save to query_logs for analytics
                try:
                    from app.models.query_logs import QueryLogCreate
                    await db_manager.query_logs.insert_one(
                        QueryLogCreate(
                            session_id=session_id,
                            query=query,
                            answer=canned,
                            latency_ms=elapsed,
                            model_used="none",
                            metadata={"pre_check": "no_chunks"},
                        ).model_dump()
                    )
                except Exception:
                    pass
                return QueryResult({
                    "query": query, "session_id": session_id,
                    "search_query": "", "retrieval_needed": False,
                    "retrieval_done": False, "retry_count": 0,
                    "chunks": [], "filtered_chunks": [],
                    "answer": canned, "sources": [],
                    "confidence": 0.0, "latency_ms": elapsed,
                    "start_ts": start_ts, "error": None, "routing": "no_chunks",
                })
        except Exception as exc:
            # DB unavailable — log and continue to LangGraph
            logger.warning(f"Pre-check DB query failed: {exc}")

        initial: QueryState = {
            "query":            query,
            "session_id":       session_id,
            "search_query":     "",
            "retrieval_needed": True,
            "retrieval_done":   False,
            "retry_count":      0,
            "chunks":           [],
            "filtered_chunks":  [],
            "answer":           "",
            "sources":          [],
            "confidence":       0.0,
            "latency_ms":       0,
            "start_ts":         start_ts,
            "error":            None,
            "routing":          "",
        }

        try:
            final = await self._graph.ainvoke(
                initial,
                config={"recursion_limit": 10},
            )
            return QueryResult(final)
        except Exception as exc:
            logger.error(f"QueryAgent.run failed: {exc}", exc_info=True)
            elapsed = int((time.time() - start_ts) * 1000)
            return QueryResult({
                **initial,
                "answer": f"An error occurred: {exc}",
                "confidence": 0.0,
                "latency_ms": elapsed,
                "error": str(exc),
            })


# ── Module-level singleton ─────────────────────────────────────────
query_agent = QueryAgent()
