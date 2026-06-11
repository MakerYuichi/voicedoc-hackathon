"""
POST /api/query — agentic RAG question answering
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.query_agent import query_agent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["query"])


# ── request / response models ──────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    session_id: str = Field(..., min_length=1, max_length=128)


class SourceDoc(BaseModel):
    chunk_id: str
    document_id: str
    source_url: str
    title: Optional[str] = None
    relevance_score: float
    excerpt: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    sources: List[Dict[str, Any]]
    confidence: float
    latency_ms: int
    chunks_used: int
    retrieval_needed: bool
    search_query: Optional[str] = None
    error: Optional[str] = None


# ── endpoint ───────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse)
async def run_query(req: QueryRequest) -> QueryResponse:
    """
    Run the agentic RAG pipeline:
    1. LLM decides if retrieval is needed
    2. Embeds query → MongoDB Atlas Vector Search
    3. LLM scores retrieved chunks
    4. Generates answer with source citations
    5. Saves to query_logs

    Returns the answer with source documents and confidence score.
    """
    logger.info(
        f"POST /api/query | session={req.session_id} | "
        f"question={req.question[:80]!r}"
    )

    try:
        result = await query_agent.run(
            query=req.question,
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error(f"QueryAgent.run failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        confidence=result.confidence,
        latency_ms=result.latency_ms,
        chunks_used=result.chunks_used,
        retrieval_needed=result.retrieval_needed,
        search_query=result.search_query or None,
        error=result.error,
    )
