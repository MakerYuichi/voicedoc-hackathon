"""
Pydantic models for the `query_logs` collection.

Records every user question and the QueryAgent's RAG answer for analytics
and session replay.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SourceReference(BaseModel):
    chunk_id: str
    document_id: str
    source_url: str
    title: Optional[str] = None
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    excerpt: Optional[str] = Field(None, description="Short snippet from the matching chunk")


class QueryLogBase(BaseModel):
    session_id: str = Field(..., description="Session that issued the query")
    query: str = Field(..., description="Raw user question")
    answer: Optional[str] = Field(None, description="Generated answer from QueryAgent")
    sources: List[SourceReference] = Field(
        default_factory=list,
        description="Chunks retrieved and cited in the answer",
    )
    latency_ms: Optional[int] = Field(None, description="End-to-end query latency in ms")
    model_used: Optional[str] = Field(None, description="Gemini model variant used")
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QueryLogCreate(QueryLogBase):
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class QueryLogInDB(QueryLogCreate):
    id: Optional[str] = Field(None, alias="_id")

    model_config = {"populate_by_name": True}
