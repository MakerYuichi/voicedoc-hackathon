"""
Pydantic models for the `chunks` collection.

Each chunk is a sub-section of a document with a vector embedding,
produced by the ProcessorAgent and used by the QueryAgent for RAG.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field


class ChunkBase(BaseModel):
    document_id: str = Field(..., description="Parent document _id (string)")
    session_id: str = Field(..., description="Session that created this chunk")
    content: str = Field(..., description="Raw text content of this chunk")
    embedding: List[float] = Field(
        ..., description="Dense vector embedding (3072-dim for gemini-embedding-001)"
    )
    chunk_index: int = Field(..., ge=0, description="0-based position in the parent document")
    token_count: int = Field(default=0, description="Approximate token count")
    source_url: str = Field(..., description="Inherited from parent document for convenience")
    title: Optional[str] = Field(None, description="Parent document title")
    metadata: dict = Field(default_factory=dict)


class ChunkCreate(ChunkBase):
    """Used when inserting a new chunk."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChunkInDB(ChunkCreate):
    """Shape returned when reading from MongoDB."""
    id: Optional[str] = Field(None, alias="_id")

    model_config = {"populate_by_name": True}
