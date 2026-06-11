"""
Pydantic models for the `documents` collection.

Each document represents a raw web page or file fetched by the ScannerAgent.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, HttpUrl, field_validator


class DocumentStatus(str, Enum):
    PENDING = "pending"
    SCANNING = "scanning"
    EVALUATED = "evaluated"
    EXTRACTED = "extracted"
    PROCESSED = "processed"
    FAILED = "failed"


class DocumentBase(BaseModel):
    source_url: str = Field(..., description="URL the document was fetched from")
    title: Optional[str] = Field(None, description="Page or document title")
    content_type: Optional[str] = Field(None, description="MIME type, e.g. text/html")
    raw_content: Optional[str] = Field(None, description="Raw HTML / text before extraction")
    markdown_content: Optional[str] = Field(None, description="Cleaned markdown from ExtractorAgent")
    relevance_score: Optional[float] = Field(
        None, ge=0.0, le=10.0, description="Score assigned by EvaluatorAgent (0–10)"
    )
    status: DocumentStatus = Field(default=DocumentStatus.PENDING)
    session_id: str = Field(..., description="Session that triggered the fetch")
    job_id: Optional[str] = Field(None, description="Celery job ID")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Extra metadata")
    chunk_count: int = Field(default=0, description="Number of chunks stored for this document")
    error_message: Optional[str] = Field(None, description="Set when status=failed")


class DocumentCreate(DocumentBase):
    """Used when inserting a new document."""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentUpdate(BaseModel):
    """Partial update — all fields optional."""
    title: Optional[str] = None
    content_type: Optional[str] = None
    raw_content: Optional[str] = None
    markdown_content: Optional[str] = None
    relevance_score: Optional[float] = Field(None, ge=0.0, le=10.0)
    status: Optional[DocumentStatus] = None
    job_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    chunk_count: Optional[int] = None
    error_message: Optional[str] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class DocumentInDB(DocumentCreate):
    """Shape returned when reading from MongoDB (includes _id as string)."""
    id: Optional[str] = Field(None, alias="_id")
    updated_at: Optional[datetime] = None

    model_config = {"populate_by_name": True}
