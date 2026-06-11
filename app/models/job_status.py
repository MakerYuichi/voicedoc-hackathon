"""
Pydantic models for the `job_status` collection.

Tracks the lifecycle of every Celery task spawned by the SupervisorAgent.
The WebSocket server polls this collection to push progress to the frontend.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class JobState(str, Enum):
    PENDING = "pending"
    STARTED = "started"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    REVOKED = "revoked"
    RETRY = "retry"


class JobStatusBase(BaseModel):
    job_id: str = Field(..., description="Celery task ID")
    session_id: str = Field(..., description="Session that spawned this job")
    agent_name: str = Field(
        ...,
        description="ScannerAgent | EvaluatorAgent | ExtractorAgent | ProcessorAgent",
    )
    status: JobState = Field(default=JobState.PENDING)
    progress: int = Field(default=0, ge=0, le=100, description="0-100 completion percent")
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Task input payload")
    result: Optional[Dict[str, Any]] = Field(None, description="Task output on success")
    error: Optional[str] = Field(None, description="Error message on failure")
    retry_count: int = Field(default=0)
    worker_id: Optional[str] = Field(None, description="Celery worker hostname")


class JobStatusCreate(JobStatusBase):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class JobStatusUpdate(BaseModel):
    status: Optional[JobState] = None
    progress: Optional[int] = Field(None, ge=0, le=100)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    retry_count: Optional[int] = None
    worker_id: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class JobStatusInDB(JobStatusCreate):
    id: Optional[str] = Field(None, alias="_id")

    model_config = {"populate_by_name": True}
