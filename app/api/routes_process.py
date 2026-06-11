"""
POST /api/process  — trigger the document-ingestion pipeline
GET  /api/job/{job_id} — poll per-agent progress
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.agents.supervisor_agent import supervisor_agent
from app.utils.job_manager import get_job_status

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["process"])


# ── request / response models ──────────────────────────────────────

class ProcessRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000,
                       description="Voice-transcribed or typed user query")
    session_id: str = Field(..., min_length=1, max_length=128,
                            description="Browser session identifier")
    direct_urls: Optional[list[str]] = Field(
        default=None, description="Optional list of URLs to scan directly"
    )


class ProcessResponse(BaseModel):
    job_id: str
    status: str                     # "dispatched" | "error"
    session_id: str
    task_count: int                 # Celery chains launched
    plan: dict
    error: Optional[str] = None
    dispatched_at: Optional[str] = None


# ── endpoints ──────────────────────────────────────────────────────

@router.post("/process", response_model=ProcessResponse, status_code=202)
async def start_processing(req: ProcessRequest) -> ProcessResponse:
    """
    Receive a voice-transcribed query, run the SupervisorAgent to plan
    subtasks and dispatch parallel Celery chains.

    Returns immediately with a job_id — the actual processing is async.
    The frontend polls GET /api/job/{job_id} or listens on WebSocket.
    """
    logger.info(
        f"POST /api/process | session={req.session_id} | "
        f"query={req.query[:80]!r}"
    )

    try:
        result = await supervisor_agent.run(
            query=req.query,
            session_id=req.session_id,
        )
    except Exception as exc:
        logger.error(f"SupervisorAgent.run failed: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))

    if result.get("status") == "error" and not result.get("job_id"):
        raise HTTPException(
            status_code=422,
            detail=result.get("error", "SupervisorAgent returned an error"),
        )

    return ProcessResponse(
        job_id=result["job_id"],
        status=result["status"],
        session_id=result["session_id"],
        task_count=result["task_count"],
        plan=result["plan"],
        error=result.get("error"),
        dispatched_at=result.get("dispatched_at"),
    )


@router.get("/job/{job_id}", tags=["process"])
async def get_job(job_id: str) -> dict:
    """
    Return the current status of a pipeline job including
    per-agent progress bars.

    Shape:
        {
            "job_id": str,
            "session_id": str,
            "status": "pending|running|success|failure",
            "progress": int (0-100, avg of sub-agents),
            "agents": {
                "ScannerAgent":   { "status": ..., "progress": ... },
                "EvaluatorAgent": { ... },
                "ExtractorAgent": { ... },
                "ProcessorAgent": { ... },
            },
            "created_at": str,
            "updated_at": str,
        }
    """
    state = await get_job_status(job_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return state
