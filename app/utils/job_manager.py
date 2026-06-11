"""
Job Manager — the bridge between Celery workers and the rest of the system.

Every function here is async so it can be called directly from FastAPI
route handlers or from an asyncio-aware context.

Celery workers are sync, so they call the *_sync variants via a helper
that runs a new event loop in a thread.

Public API
----------
create_job(...)            → inserts the parent job document
create_agent_job(...)      → inserts a per-agent sub-job document
update_job_progress(...)   → upserts progress for one agent sub-job
get_job_status(...)        → returns parent job + all sub-jobs
broadcast_progress(...)    → fetches latest state and pushes over WS
mark_job_complete(...)     → marks the parent job SUCCESS / FAILURE
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.database.db import db_manager
from app.models.job_status import JobState, JobStatusCreate
from app.utils.websocket_manager import ws_manager

logger = logging.getLogger(__name__)

# ── helpers ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _run_sync(coro):
    """
    Run an async coroutine from a sync (Celery worker) context.
    Creates a new event loop in the calling thread.
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── create ─────────────────────────────────────────────────────────

async def create_job(
    job_id: str,
    session_id: str,
    query: str,
    total_tasks: int,
    agent_names: List[str],
    input_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Insert the parent (supervisor) job record into MongoDB.

    Returns the inserted document dict.
    """
    doc = {
        "job_id": job_id,
        "session_id": session_id,
        "agent_name": "SupervisorAgent",
        "status": JobState.PENDING.value,
        "progress": 0,
        "input_data": {
            "query": query,
            "total_tasks": total_tasks,
            "agent_names": agent_names,
            **(input_data or {}),
        },
        "result": None,
        "error": None,
        "retry_count": 0,
        "worker_id": None,
        "created_at": _now(),
        "updated_at": _now(),
        "started_at": None,
        "completed_at": None,
    }
    await db_manager.job_status.insert_one(doc)
    logger.info(f"📋 Job created | job_id={job_id} | session={session_id} | tasks={total_tasks}")
    return doc


async def create_agent_job(
    job_id: str,
    session_id: str,
    agent_name: str,
    input_data: Optional[Dict[str, Any]] = None,
) -> None:
    """Insert a sub-job record for an individual agent task."""
    doc = JobStatusCreate(
        job_id=job_id,
        session_id=session_id,
        agent_name=agent_name,
        status=JobState.PENDING,
        progress=0,
        input_data=input_data or {},
    ).model_dump()
    await db_manager.job_status.insert_one(doc)
    logger.debug(f"  Sub-job created | {agent_name} | job_id={job_id}")


# ── update ─────────────────────────────────────────────────────────

async def update_job_progress(
    job_id: str,
    agent_name: str,
    status: JobState,
    progress: int = 0,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    worker_id: Optional[str] = None,
) -> None:
    """
    Upsert the progress record for one agent sub-job, then broadcast
    the updated state over WebSocket.

    Called by Celery workers via update_job_progress_sync().
    """
    now = _now()

    update_fields: Dict[str, Any] = {
        "status": status.value,
        "progress": max(0, min(100, progress)),
        "updated_at": now,
    }
    if result is not None:
        update_fields["result"] = result
    if error is not None:
        update_fields["error"] = error
    if worker_id is not None:
        update_fields["worker_id"] = worker_id
    if status == JobState.STARTED:
        update_fields["started_at"] = now
    if status in (JobState.SUCCESS, JobState.FAILURE):
        update_fields["completed_at"] = now

    await db_manager.job_status.update_one(
        {"job_id": job_id, "agent_name": agent_name},
        {"$set": update_fields},
        upsert=True,
    )

    logger.debug(
        f"  Progress updated | {agent_name} | job_id={job_id} "
        f"| status={status.value} | progress={progress}%"
    )

    # Push real-time update over WebSocket
    await broadcast_progress(job_id)


def update_job_progress_sync(
    job_id: str,
    agent_name: str,
    status: JobState,
    progress: int = 0,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
    worker_id: Optional[str] = None,
) -> None:
    """Sync wrapper — call this from Celery task functions."""
    _run_sync(
        update_job_progress(
            job_id=job_id,
            agent_name=agent_name,
            status=status,
            progress=progress,
            result=result,
            error=error,
            worker_id=worker_id,
        )
    )


# ── query ──────────────────────────────────────────────────────────

async def get_job_status(job_id: str) -> Optional[Dict[str, Any]]:
    """
    Return the parent job record merged with all agent sub-jobs.

    Shape:
        {
            "job_id": "...",
            "session_id": "...",
            "status": "running",
            "progress": 42,          # average across sub-jobs
            "agents": {
                "ScannerAgent":   { "status": "success", "progress": 100, ... },
                "EvaluatorAgent": { "status": "running", "progress": 60,  ... },
                ...
            },
            "created_at": "...",
            "updated_at": "...",
        }
    """
    cursor = db_manager.job_status.find({"job_id": job_id})
    records = await cursor.to_list(length=100)

    if not records:
        return None

    # Separate supervisor record from agent sub-jobs
    supervisor = next(
        (r for r in records if r.get("agent_name") == "SupervisorAgent"),
        records[0],
    )
    sub_jobs = {
        r["agent_name"]: _serialize_record(r)
        for r in records
        if r.get("agent_name") != "SupervisorAgent"
    }

    # Compute aggregate progress
    if sub_jobs:
        avg_progress = int(
            sum(v.get("progress", 0) for v in sub_jobs.values()) / len(sub_jobs)
        )
    else:
        avg_progress = supervisor.get("progress", 0)

    # Derive overall status
    statuses = [v.get("status") for v in sub_jobs.values()] if sub_jobs else [supervisor.get("status")]
    if all(s == JobState.SUCCESS.value for s in statuses):
        overall_status = JobState.SUCCESS.value
    elif any(s == JobState.FAILURE.value for s in statuses):
        overall_status = JobState.FAILURE.value
    elif any(s in (JobState.RUNNING.value, JobState.STARTED.value) for s in statuses):
        overall_status = JobState.RUNNING.value
    else:
        overall_status = supervisor.get("status", JobState.PENDING.value)

    return {
        "job_id": job_id,
        "session_id": supervisor.get("session_id"),
        "status": overall_status,
        "progress": avg_progress,
        "query": supervisor.get("input_data", {}).get("query"),
        "agents": sub_jobs,
        "created_at": _fmt_dt(supervisor.get("created_at")),
        "updated_at": _fmt_dt(supervisor.get("updated_at")),
        "completed_at": _fmt_dt(supervisor.get("completed_at")),
    }


async def get_session_jobs(session_id: str) -> List[Dict[str, Any]]:
    """Return all parent job IDs for a session."""
    cursor = db_manager.job_status.find(
        {"session_id": session_id, "agent_name": "SupervisorAgent"},
        {"job_id": 1, "status": 1, "progress": 1, "created_at": 1},
    ).sort("created_at", -1)
    records = await cursor.to_list(length=50)
    return [_serialize_record(r) for r in records]


# ── broadcast ──────────────────────────────────────────────────────

async def broadcast_progress(job_id: str) -> None:
    """
    Fetch the current job state and push it as a WebSocket message
    to all browser connections for the owning session.

    Message shape:
        {
            "type": "progress",
            "job_id": "...",
            "status": "running",
            "progress": 42,
            "agents": { ... }
        }
    """
    state = await get_job_status(job_id)
    if not state:
        return

    session_id = state.get("session_id")
    if not session_id:
        return

    await ws_manager.broadcast_to_session(
        session_id=session_id,
        payload={
            "type": "progress",
            **state,
        },
    )


async def broadcast_completion(job_id: str, message: str) -> None:
    """
    Push a completion notification once all pipeline stages are done.
    """
    state = await get_job_status(job_id)
    if not state:
        return

    session_id = state.get("session_id")
    if not session_id:
        return

    await ws_manager.broadcast_to_session(
        session_id=session_id,
        payload={
            "type": "complete",
            "job_id": job_id,
            "status": state.get("status"),
            "progress": 100,
            "message": message,
            "agents": state.get("agents", {}),
        },
    )


# ── mark complete ──────────────────────────────────────────────────

async def mark_job_complete(
    job_id: str,
    success: bool,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Update the supervisor job record to a terminal state."""
    now = _now()
    state = JobState.SUCCESS if success else JobState.FAILURE
    await db_manager.job_status.update_one(
        {"job_id": job_id, "agent_name": "SupervisorAgent"},
        {
            "$set": {
                "status": state.value,
                "progress": 100 if success else 0,
                "result": result,
                "error": error,
                "updated_at": now,
                "completed_at": now,
            }
        },
    )
    logger.info(f"🏁 Job {'completed' if success else 'FAILED'} | job_id={job_id}")


def mark_job_complete_sync(
    job_id: str,
    success: bool,
    result: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> None:
    """Sync wrapper for Celery workers."""
    _run_sync(mark_job_complete(job_id, success, result, error))


# ── Redis health check ─────────────────────────────────────────────

async def redis_health_check() -> Dict[str, str]:
    """Ping Redis via the async redis client."""
    try:
        import redis.asyncio as aioredis
        from app.config import settings

        client = aioredis.from_url(settings.redis_url, socket_connect_timeout=3)
        pong = await client.ping()
        await client.aclose()
        return {"redis": "healthy" if pong else "unhealthy"}
    except Exception as exc:
        logger.error(f"Redis health check failed: {exc}")
        return {"redis": "unhealthy", "error": str(exc)}


# ── serialisation helpers ──────────────────────────────────────────

def _serialize_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw MongoDB doc to a JSON-safe dict."""
    out = dict(record)
    out.pop("_id", None)             # remove ObjectId
    out["created_at"]   = _fmt_dt(out.get("created_at"))
    out["updated_at"]   = _fmt_dt(out.get("updated_at"))
    out["started_at"]   = _fmt_dt(out.get("started_at"))
    out["completed_at"] = _fmt_dt(out.get("completed_at"))
    return out


def _fmt_dt(dt: Any) -> Optional[str]:
    if dt is None:
        return None
    if isinstance(dt, datetime):
        return dt.isoformat()
    return str(dt)
