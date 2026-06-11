"""
Utility helpers for VoiceDoc Intelligence.
"""
from app.utils.websocket_manager import ws_manager, WebSocketManager
from app.utils.job_manager import (
    create_job,
    create_agent_job,
    update_job_progress,
    update_job_progress_sync,
    get_job_status,
    get_session_jobs,
    broadcast_progress,
    broadcast_completion,
    mark_job_complete,
    mark_job_complete_sync,
    redis_health_check,
)
from app.utils.llm import get_llm, get_llm_sync, active_provider, llm_health_check
from app.utils.worker_db import run_async_with_db

__all__ = [
    "ws_manager",
    "WebSocketManager",
    "create_job",
    "create_agent_job",
    "update_job_progress",
    "update_job_progress_sync",
    "get_job_status",
    "get_session_jobs",
    "broadcast_progress",
    "broadcast_completion",
    "mark_job_complete",
    "mark_job_complete_sync",
    "redis_health_check",
    "get_llm",
    "get_llm_sync",
    "active_provider",
    "llm_health_check",
    "run_async_with_db",
]
