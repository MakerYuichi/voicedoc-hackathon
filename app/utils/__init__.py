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
]
