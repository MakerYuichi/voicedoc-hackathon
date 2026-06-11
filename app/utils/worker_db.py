"""
worker_db.py — Database connection helper for Celery workers.

Celery worker processes are forked from the main process BEFORE
FastAPI's lifespan runs, so db_manager is never connected in workers.

Design
------
Each Celery task calls `ensure_worker_db()` once at the start of the
task body (not on every MongoDB call). This creates a single Motor
client + event loop that persists for the lifetime of the task.

The task's sync helpers (`_run_async`, `update_job_progress_sync`, etc.)
reuse that shared loop and client via `call_async()`.

Usage in task functions
-----------------------
    from app.utils.worker_db import ensure_worker_db, call_async

    def my_celery_task(self, job_id, session_id):
        ensure_worker_db()          # connect once at task start

        # all DB calls within this task use call_async():
        call_async(some_mongo_coroutine())
        call_async(another_coroutine())

        # no explicit disconnect needed — the loop stays open until
        # the worker process is recycled (worker_max_tasks_per_child)

Usage for one-off DB coroutines from non-task code
--------------------------------------------------
    from app.utils.worker_db import run_async_with_db

    result = run_async_with_db(some_coroutine())
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── per-thread state ───────────────────────────────────────────────
# Each Celery prefork worker thread gets its own event loop + Motor client
# via threading.local(), so concurrent tasks on the same worker process
# don't share loops.

_local = threading.local()


def _get_loop() -> Optional[asyncio.AbstractEventLoop]:
    return getattr(_local, 'loop', None)


def _set_loop(loop: asyncio.AbstractEventLoop) -> None:
    _local.loop = loop


def ensure_worker_db() -> None:
    """
    Call once at the top of a Celery task.
    Creates (or reuses) a persistent event loop + Motor connection for
    this worker thread, then connects db_manager on that loop.
    """
    loop = _get_loop()

    # If we already have a running loop with a connected db_manager, reuse it
    if loop is not None and not loop.is_closed():
        from app.database.db import db_manager
        if db_manager._client is not None:
            return  # already connected on this loop — nothing to do

    # Create a fresh event loop for this thread
    if loop is not None and not loop.is_closed():
        loop.close()
    loop = asyncio.new_event_loop()
    _set_loop(loop)

    # Reset any stale Motor client and connect on the new loop
    from app.database.db import db_manager
    db_manager._client = None
    db_manager._db = None

    logger.debug("Worker DB: connecting Motor on new event loop")
    loop.run_until_complete(db_manager.connect())
    logger.debug("Worker DB: Motor connected")


def call_async(coro) -> Any:
    """
    Run an async coroutine on the worker's persistent event loop.
    Must call ensure_worker_db() first.
    """
    loop = _get_loop()
    if loop is None or loop.is_closed():
        # Fallback: create a fresh loop (shouldn't happen if task calls ensure_worker_db)
        logger.warning("Worker DB: call_async called without ensure_worker_db — auto-fixing")
        ensure_worker_db()
        loop = _get_loop()
    return loop.run_until_complete(coro)


def run_async_with_db(coro) -> Any:
    """
    One-shot helper: connect, run coroutine, disconnect.
    Use for infrequent calls outside of a Celery task context.
    For task code, prefer ensure_worker_db() + call_async().
    """
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(_one_shot(coro))
    finally:
        loop.close()


async def _one_shot(coro) -> Any:
    from app.database.db import db_manager
    db_manager._client = None
    db_manager._db = None
    # Use a lightweight connect that skips slow index creation
    from motor.motor_asyncio import AsyncIOMotorClient
    from app.config import settings
    db_manager._client = AsyncIOMotorClient(
        settings.mongodb_uri,
        serverSelectionTimeoutMS=10_000,
        connectTimeoutMS=10_000,
        socketTimeoutMS=30_000,
    )
    db_manager._db = db_manager._client[settings.mongodb_database]
    await db_manager._ping()
    try:
        return await coro
    finally:
        db_manager._client.close()
        db_manager._client = None
        db_manager._db = None
