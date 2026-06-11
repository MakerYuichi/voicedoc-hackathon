"""
ScannerAgent — Celery task.
Full implementation in Step 5.

Chain signature
---------------
scanner_task.s(job_id, session_id, query, search_queries)
    → returns dict  (passed as prev_result to evaluator_task)
"""
from __future__ import annotations

from app.celery_config import QUEUE_SCANNER, celery_app


@celery_app.task(
    name="app.agents.scanner_agent.scanner_task",
    bind=True,
    queue=QUEUE_SCANNER,
    max_retries=3,
    default_retry_delay=5,
)
def scanner_task(
    self,
    job_id: str,
    session_id: str,
    query: str,
    search_queries: list,
) -> dict:
    """
    Fetch raw documents from the web for each search query.
    Returns a dict that is forwarded as `prev_result` to evaluator_task.
    Implemented in Step 5.
    """
    raise NotImplementedError("ScannerAgent not yet implemented (Step 5)")
