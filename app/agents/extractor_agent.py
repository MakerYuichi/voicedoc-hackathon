"""
ExtractorAgent — Celery task.
Full implementation in Step 7.

Chain signature
---------------
extractor_task(prev_result, job_id, session_id)
prev_result = evaluator_task return value.
"""
from __future__ import annotations

from app.celery_config import QUEUE_EXTRACTOR, celery_app


@celery_app.task(
    name="app.agents.extractor_agent.extractor_task",
    bind=True,
    queue=QUEUE_EXTRACTOR,
    max_retries=3,
    default_retry_delay=5,
)
def extractor_task(
    self,
    prev_result: dict,
    job_id: str,
    session_id: str,
) -> dict:
    """
    Convert high-scoring HTML pages to clean markdown.
    prev_result = evaluator_task return value.
    Returns a dict forwarded to processor_task.
    Implemented in Step 7.
    """
    raise NotImplementedError("ExtractorAgent not yet implemented (Step 7)")
