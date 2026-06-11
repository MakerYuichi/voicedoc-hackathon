"""
EvaluatorAgent — Celery task.
Full implementation in Step 6.

Chain signature
---------------
Previous task (scanner) returns a dict → Celery passes it as first arg.
evaluator_task(prev_result, job_id, session_id)
"""
from __future__ import annotations

from app.celery_config import QUEUE_EVALUATOR, celery_app


@celery_app.task(
    name="app.agents.evaluator_agent.evaluator_task",
    bind=True,
    queue=QUEUE_EVALUATOR,
    max_retries=3,
    default_retry_delay=5,
)
def evaluator_task(
    self,
    prev_result: dict,
    job_id: str,
    session_id: str,
) -> dict:
    """
    Score each scanned document for relevance (0-10).
    prev_result = scanner_task return value.
    Returns a dict forwarded to extractor_task.
    Implemented in Step 6.
    """
    raise NotImplementedError("EvaluatorAgent not yet implemented (Step 6)")
