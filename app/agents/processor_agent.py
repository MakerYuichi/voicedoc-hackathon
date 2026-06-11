"""
ProcessorAgent — Celery task.
Full implementation in Step 8.

Chain signature
---------------
processor_task(prev_result, job_id, session_id)
prev_result = extractor_task return value.
"""
from __future__ import annotations

from app.celery_config import QUEUE_PROCESSOR, celery_app


@celery_app.task(
    name="app.agents.processor_agent.processor_task",
    bind=True,
    queue=QUEUE_PROCESSOR,
    max_retries=3,
    default_retry_delay=5,
)
def processor_task(
    self,
    prev_result: dict,
    job_id: str,
    session_id: str,
) -> dict:
    """
    Chunk the markdown, generate embeddings, store in MongoDB Atlas.
    prev_result = extractor_task return value.
    Implemented in Step 8.
    """
    raise NotImplementedError("ProcessorAgent not yet implemented (Step 8)")
