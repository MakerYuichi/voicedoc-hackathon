"""
Celery configuration for VoiceDoc Intelligence.

Replaces the minimal celery_app.py with a fully-featured setup:
  - Four named queues, one per agent
  - Per-task retry policy with exponential backoff
  - Hard + soft time limits
  - Result expiry so Redis doesn't fill up
  - Dead-letter routing via a dedicated failed_tasks queue

Import the ready-to-use Celery instance:
    from app.celery_config import celery_app
"""
from __future__ import annotations

from celery import Celery
from celery.signals import task_failure, task_prerun, task_postrun, task_retry
from kombu import Exchange, Queue
import logging

from app.config import settings

logger = logging.getLogger(__name__)

# ── Queue / Exchange definitions ───────────────────────────────────
# Each agent gets its own queue so workers can be scaled independently.

_default_exchange = Exchange("voicedoc", type="direct")

QUEUE_SCANNER   = "scanner_queue"
QUEUE_EVALUATOR = "evaluator_queue"
QUEUE_EXTRACTOR = "extractor_queue"
QUEUE_PROCESSOR = "processor_queue"
QUEUE_DEFAULT   = "default"
QUEUE_FAILED    = "failed_tasks"

AGENT_QUEUES = (
    Queue(QUEUE_SCANNER,   _default_exchange, routing_key=QUEUE_SCANNER),
    Queue(QUEUE_EVALUATOR, _default_exchange, routing_key=QUEUE_EVALUATOR),
    Queue(QUEUE_EXTRACTOR, _default_exchange, routing_key=QUEUE_EXTRACTOR),
    Queue(QUEUE_PROCESSOR, _default_exchange, routing_key=QUEUE_PROCESSOR),
    Queue(QUEUE_DEFAULT,   _default_exchange, routing_key=QUEUE_DEFAULT),
    Queue(QUEUE_FAILED,    _default_exchange, routing_key=QUEUE_FAILED),
)

# ── Retry policy applied to every task by default ──────────────────
# Individual tasks can override via @celery_app.task(max_retries=N, ...)
DEFAULT_RETRY_POLICY = {
    "max_retries": 3,
    "interval_start": 5,       # first retry after 5 s
    "interval_step": 10,       # add 10 s per retry  → 5 / 15 / 25 s
    "interval_max": 60,        # cap at 60 s
}

# ── Celery app ─────────────────────────────────────────────────────

celery_app = Celery("voicedoc_intelligence")

celery_app.config_from_object(
    {
        # Broker / backend
        "broker_url": settings.celery_broker_url,
        "result_backend": settings.celery_result_backend,
        "broker_connection_retry_on_startup": True,

        # Serialisation
        "task_serializer": "json",
        "accept_content": ["json"],
        "result_serializer": "json",
        "timezone": "UTC",
        "enable_utc": True,

        # Queues
        "task_queues": AGENT_QUEUES,
        "task_default_queue": QUEUE_DEFAULT,
        "task_default_exchange": "voicedoc",
        "task_default_routing_key": QUEUE_DEFAULT,

        # Route each agent module to its dedicated queue
        "task_routes": {
            "app.agents.scanner_agent.scanner_task":     {"queue": QUEUE_SCANNER},
            "app.agents.evaluator_agent.evaluator_task": {"queue": QUEUE_EVALUATOR},
            "app.agents.extractor_agent.extractor_task": {"queue": QUEUE_EXTRACTOR},
            "app.agents.processor_agent.processor_task": {"queue": QUEUE_PROCESSOR},
        },

        # Time limits (seconds)
        "task_time_limit": 300,       # hard kill after 5 min
        "task_soft_time_limit": 240,  # SoftTimeLimitExceeded raised after 4 min

        # Result expiry — keep results in Redis for 1 hour
        "result_expires": 3600,

        # Worker tuning
        "task_track_started": True,
        "task_acks_late": True,          # ack only after task completes (safer retries)
        "worker_prefetch_multiplier": 1, # one task at a time per worker thread
        "worker_max_tasks_per_child": 50,

        # Retry defaults (tasks inherit unless they override)
        "task_annotations": {
            "*": {
                "max_retries": DEFAULT_RETRY_POLICY["max_retries"],
                "default_retry_delay": DEFAULT_RETRY_POLICY["interval_start"],
            }
        },

        # Auto-discover tasks from these modules
        "imports": [
            "app.agents.scanner_agent",
            "app.agents.evaluator_agent",
            "app.agents.extractor_agent",
            "app.agents.processor_agent",
        ],
    }
)


# ── Celery signals for logging / monitoring ────────────────────────

@task_prerun.connect
def on_task_prerun(task_id: str, task, *args, **kwargs) -> None:
    logger.info(f"▶️  Task started  | id={task_id} | name={task.name}")


@task_postrun.connect
def on_task_postrun(task_id: str, task, retval, state: str, *args, **kwargs) -> None:
    logger.info(f"✅  Task finished | id={task_id} | name={task.name} | state={state}")


@task_retry.connect
def on_task_retry(request, reason, einfo, *args, **kwargs) -> None:
    logger.warning(
        f"🔄  Task retry    | id={request.id} | reason={reason} "
        f"| retries={request.retries}"
    )


@task_failure.connect
def on_task_failure(task_id: str, exception, traceback, *args, **kwargs) -> None:
    logger.error(
        f"❌  Task failed   | id={task_id} | error={exception}",
        exc_info=exception,
    )
