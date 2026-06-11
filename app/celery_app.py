"""
Celery application configuration for parallel agent execution
"""
from celery import Celery
from app.config import settings

# Initialize Celery
celery_app = Celery(
    "voicedoc_intelligence",
    broker=settings.celery_broker_url,
    backend=settings.celery_result_backend,
    include=[
        "app.agents.scanner",
        "app.agents.evaluator",
        "app.agents.extractor",
        "app.agents.processor",
    ],
)

# Celery configuration
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minutes
    task_soft_time_limit=240,  # 4 minutes
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=50,
)

# Task routes (optional)
celery_app.conf.task_routes = {
    "app.agents.scanner.*": {"queue": "scanner"},
    "app.agents.evaluator.*": {"queue": "evaluator"},
    "app.agents.extractor.*": {"queue": "extractor"},
    "app.agents.processor.*": {"queue": "processor"},
}
