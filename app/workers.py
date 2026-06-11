"""
Celery worker entry-point.

All agent task modules are imported here so that when a worker process
starts (via `celery -A app.workers worker ...`), every @celery_app.task
decorator is registered before the worker starts consuming messages.

Usage
-----
Start all queues on one worker (dev):
    celery -A app.workers worker --loglevel=info \
        --queues=scanner_queue,evaluator_queue,extractor_queue,processor_queue

Start a dedicated scanner worker (production):
    celery -A app.workers worker --loglevel=info \
        --queues=scanner_queue --concurrency=4 --hostname=scanner@%h

Monitor with Flower:
    celery -A app.workers flower --port=5555
"""
# ── import the configured Celery app ──────────────────────────────
from app.celery_config import celery_app  # noqa: F401  (re-exported for CLI)

# ── register all agent task modules ───────────────────────────────
# These imports trigger the @celery_app.task decorators inside each module.
# Stub files exist for Steps 5-8; replace with real implementations later.

try:
    import app.agents.scanner_agent    # noqa: F401
except ImportError:
    pass  # not yet implemented

try:
    import app.agents.evaluator_agent  # noqa: F401
except ImportError:
    pass

try:
    import app.agents.extractor_agent  # noqa: F401
except ImportError:
    pass

try:
    import app.agents.processor_agent  # noqa: F401
except ImportError:
    pass
