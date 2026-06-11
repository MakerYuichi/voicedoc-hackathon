"""
VoiceDoc Intelligence — FastAPI application entry point.
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.db import db_manager
from app.utils.job_manager import redis_health_check
from app.utils.llm import llm_health_check, active_provider

# Configure logging before anything else
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# ── lifespan ───────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Connect / disconnect all external resources on startup / shutdown."""
    logger.info("🚀 Starting VoiceDoc Intelligence...")
    logger.info(f"  env         : {settings.app_env}")
    logger.info(f"  LLM         : {active_provider()} ({settings.gemini_model})")
    logger.info(f"  embeddings  : {settings.embedding_model}")

    # MongoDB
    await db_manager.connect()

    # Redis (warn-only — app still starts if Redis is down)
    redis_status = await redis_health_check()
    if redis_status.get("redis") == "healthy":
        logger.info("✅ Redis connected")
    else:
        logger.warning(
            f"⚠️  Redis unavailable: {redis_status.get('error')} "
            "— Celery workers won't process tasks until Redis is reachable"
        )

    yield

    await db_manager.disconnect()
    logger.info("🛑 VoiceDoc Intelligence stopped.")


# ── app factory ────────────────────────────────────────────────────

app = FastAPI(
    title="VoiceDoc Intelligence",
    description=(
        "Voice-Commanded Document Intelligence Multi-Agent System.\n\n"
        "**Hackathon**: Google Cloud Rapid Agent Hackathon\n"
        "**LLM**: Gemini 2.0 Flash · **Orchestration**: LangGraph · "
        "**Storage**: MongoDB Atlas"
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS — allow the frontend dev server and the API itself
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routers ────────────────────────────────────────────────────────

from app.api import process_router, query_router, websocket_router  # noqa: E402

app.include_router(process_router)
app.include_router(query_router)
app.include_router(websocket_router)


# ── meta routes ────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    """API root — confirms the service is running."""
    return {
        "service": "VoiceDoc Intelligence",
        "version": "0.1.0",
        "status":  "operational",
        "docs":    "/docs",
        "endpoints": {
            "process":   "POST /api/process",
            "job_status":"GET  /api/job/{job_id}",
            "query":     "POST /api/query",
            "websocket": "WS   /ws/{session_id}",
            "health":    "GET  /api/health",
        },
    }


@app.get("/api/health", tags=["meta"])
async def health_check():
    """
    Dependency health check used by Docker HEALTHCHECK and Cloud Run
    liveness probes.

    Returns 200 when MongoDB is reachable; 503 otherwise.
    Redis and LLM failures are reported but do not degrade the HTTP status
    (they are non-fatal for serving existing data).
    """
    db_health    = await db_manager.health_check()
    redis_health = await redis_health_check()
    llm_health   = await llm_health_check()

    mongo_ok = db_health.get("mongodb") == "healthy"
    overall  = "healthy" if mongo_ok else "degraded"

    return JSONResponse(
        content={
            "status":      overall,
            "environment": settings.app_env,
            "dependencies": {
                **db_health,
                **redis_health,
                **llm_health,
            },
            "config": {
                "llm_provider":   active_provider(),
                "llm_model":      settings.gemini_model,
                "embedding_model": settings.embedding_model,
                "vector_dims":    settings.vector_dimensions,
                "chunk_size":     settings.document_chunk_size,
            },
        },
        status_code=200 if mongo_ok else 503,
    )


# ── dev entry point ────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
        log_level=settings.log_level.lower(),
    )
