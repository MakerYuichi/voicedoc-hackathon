"""
FastAPI application entry point
"""
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import settings
from app.database.db import db_manager
from app.utils.job_manager import redis_health_check

# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    logger.info("🚀 Starting VoiceDoc Intelligence...")
    logger.info(f"Environment : {settings.app_env}")
    logger.info(f"Gemini Model: {settings.gemini_model}")

    # ── startup ────────────────────────────────────────────────────
    await db_manager.connect()

    redis_status = await redis_health_check()
    if redis_status.get("redis") == "healthy":
        logger.info("✅ Redis connected")
    else:
        logger.warning(f"⚠️  Redis unavailable: {redis_status.get('error')} — Celery workers won't start")

    yield

    # ── shutdown ───────────────────────────────────────────────────
    await db_manager.disconnect()
    logger.info("🛑 VoiceDoc Intelligence stopped.")


# ── app factory ────────────────────────────────────────────────────
app = FastAPI(
    title="VoiceDoc Intelligence",
    description="Voice-Commanded Document Intelligence Multi-Agent System",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── routes ─────────────────────────────────────────────────────────

@app.get("/", tags=["meta"])
async def root():
    return {
        "message": "VoiceDoc Intelligence API",
        "version": "0.1.0",
        "status": "operational",
        "docs": "/docs",
    }


@app.get("/health", tags=["meta"])
async def health_check():
    """
    Returns the health of the API and its dependencies.
    Used by Docker HEALTHCHECK and Cloud Run liveness probes.
    """
    db_health = await db_manager.health_check()
    all_healthy = db_health.get("mongodb") == "healthy"

    return JSONResponse(
        content={
            "status": "healthy" if all_healthy else "degraded",
            "environment": settings.app_env,
            "model": settings.gemini_model,
            "dependencies": {
                **db_health,
                **(await redis_health_check()),
            },
        },
        status_code=200 if all_healthy else 503,
    )


# TODO Step 10: include API routers
# from app.api import process, query, websocket
# app.include_router(process.router)
# app.include_router(query.router)
# app.include_router(websocket.router)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_env == "development",
    )
