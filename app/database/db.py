"""
MongoDB async connection manager (Motor).

Usage
-----
    from app.database.db import db_manager

    # In FastAPI lifespan:
    await db_manager.connect()
    await db_manager.disconnect()

    # Anywhere in the app:
    docs = db_manager.documents          # AsyncIOMotorCollection
    chunks = db_manager.chunks
    ...
"""
from __future__ import annotations

import logging
from typing import Optional

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import OperationFailure, ServerSelectionTimeoutError

from app.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# Collection names (single source of truth)
# ─────────────────────────────────────────────────────────────────
COLLECTION_DOCUMENTS = "documents"
COLLECTION_CHUNKS = "chunks"
COLLECTION_AGENT_MEMORY = "agent_memory"
COLLECTION_JOB_STATUS = "job_status"
COLLECTION_QUERY_LOGS = "query_logs"


class DatabaseManager:
    """Async MongoDB connection manager backed by Motor."""

    def __init__(self) -> None:
        self._client: Optional[AsyncIOMotorClient] = None
        self._db: Optional[AsyncIOMotorDatabase] = None

    # ── connection lifecycle ───────────────────────────────────────

    async def connect(self) -> None:
        """Open the Motor client and create all indexes."""
        logger.info("🔌 Connecting to MongoDB Atlas...")
        self._client = AsyncIOMotorClient(
            settings.mongodb_uri,
            serverSelectionTimeoutMS=10_000,
            connectTimeoutMS=10_000,
            socketTimeoutMS=30_000,
            maxPoolSize=50,
            minPoolSize=5,
        )
        self._db = self._client[settings.mongodb_database]

        # Validate the connection before proceeding
        await self._ping()
        logger.info(f"✅ Connected to database: '{settings.mongodb_database}'")

        await self._create_indexes()

    async def disconnect(self) -> None:
        """Close the Motor client gracefully."""
        if self._client:
            self._client.close()
            self._client = None
            self._db = None
            logger.info("🔒 MongoDB connection closed.")

    # ── collection properties ──────────────────────────────────────

    def _get_collection(self, name: str) -> AsyncIOMotorCollection:
        if self._db is None:
            raise RuntimeError("Database not connected. Call connect() first.")
        return self._db[name]

    @property
    def documents(self) -> AsyncIOMotorCollection:
        return self._get_collection(COLLECTION_DOCUMENTS)

    @property
    def chunks(self) -> AsyncIOMotorCollection:
        return self._get_collection(COLLECTION_CHUNKS)

    @property
    def agent_memory(self) -> AsyncIOMotorCollection:
        return self._get_collection(COLLECTION_AGENT_MEMORY)

    @property
    def job_status(self) -> AsyncIOMotorCollection:
        return self._get_collection(COLLECTION_JOB_STATUS)

    @property
    def query_logs(self) -> AsyncIOMotorCollection:
        return self._get_collection(COLLECTION_QUERY_LOGS)

    # ── health check ───────────────────────────────────────────────

    async def health_check(self) -> dict:
        """Return a dict indicating MongoDB connectivity status."""
        try:
            await self._ping()
            return {"mongodb": "healthy", "database": settings.mongodb_database}
        except Exception as exc:
            logger.error(f"MongoDB health check failed: {exc}")
            return {"mongodb": "unhealthy", "error": str(exc)}

    # ── internal helpers ───────────────────────────────────────────

    async def _ping(self) -> None:
        """Raise if the server is unreachable."""
        try:
            await self._client.admin.command("ping")  # type: ignore[union-attr]
        except ServerSelectionTimeoutError as exc:
            raise RuntimeError(
                "Cannot reach MongoDB Atlas. Check MONGODB_URI and network access."
            ) from exc

    async def _create_indexes(self) -> None:
        """Create all regular indexes. Vector search index is handled separately."""
        logger.info("📐 Creating MongoDB indexes...")

        await self._create_documents_indexes()
        await self._create_chunks_indexes()
        await self._create_agent_memory_indexes()
        await self._create_job_status_indexes()
        await self._create_query_logs_indexes()
        await self._create_vector_search_index()

        logger.info("✅ All indexes ready.")

    async def _create_documents_indexes(self) -> None:
        await self.documents.create_index(
            [("source_url", ASCENDING)],
            name="idx_documents_source_url",
            unique=True,           # one document per URL per session
            sparse=True,
        )
        await self.documents.create_index(
            [("timestamp", DESCENDING)],
            name="idx_documents_timestamp",
        )
        await self.documents.create_index(
            [("session_id", ASCENDING), ("status", ASCENDING)],
            name="idx_documents_session_status",
        )
        logger.debug("  ✓ documents indexes")

    async def _create_chunks_indexes(self) -> None:
        await self.chunks.create_index(
            [("document_id", ASCENDING)],
            name="idx_chunks_document_id",
        )
        await self.chunks.create_index(
            [("session_id", ASCENDING)],
            name="idx_chunks_session_id",
        )
        await self.chunks.create_index(
            [("document_id", ASCENDING), ("chunk_index", ASCENDING)],
            name="idx_chunks_document_order",
        )
        logger.debug("  ✓ chunks indexes")

    async def _create_agent_memory_indexes(self) -> None:
        await self.agent_memory.create_index(
            [("session_id", ASCENDING)],
            name="idx_agent_memory_session_id",
            unique=True,
        )
        await self.agent_memory.create_index(
            [("session_id", ASCENDING), ("agent_name", ASCENDING)],
            name="idx_agent_memory_session_agent",
        )
        logger.debug("  ✓ agent_memory indexes")

    async def _create_job_status_indexes(self) -> None:
        await self.job_status.create_index(
            [("job_id", ASCENDING), ("agent_name", ASCENDING)],
            name="idx_job_status_job_agent",
            unique=True,
        )
        await self.job_status.create_index(
            [("agent_name", ASCENDING)],
            name="idx_job_status_agent_name",
        )
        await self.job_status.create_index(
            [("status", ASCENDING)],
            name="idx_job_status_status",
        )
        await self.job_status.create_index(
            [("session_id", ASCENDING), ("status", ASCENDING)],
            name="idx_job_status_session_status",
        )
        logger.debug("  ✓ job_status indexes")

    async def _create_query_logs_indexes(self) -> None:
        await self.query_logs.create_index(
            [("session_id", ASCENDING)],
            name="idx_query_logs_session_id",
        )
        await self.query_logs.create_index(
            [("timestamp", DESCENDING)],
            name="idx_query_logs_timestamp",
        )
        await self.query_logs.create_index(
            [("session_id", ASCENDING), ("timestamp", DESCENDING)],
            name="idx_query_logs_session_timestamp",
        )
        logger.debug("  ✓ query_logs indexes")

    async def _create_vector_search_index(self) -> None:
        """
        Create the Atlas Vector Search index on chunks.embedding.

        Atlas Vector Search indexes must be created via the Atlas Data API
        or the Atlas Admin API — they cannot be created via the regular
        createIndex driver command. We attempt the driver command anyway
        (works in Atlas M10+ with the `$vectorSearch` operator) and skip
        gracefully if the cluster plan doesn't support it.

        The index definition below matches the Atlas Search index JSON:

            {
              "mappings": {
                "dynamic": false,
                "fields": {
                  "embedding": {
                    "type": "knnVector",
                    "dimensions": 1536,
                    "similarity": "cosine"
                  }
                }
              }
            }
        """
        index_name = settings.vector_index_name
        try:
            # Check if it already exists to avoid duplicate errors
            existing = await self._db[COLLECTION_CHUNKS].list_search_indexes().to_list(None)  # type: ignore
            existing_names = [idx.get("name") for idx in existing]

            if index_name in existing_names:
                logger.debug(f"  ✓ vector search index '{index_name}' already exists")
                return

            await self._db[COLLECTION_CHUNKS].create_search_index(  # type: ignore
                {
                    "name": index_name,
                    "definition": {
                        "mappings": {
                            "dynamic": False,
                            "fields": {
                                "embedding": {
                                    "type": "knnVector",
                                    "dimensions": settings.vector_dimensions,
                                    "similarity": "cosine",
                                }
                            },
                        }
                    },
                }
            )
            logger.info(f"  ✓ vector search index '{index_name}' created")

        except OperationFailure as exc:
            # Free-tier (M0) or shared clusters don't support Atlas Search
            logger.warning(
                f"  ⚠️  Could not create vector search index '{index_name}': {exc.details}. "
                "Create it manually in the Atlas UI under Search → Create Index."
            )
        except Exception as exc:
            logger.warning(
                f"  ⚠️  Vector search index creation skipped: {exc}. "
                "Create it manually in the Atlas UI."
            )


# ── singleton ──────────────────────────────────────────────────────
db_manager = DatabaseManager()
