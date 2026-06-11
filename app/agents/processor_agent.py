"""
ProcessorAgent — Celery worker task (Step 8).

Responsibilities
----------------
1. Receive extracted docs from ExtractorAgent (prev_result)
2. For each doc:
   a. Chunk markdown with RecursiveCharacterTextSplitter (512 tokens, 50 overlap)
   b. Embed each chunk via Google text-embedding-004
   c. Bulk-insert chunks into MongoDB `chunks` collection
   d. Update parent document: status=processed, chunk_count=N
3. Mark ProcessorAgent job_status SUCCESS
4. Check whether ALL pipeline agents for this job_id are now done
5. If complete: mark SupervisorAgent job SUCCESS + broadcast WS completion

Chain position
--------------
    extractor_task → processor_task(prev_result, job_id, session_id)
                         (end of chain — no further tasks)

prev_result shape (from extractor)
------------------------------------
{
    "job_id": str, "session_id": str, "query": str,
    "extracted": int,
    "docs": [
        {
            "url": str, "title": str, "content_markdown": str,
            "summary": str, "word_count": int, "score": float,
            "key_topics": [str], "author": str, "date": str,
            "extracted_at": str
        }
    ]
}

Return shape
------------
{
    "job_id": str,
    "session_id": str,
    "query": str,
    "docs_processed": int,
    "total_chunks": int,
    "chunk_ids": [str]   # MongoDB ObjectIds as strings
}
"""
from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from celery.exceptions import SoftTimeLimitExceeded
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.celery_config import QUEUE_PROCESSOR, celery_app
from app.config import settings
from app.models.job_status import JobState
from app.utils.job_manager import (
    broadcast_completion,
    mark_job_complete_sync,
    update_job_progress_sync,
)

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME  = "ProcessorAgent"
PIPELINE_AGENTS = ["ScannerAgent", "EvaluatorAgent", "ExtractorAgent", "ProcessorAgent"]

# Embedding batch size — text-embedding-004 accepts up to 100 texts per call
EMBEDDING_BATCH_SIZE = 50


# ── helpers ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _run_async(coro):
    """Run async coroutine from a sync Celery worker thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── chunking ───────────────────────────────────────────────────────

def _build_splitter() -> RecursiveCharacterTextSplitter:
    """
    RecursiveCharacterTextSplitter with settings from config.

    chunk_size=512 and chunk_overlap=50 are token-approximate values.
    LangChain's default length function is len(text) in characters;
    512 chars ≈ 128-170 tokens for English text, which keeps chunks
    well within embedding model limits.
    """
    return RecursiveCharacterTextSplitter(
        chunk_size=settings.document_chunk_size,      # 512
        chunk_overlap=settings.document_chunk_overlap, # 50
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )


def _chunk_text(text: str) -> List[str]:
    """Split text into overlapping chunks. Returns list of chunk strings."""
    if not text or not text.strip():
        return []
    splitter = _build_splitter()
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c.strip()]


# ── embeddings ─────────────────────────────────────────────────────

def _get_embeddings_model() -> GoogleGenerativeAIEmbeddings:
    """
    Return a GoogleGenerativeAIEmbeddings instance for text-embedding-004.

    Always uses the Google API directly — embeddings are not routed
    through Groq (Groq does not offer embedding endpoints).
    Uses gemini-embedding-001 which outputs 3072-dim vectors.
    """
    return GoogleGenerativeAIEmbeddings(
        model=settings.embedding_model,         # "models/text-embedding-004"
        google_api_key=settings.google_api_key,
        task_type="retrieval_document",         # optimal for RAG storage
    )


def _embed_chunks(chunks: List[str]) -> List[List[float]]:
    """
    Embed a list of text chunks in batches.
    Returns a parallel list of embedding vectors.
    Falls back to zero vectors on error to avoid losing all chunks.
    """
    if not chunks:
        return []

    embedder = _get_embeddings_model()
    all_embeddings: List[List[float]] = []

    for batch_start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[batch_start : batch_start + EMBEDDING_BATCH_SIZE]
        try:
            vectors = embedder.embed_documents(batch)
            all_embeddings.extend(vectors)
            logger.debug(
                f"  Embedded batch [{batch_start}:{batch_start + len(batch)}] "
                f"→ {len(vectors)} vectors, dim={len(vectors[0]) if vectors else 0}"
            )
        except Exception as exc:
            logger.error(f"Embedding batch failed: {exc}")
            # Fallback: zero vectors so storage isn't blocked
            zero_vec = [0.0] * settings.vector_dimensions
            all_embeddings.extend([zero_vec] * len(batch))

    return all_embeddings


# ── MongoDB helpers ────────────────────────────────────────────────

def _get_document_id_sync(url: str, session_id: str) -> Optional[str]:
    """Look up the MongoDB _id for a document by URL + session."""
    from app.database.db import db_manager

    async def _find():
        doc = await db_manager.documents.find_one(
            {"source_url": url, "session_id": session_id},
            {"_id": 1},
        )
        return str(doc["_id"]) if doc else None

    return _run_async(_find())


def _save_chunks_to_mongo(
    chunks_data: List[Dict[str, Any]],
) -> List[str]:
    """
    Bulk-insert chunk documents into MongoDB.
    Returns list of inserted _id strings.
    """
    from app.database.db import db_manager

    async def _insert():
        if not chunks_data:
            return []
        result = await db_manager.chunks.insert_many(chunks_data)
        return [str(oid) for oid in result.inserted_ids]

    return _run_async(_insert())


def _update_document_status(
    url: str,
    session_id: str,
    chunk_count: int,
) -> None:
    """Set document status=processed and record chunk_count."""
    from app.database.db import db_manager
    from app.models.documents import DocumentStatus

    async def _update():
        await db_manager.documents.update_one(
            {"source_url": url, "session_id": session_id},
            {
                "$set": {
                    "status": DocumentStatus.PROCESSED.value,
                    "chunk_count": chunk_count,
                    "updated_at": _now(),
                }
            },
        )

    _run_async(_update())


def _count_successful_agents(job_id: str) -> Dict[str, int]:
    """
    Return counts of agent sub-jobs by status for this job_id.
    Used to decide whether the full pipeline is done.
    """
    from app.database.db import db_manager

    async def _count():
        cursor = db_manager.job_status.find(
            {"job_id": job_id, "agent_name": {"$in": PIPELINE_AGENTS}},
            {"agent_name": 1, "status": 1},
        )
        records = await cursor.to_list(length=20)
        success = sum(1 for r in records if r.get("status") == JobState.SUCCESS.value)
        total   = len(records)
        return {"success": success, "total": total}

    return _run_async(_count())


def _count_total_chunks_for_job(job_id: str) -> int:
    """Count all chunks stored for a job_id."""
    from app.database.db import db_manager

    async def _count():
        return await db_manager.chunks.count_documents({"job_id": job_id})

    return _run_async(_count())


def _count_processed_docs(job_id: str) -> int:
    """Count documents in status=processed for a job_id."""
    from app.database.db import db_manager
    from app.models.documents import DocumentStatus

    async def _count():
        return await db_manager.documents.count_documents({
            "job_id": job_id,
            "status": DocumentStatus.PROCESSED.value,
        })

    return _run_async(_count())


def _broadcast_completion_sync(job_id: str, message: str) -> None:
    """Broadcast job completion over WebSocket from a Celery worker."""
    _run_async(broadcast_completion(job_id=job_id, message=message))


# ── Main Celery task ───────────────────────────────────────────────

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
    Chunk, embed, and store extracted documents in MongoDB Atlas.

    Parameters
    ----------
    prev_result  : extractor_task return dict (contains docs + query)
    job_id       : UUID from SupervisorAgent
    session_id   : browser session ID

    Returns
    -------
    dict — summary of chunks stored
    """
    worker_id = self.request.hostname
    query = prev_result.get("query", "")
    docs  = prev_result.get("docs", [])

    logger.info(
        f"⚙️  ProcessorAgent started | job_id={job_id} | docs={len(docs)}"
    )

    # ── mark started ──────────────────────────────────────────────
    update_job_progress_sync(
        job_id=job_id,
        agent_name=AGENT_NAME,
        status=JobState.STARTED,
        progress=5,
        worker_id=worker_id,
    )

    # Edge case: nothing to process
    if not docs:
        logger.warning(f"ProcessorAgent: no docs to process | job_id={job_id}")
        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.SUCCESS, progress=100,
            result={"docs_processed": 0, "total_chunks": 0},
            worker_id=worker_id,
        )
        _finalise_job(job_id, session_id, query, docs_processed=0)
        return {
            "job_id": job_id, "session_id": session_id, "query": query,
            "docs_processed": 0, "total_chunks": 0, "chunk_ids": [],
        }

    try:
        all_chunk_ids: List[str] = []
        docs_processed = 0
        total = len(docs)

        for idx, doc in enumerate(docs, 1):
            url      = doc.get("url", "")
            title    = doc.get("title", "")
            markdown = doc.get("content_markdown", "")
            score    = doc.get("score", 0.0)
            topics   = doc.get("key_topics", [])
            summary  = doc.get("summary", "")

            # Progress: 5 → 85% during processing
            progress = 5 + int((idx - 1) / total * 80)
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.RUNNING, progress=progress, worker_id=worker_id,
            )

            logger.info(f"  [{idx}/{total}] Processing: {url[:70]}")

            if not markdown:
                logger.warning(f"  No markdown for {url[:60]}, skipping")
                continue

            # ── step A: resolve parent document _id ───────────────
            document_id = _get_document_id_sync(url, session_id)
            if not document_id:
                logger.warning(f"  Document not found in MongoDB: {url[:60]}")
                # Store with url as fallback id so chunks aren't lost
                document_id = url

            # ── step B: chunk ──────────────────────────────────────
            chunks = _chunk_text(markdown)
            if not chunks:
                logger.warning(f"  No chunks produced for {url[:60]}")
                continue

            logger.info(f"  Chunks: {len(chunks)} from {doc.get('word_count', 0)} words")

            # ── step C: embed ──────────────────────────────────────
            embeddings = _embed_chunks(chunks)
            if len(embeddings) != len(chunks):
                logger.error(
                    f"  Embedding count mismatch: {len(embeddings)} vs {len(chunks)}"
                )
                # Pad with zeros if mismatch
                while len(embeddings) < len(chunks):
                    embeddings.append([0.0] * settings.vector_dimensions)

            # ── step D: build chunk documents ──────────────────────
            now = _now()
            chunk_docs = [
                {
                    "document_id": document_id,
                    "job_id": job_id,
                    "session_id": session_id,
                    "source_url": url,
                    "title": title,
                    "content": chunk_text,
                    "embedding": embedding_vec,
                    "chunk_index": chunk_idx,
                    "token_count": len(chunk_text.split()),  # word count approx
                    "metadata": {
                        "relevance_score": score,
                        "key_topics": topics,
                        "summary": summary,
                        "query": query,
                    },
                    "timestamp": now,
                }
                for chunk_idx, (chunk_text, embedding_vec) in enumerate(
                    zip(chunks, embeddings)
                )
            ]

            # ── step E: bulk insert chunks ─────────────────────────
            chunk_ids = _save_chunks_to_mongo(chunk_docs)
            all_chunk_ids.extend(chunk_ids)
            logger.info(f"  Stored {len(chunk_ids)} chunks for '{title[:40]}'")

            # ── step F: mark document processed ───────────────────
            _update_document_status(url, session_id, len(chunk_ids))
            docs_processed += 1

        # ── mark ProcessorAgent complete ───────────────────────────
        update_job_progress_sync(
            job_id=job_id,
            agent_name=AGENT_NAME,
            status=JobState.SUCCESS,
            progress=100,
            result={
                "docs_processed": docs_processed,
                "total_chunks": len(all_chunk_ids),
            },
            worker_id=worker_id,
        )

        logger.info(
            f"✅ ProcessorAgent done | job_id={job_id} "
            f"| docs={docs_processed}/{total} | chunks={len(all_chunk_ids)}"
        )

        # ── finalise job if all pipeline agents succeeded ──────────
        _finalise_job(job_id, session_id, query, docs_processed)

        return {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "docs_processed": docs_processed,
            "total_chunks": len(all_chunk_ids),
            "chunk_ids": all_chunk_ids,
        }

    except SoftTimeLimitExceeded:
        msg = "ProcessorAgent hit the 4-minute soft time limit"
        logger.error(f"⏱️  {msg} | job_id={job_id}")
        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.FAILURE, progress=0,
            error=msg, worker_id=worker_id,
        )
        raise

    except Exception as exc:
        msg = str(exc)
        logger.error(
            f"❌ ProcessorAgent failed | job_id={job_id} | error={msg}",
            exc_info=True,
        )
        try:
            raise self.retry(
                exc=exc,
                countdown=5 * (2 ** self.request.retries),
            )
        except self.MaxRetriesExceededError:
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.FAILURE, progress=0,
                error=f"Max retries exceeded: {msg}", worker_id=worker_id,
            )
            return {
                "job_id": job_id, "session_id": session_id, "query": query,
                "docs_processed": 0, "total_chunks": 0, "chunk_ids": [],
                "error": msg,
            }


# ── Pipeline completion check ──────────────────────────────────────

def _finalise_job(
    job_id: str,
    session_id: str,
    query: str,
    docs_processed: int,
) -> None:
    """
    Check if all four pipeline agents have reached SUCCESS.
    If so, mark the SupervisorAgent job complete and broadcast
    the completion WebSocket message to the frontend.
    """
    counts = _count_successful_agents(job_id)
    logger.info(
        f"  Pipeline check | job_id={job_id} "
        f"| success={counts['success']}/{counts['total']}"
    )

    # All four pipeline agents must be SUCCESS
    if counts["success"] < len(PIPELINE_AGENTS):
        logger.info("  Pipeline not yet complete, waiting for other agents")
        return

    # Count what was stored
    total_chunks = _count_total_chunks_for_job(job_id)
    total_docs   = _count_processed_docs(job_id)

    completion_msg = (
        f"Done! Processed {total_docs} document{'s' if total_docs != 1 else ''}, "
        f"stored {total_chunks} chunk{'s' if total_chunks != 1 else ''} "
        f"from query: \"{query[:80]}\""
    )

    logger.info(f"🏁 {completion_msg}")

    # Mark supervisor job SUCCESS
    mark_job_complete_sync(
        job_id=job_id,
        success=True,
        result={
            "total_docs": total_docs,
            "total_chunks": total_chunks,
            "message": completion_msg,
        },
    )

    # Broadcast final WebSocket message to frontend
    _broadcast_completion_sync(job_id=job_id, message=completion_msg)
