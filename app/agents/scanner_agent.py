"""
ScannerAgent — Celery worker task (Step 5).

Responsibilities
----------------
1. Accept job_id + search_queries from SupervisorAgent
2. Run DuckDuckGo searches to discover candidate URLs
3. Accept any direct URLs injected by the user
4. Deduplicate and basic-filter low-quality sources
5. Ask Gemini to score each URL for relevance (0-10)
6. Persist every found URL to MongoDB `documents` collection
7. Update ScannerAgent progress in `job_status`
8. Broadcast progress over WebSocket
9. Return a structured result dict — Celery chains it to EvaluatorAgent

Chain position
--------------
    scanner_task.s(job_id, session_id, query, search_queries)
        → evaluator_task(prev_result, job_id, session_id)

Return shape
------------
{
    "job_id":      str,
    "session_id":  str,
    "query":       str,
    "urls_found":  int,
    "urls": [
        {"url": str, "title": str, "snippet": str, "relevance_score": float}
    ]
}
"""
from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

from celery.exceptions import SoftTimeLimitExceeded
from duckduckgo_search import DDGS
from langchain_core.messages import HumanMessage

from app.celery_config import QUEUE_SCANNER, celery_app
from app.config import settings
from app.models.job_status import JobState
from app.utils.job_manager import update_job_progress_sync
from app.utils.llm import get_llm_sync

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME = "ScannerAgent"

# DuckDuckGo results per query
DDG_MAX_RESULTS = 8

# Minimum Gemini relevance score (0-10) to keep a URL
RELEVANCE_THRESHOLD = 4.0

# Domains that are almost always low-quality for research
_BLOCKLIST_DOMAINS = {
    "pinterest.com", "instagram.com", "facebook.com", "twitter.com",
    "tiktok.com", "reddit.com", "quora.com", "answers.yahoo.com",
    "amazon.com", "ebay.com", "etsy.com",
}

# ── helpers ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().lstrip("www.")
    except Exception:
        return ""


def _is_valid_url(url: str) -> bool:
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and bool(parsed.netloc)
    except Exception:
        return False


def _deduplicate(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Remove duplicate URLs, keeping the first occurrence."""
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for c in candidates:
        url = c["url"].rstrip("/").lower()
        if url not in seen:
            seen.add(url)
            out.append(c)
    return out


def _filter_blocklist(candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop URLs from known low-quality domains."""
    return [
        c for c in candidates
        if _extract_domain(c["url"]) not in _BLOCKLIST_DOMAINS
    ]


# ── DuckDuckGo search ──────────────────────────────────────────────

def _ddg_search(query: str, max_results: int = DDG_MAX_RESULTS) -> List[Dict[str, Any]]:
    """
    Run a DuckDuckGo text search and normalise results.
    Returns [{"url", "title", "snippet"}]
    """
    try:
        with DDGS() as ddgs:
            results = ddgs.text(
                keywords=query,
                region="wt-wt",
                safesearch="moderate",
                max_results=max_results,
            )
        return [
            {
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
            if r.get("href") and _is_valid_url(r.get("href", ""))
        ]
    except Exception as exc:
        logger.warning(f"DDG search failed for '{query}': {exc}")
        return []


# ── LLM scoring ───────────────────────────────────────────────────

def _score_urls_with_llm(
    query: str,
    candidates: list,
) -> list:
    """
    Ask Gemini to rate each candidate URL's relevance to the query (0-10).
    Returns candidates with a `relevance_score` field added.

    Falls back to score=5.0 for all if Gemini fails.
    """
    if not candidates:
        return candidates

    # Build a compact list for the prompt (avoid huge context)
    url_list = "\n".join(
        f'{i+1}. URL: {c["url"]}\n   Title: {c["title"]}\n   Snippet: {c["snippet"][:200]}'
        for i, c in enumerate(candidates)
    )

    prompt = f"""You are evaluating web URLs for relevance to a research query.

Query: "{query}"

Rate each URL's relevance on a scale of 0-10:
- 0-3: Irrelevant or very low quality
- 4-6: Somewhat relevant
- 7-9: Highly relevant
- 10:  Perfect match (authoritative primary source)

URLs to evaluate:
{url_list}

Return ONLY a JSON array of numbers (one score per URL, in the same order):
[score1, score2, ...]

No explanation, no markdown, just the JSON array."""

    try:
        model = get_llm_sync(temperature=0.1, max_tokens=512)
        # Synchronous invoke — we're inside a Celery worker thread
        response = model.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences if present
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        scores = json.loads(raw)
        if not isinstance(scores, list):
            raise ValueError("Expected a JSON array")

        for i, candidate in enumerate(candidates):
            try:
                candidate["relevance_score"] = float(scores[i])
            except (IndexError, TypeError, ValueError):
                candidate["relevance_score"] = 5.0

        logger.info(f"  LLM scored {len(candidates)} URLs for query '{query[:60]}'")

    except Exception as exc:
        logger.warning(f"LLM URL scoring failed, using default scores: {exc}")
        for c in candidates:
            c.setdefault("relevance_score", 5.0)

    return candidates


# ── MongoDB helpers (sync via new event loop) ─────────────────────

def _save_documents_to_mongo(
    candidates: List[Dict[str, Any]],
    job_id: str,
    session_id: str,
    query: str,
) -> None:
    """
    Upsert each URL as a document record in MongoDB.
    Uses update_one with upsert=True to avoid duplicates.
    """
    import asyncio
    from app.database.db import db_manager
    from app.models.documents import DocumentStatus

    async def _upsert_all():
        for c in candidates:
            doc = {
                "source_url": c["url"],
                "title": c.get("title", ""),
                "status": DocumentStatus.SCANNING.value,
                "session_id": session_id,
                "job_id": job_id,
                "metadata": {
                    "snippet": c.get("snippet", ""),
                    "relevance_score_scanner": c.get("relevance_score", 5.0),
                    "query": query,
                    "discovered_by": AGENT_NAME,
                },
                "relevance_score": None,   # set by EvaluatorAgent
                "raw_content": None,
                "markdown_content": None,
                "chunk_count": 0,
                "error_message": None,
                "timestamp": _now(),
                "updated_at": _now(),
            }
            await db_manager.documents.update_one(
                {"source_url": c["url"], "session_id": session_id},
                {"$setOnInsert": doc},
                upsert=True,
            )

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(_upsert_all())
    finally:
        loop.close()


# ── Main Celery task ───────────────────────────────────────────────

@celery_app.task(
    name="app.agents.scanner_agent.scanner_task",
    bind=True,
    queue=QUEUE_SCANNER,
    max_retries=3,
    default_retry_delay=5,
)
def scanner_task(
    self,
    job_id: str,
    session_id: str,
    query: str,
    search_queries: list,
    direct_urls: Optional[List[str]] = None,
) -> dict:
    """
    Scan the web for documents relevant to the given search queries.

    Parameters
    ----------
    job_id         : UUID from SupervisorAgent
    session_id     : browser session ID
    query          : the original user query (for context / scoring)
    search_queries : list of DDG search strings from the plan
    direct_urls    : optional list of URLs the user explicitly provided

    Returns
    -------
    dict  — forwarded as prev_result to evaluator_task
    """
    worker_id = self.request.hostname
    logger.info(f"🔍 ScannerAgent started | job_id={job_id} | queries={len(search_queries)}")

    # ── mark started ──────────────────────────────────────────────
    update_job_progress_sync(
        job_id=job_id,
        agent_name=AGENT_NAME,
        status=JobState.STARTED,
        progress=5,
        worker_id=worker_id,
    )

    try:
        all_candidates: List[Dict[str, Any]] = []

        # ── step 1: inject direct URLs (progress 5→15) ────────────
        if direct_urls:
            for url in direct_urls:
                if _is_valid_url(url):
                    all_candidates.append({
                        "url": url,
                        "title": "",
                        "snippet": "User-provided URL",
                        "source": "direct",
                    })
            logger.info(f"  {len(direct_urls)} direct URLs accepted")

        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.RUNNING, progress=15, worker_id=worker_id,
        )

        # ── step 2: DuckDuckGo search (progress 15→50) ────────────
        total_queries = len(search_queries)
        for idx, sq in enumerate(search_queries, 1):
            results = _ddg_search(sq)
            for r in results:
                r["source"] = "ddg"
            all_candidates.extend(results)
            logger.info(f"  DDG [{idx}/{total_queries}] '{sq[:60]}' → {len(results)} results")

            # Incremental progress within the search phase
            search_progress = 15 + int((idx / total_queries) * 35)
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.RUNNING, progress=search_progress, worker_id=worker_id,
            )

        # ── step 3: dedup + blocklist filter (progress 50→55) ─────
        before_filter = len(all_candidates)
        all_candidates = _deduplicate(all_candidates)
        all_candidates = _filter_blocklist(all_candidates)
        all_candidates = [c for c in all_candidates if _is_valid_url(c["url"])]
        logger.info(
            f"  Filter: {before_filter} → {len(all_candidates)} candidates "
            f"(removed {before_filter - len(all_candidates)})"
        )

        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.RUNNING, progress=55, worker_id=worker_id,
        )

        # ── step 4: LLM relevance scoring (progress 55→80) ───────
        all_candidates = _score_urls_with_llm(query, all_candidates)

        # Keep only URLs above the relevance threshold
        before_threshold = len(all_candidates)
        qualified = [
            c for c in all_candidates
            if c.get("relevance_score", 0) >= RELEVANCE_THRESHOLD
        ]

        # Always keep at least 3 if we have them (even below threshold)
        if len(qualified) < 3 and len(all_candidates) >= 3:
            qualified = sorted(
                all_candidates,
                key=lambda x: x.get("relevance_score", 0),
                reverse=True,
            )[:3]

        logger.info(
            f"  Scoring: {before_threshold} → {len(qualified)} URLs "
            f"above threshold {RELEVANCE_THRESHOLD}"
        )

        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.RUNNING, progress=80, worker_id=worker_id,
        )

        # ── step 5: persist to MongoDB (progress 80→95) ───────────
        if qualified:
            _save_documents_to_mongo(qualified, job_id, session_id, query)
            logger.info(f"  Saved {len(qualified)} document stubs to MongoDB")

        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.RUNNING, progress=95, worker_id=worker_id,
        )

        # ── step 6: build return payload ──────────────────────────
        result = {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "urls_found": len(qualified),
            "urls": [
                {
                    "url": c["url"],
                    "title": c.get("title", ""),
                    "snippet": c.get("snippet", ""),
                    "relevance_score": round(c.get("relevance_score", 5.0), 2),
                }
                for c in qualified
            ],
        }

        # ── mark complete ──────────────────────────────────────────
        update_job_progress_sync(
            job_id=job_id,
            agent_name=AGENT_NAME,
            status=JobState.SUCCESS,
            progress=100,
            result={"urls_found": len(qualified)},
            worker_id=worker_id,
        )

        logger.info(
            f"✅ ScannerAgent done | job_id={job_id} | urls_found={len(qualified)}"
        )
        return result

    except SoftTimeLimitExceeded:
        msg = "ScannerAgent hit the 4-minute soft time limit"
        logger.error(f"⏱️  {msg} | job_id={job_id}")
        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.FAILURE, progress=0,
            error=msg, worker_id=worker_id,
        )
        raise

    except Exception as exc:
        msg = str(exc)
        logger.error(f"❌ ScannerAgent failed | job_id={job_id} | error={msg}", exc_info=True)

        # Retry with exponential backoff
        try:
            raise self.retry(
                exc=exc,
                countdown=5 * (2 ** self.request.retries),  # 5s, 10s, 20s
            )
        except self.MaxRetriesExceededError:
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.FAILURE, progress=0,
                error=f"Max retries exceeded: {msg}", worker_id=worker_id,
            )
            # Return empty result so the chain can still continue gracefully
            return {
                "job_id": job_id,
                "session_id": session_id,
                "query": query,
                "urls_found": 0,
                "urls": [],
                "error": msg,
            }
