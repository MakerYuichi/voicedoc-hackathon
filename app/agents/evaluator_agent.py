"""
EvaluatorAgent — Celery worker task (Step 6).

Responsibilities
----------------
1. Receive prev_result from ScannerAgent (list of candidate URLs)
2. For each URL: fetch page content via httpx (30s timeout)
3. Extract readable text with BeautifulSoup
4. Use LLM to score relevance 0-10 against the original query
5. Documents scoring >= 6 → trigger ExtractorAgent
6. Documents scoring < 6  → mark FAILED (rejected) in MongoDB
7. Update job_status progress and broadcast over WebSocket

Chain position
--------------
    scanner_task → evaluator_task(prev_result, job_id, session_id)
                       → extractor_task(prev_result, job_id, session_id)

prev_result shape (from scanner)
---------------------------------
{
    "job_id": str,
    "session_id": str,
    "query": str,
    "urls_found": int,
    "urls": [{"url": str, "title": str, "snippet": str, "relevance_score": float}]
}

Return shape (passed to extractor)
------------------------------------
{
    "job_id": str,
    "session_id": str,
    "query": str,
    "evaluated": int,           # total URLs processed
    "accepted": int,            # URLs that passed threshold
    "rejected": int,
    "docs": [
        {
            "url": str,
            "title": str,
            "raw_html": str,    # full fetched HTML (for ExtractorAgent)
            "text_preview": str,# first 500 chars of clean text
            "score": float,     # LLM score 0-10
            "reason": str,
            "key_topics": [str],
            "should_extract": bool,
            "word_count": int,
            "content_type": str,
        }
    ]
}
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import httpx
from bs4 import BeautifulSoup
from celery.exceptions import SoftTimeLimitExceeded
from langchain_core.messages import HumanMessage

from app.celery_config import QUEUE_EVALUATOR, celery_app
from app.models.job_status import JobState
from app.utils.job_manager import update_job_progress_sync
from app.utils.llm import get_llm_sync

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME = "EvaluatorAgent"

# Minimum LLM score (0-10) to pass a document to ExtractorAgent
ACCEPT_THRESHOLD = 6.0

# httpx fetch settings
FETCH_TIMEOUT_S = 30
MAX_CONTENT_BYTES = 2_000_000   # 2 MB hard cap

# Text fed to the LLM (avoid huge prompts)
MAX_TEXT_FOR_LLM = 3_000        # chars

# Minimum word count to bother evaluating (skip stubs / error pages)
MIN_WORD_COUNT = 100

# Request headers that mimic a real browser to reduce bot-blocks
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── helpers ────────────────────────────────────────────────────────

def _now() -> datetime:
    return datetime.now(timezone.utc)


def _run_async(coro):
    """Run an async coroutine from a sync Celery worker thread."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── fetch ──────────────────────────────────────────────────────────

async def _fetch_url(url: str) -> Dict[str, Any]:
    """
    Async fetch of a URL with a 30-second timeout.

    Returns
    -------
    {"html": str, "content_type": str, "status_code": int, "error": str|None}
    """
    try:
        async with httpx.AsyncClient(
            headers=_HEADERS,
            timeout=FETCH_TIMEOUT_S,
            follow_redirects=True,
            max_redirects=5,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

            content_type = response.headers.get("content-type", "").lower()

            # Only process HTML pages (skip PDFs, images, etc.)
            if "text/html" not in content_type and "text/plain" not in content_type:
                return {
                    "html": "",
                    "content_type": content_type,
                    "status_code": response.status_code,
                    "error": f"Unsupported content-type: {content_type}",
                }

            # Cap content size
            html = response.text[:MAX_CONTENT_BYTES]
            return {
                "html": html,
                "content_type": content_type,
                "status_code": response.status_code,
                "error": None,
            }

    except httpx.TimeoutException:
        return {"html": "", "content_type": "", "status_code": 0,
                "error": f"Timeout after {FETCH_TIMEOUT_S}s"}
    except httpx.HTTPStatusError as exc:
        return {"html": "", "content_type": "", "status_code": exc.response.status_code,
                "error": f"HTTP {exc.response.status_code}"}
    except Exception as exc:
        return {"html": "", "content_type": "", "status_code": 0,
                "error": str(exc)[:200]}


def _fetch_url_sync(url: str) -> Dict[str, Any]:
    return _run_async(_fetch_url(url))


# ── text extraction ────────────────────────────────────────────────

def _extract_text(html: str, title_hint: str = "") -> Dict[str, Any]:
    """
    Use BeautifulSoup to extract clean readable text from HTML.

    Returns
    -------
    {"text": str, "title": str, "word_count": int}
    """
    if not html:
        return {"text": "", "title": title_hint, "word_count": 0}

    try:
        soup = BeautifulSoup(html, "lxml")

        # Remove noise tags
        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript", "iframe", "ads"]):
            tag.decompose()

        # Title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else title_hint

        # Main content heuristic: prefer <article>, <main>, <section>
        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"id": re.compile(r"content|main|article", re.I)})
            or soup.find("body")
            or soup
        )

        text = main.get_text(separator=" ", strip=True)

        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        word_count = len(text.split())

        return {"text": text, "title": title, "word_count": word_count}

    except Exception as exc:
        logger.warning(f"BeautifulSoup extraction error: {exc}")
        return {"text": "", "title": title_hint, "word_count": 0}


# ── LLM evaluation ────────────────────────────────────────────────

def _evaluate_with_llm(
    url: str,
    title: str,
    text: str,
    query: str,
) -> Dict[str, Any]:
    """
    Ask the LLM to score the document for relevance to the query.

    Returns
    -------
    {"score": float, "reason": str, "key_topics": list, "should_extract": bool}
    Falls back to {"score": 5.0, ...} if the LLM call fails.
    """
    # Truncate text to avoid huge prompts
    text_snippet = text[:MAX_TEXT_FOR_LLM]

    prompt = f"""You are evaluating a web document for relevance to a research query.

Query: "{query}"

Document URL: {url}
Document Title: {title}
Document Text (first {MAX_TEXT_FOR_LLM} chars):
---
{text_snippet}
---

Score this document on the following criteria (total score 0-10):
1. Relevance to query         (0-4 pts): Does it directly address the query topic?
2. Content quality            (0-2 pts): Is it substantive? Not spam/ads/thin content?
3. Information density        (0-2 pts): Does it contain useful facts, data, or analysis?
4. Source credibility         (0-2 pts): Is the source authoritative (academic, official, reputable)?

Return ONLY valid JSON, no markdown, no explanation:
{{
  "score": <float 0.0-10.0>,
  "reason": "<one sentence explaining the score>",
  "key_topics": ["<topic1>", "<topic2>", "<topic3>"],
  "should_extract": <true if score >= 6, else false>
}}"""

    try:
        llm = get_llm_sync(temperature=0.1, max_tokens=300)
        response = llm.invoke([HumanMessage(content=prompt)])
        raw = response.content.strip()

        # Strip markdown fences
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()

        # Extract first JSON object if extra text sneaks in
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            raw = match.group()

        result = json.loads(raw)

        # Normalise and validate
        score = float(result.get("score", 5.0))
        score = max(0.0, min(10.0, score))
        should_extract = score >= ACCEPT_THRESHOLD

        return {
            "score": round(score, 1),
            "reason": str(result.get("reason", ""))[:300],
            "key_topics": [str(t) for t in result.get("key_topics", [])[:5]],
            "should_extract": should_extract,
        }

    except Exception as exc:
        logger.warning(f"LLM evaluation failed for {url}: {exc}")
        return {
            "score": 5.0,
            "reason": f"LLM evaluation unavailable: {str(exc)[:100]}",
            "key_topics": [],
            "should_extract": True,   # default to accepting on error
        }


# ── MongoDB update (sync) ──────────────────────────────────────────

def _update_document_in_mongo(
    url: str,
    session_id: str,
    update_fields: Dict[str, Any],
) -> None:
    """Upsert evaluation result into MongoDB documents collection."""
    from app.database.db import db_manager

    async def _update():
        await db_manager.documents.update_one(
            {"source_url": url, "session_id": session_id},
            {"$set": {**update_fields, "updated_at": _now()}},
            upsert=False,
        )

    _run_async(_update())


# ── Main Celery task ───────────────────────────────────────────────

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
    Evaluate each URL from the ScannerAgent for content quality and relevance.

    Parameters
    ----------
    prev_result  : scanner_task return dict (contains urls list + query)
    job_id       : UUID from SupervisorAgent
    session_id   : browser session ID

    Returns
    -------
    dict — forwarded as prev_result to extractor_task
    """
    worker_id = self.request.hostname
    query = prev_result.get("query", "")
    urls = prev_result.get("urls", [])

    logger.info(
        f"📊 EvaluatorAgent started | job_id={job_id} | urls={len(urls)}"
    )

    # ── mark started ──────────────────────────────────────────────
    update_job_progress_sync(
        job_id=job_id,
        agent_name=AGENT_NAME,
        status=JobState.STARTED,
        progress=5,
        worker_id=worker_id,
    )

    # Edge case: scanner found nothing
    if not urls:
        logger.warning(f"EvaluatorAgent: no URLs to evaluate | job_id={job_id}")
        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.SUCCESS, progress=100,
            result={"evaluated": 0, "accepted": 0, "rejected": 0},
            worker_id=worker_id,
        )
        return {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "evaluated": 0,
            "accepted": 0,
            "rejected": 0,
            "docs": [],
        }

    try:
        evaluated_docs: List[Dict[str, Any]] = []
        accepted = 0
        rejected = 0
        total = len(urls)

        for idx, url_item in enumerate(urls, 1):
            url = url_item.get("url", "")
            title_hint = url_item.get("title", "")

            # Progress: 5% start, 5-90% during evaluation, 90-100% final
            progress = 5 + int((idx - 1) / total * 85)
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.RUNNING, progress=progress, worker_id=worker_id,
            )

            logger.info(f"  [{idx}/{total}] Evaluating: {url[:80]}")

            # ── step A: fetch page ─────────────────────────────────
            fetch_result = _fetch_url_sync(url)

            if fetch_result["error"] or not fetch_result["html"]:
                logger.warning(
                    f"  Fetch failed: {url[:60]} → {fetch_result['error']}"
                )
                _update_document_in_mongo(url, session_id, {
                    "status": "failed",
                    "error_message": f"Fetch error: {fetch_result['error']}",
                })
                rejected += 1
                continue

            # ── step B: extract text ───────────────────────────────
            extracted = _extract_text(fetch_result["html"], title_hint)
            text = extracted["text"]
            title = extracted["title"] or title_hint
            word_count = extracted["word_count"]

            # Skip pages with almost no content
            if word_count < MIN_WORD_COUNT:
                logger.info(
                    f"  Too thin ({word_count} words), skipping: {url[:60]}"
                )
                _update_document_in_mongo(url, session_id, {
                    "status": "failed",
                    "error_message": f"Insufficient content ({word_count} words)",
                })
                rejected += 1
                continue

            # ── step C: LLM scoring ────────────────────────────────
            evaluation = _evaluate_with_llm(url, title, text, query)
            score = evaluation["score"]
            should_extract = evaluation["should_extract"]

            logger.info(
                f"  Score {score:4.1f} | {'ACCEPT' if should_extract else 'REJECT'} "
                f"| {title[:50]}"
            )

            # ── step D: update MongoDB document record ─────────────
            from app.models.documents import DocumentStatus

            mongo_status = (
                DocumentStatus.EVALUATED.value
                if should_extract
                else DocumentStatus.FAILED.value
            )
            _update_document_in_mongo(url, session_id, {
                "title": title,
                "raw_content": fetch_result["html"],
                "relevance_score": score,
                "status": mongo_status,
                "error_message": None if should_extract else f"Rejected: score {score} < {ACCEPT_THRESHOLD}",
                "metadata": {
                    "reason": evaluation["reason"],
                    "key_topics": evaluation["key_topics"],
                    "word_count": word_count,
                    "content_type": fetch_result["content_type"],
                },
            })

            if should_extract:
                accepted += 1
                evaluated_docs.append({
                    "url": url,
                    "title": title,
                    "raw_html": fetch_result["html"],
                    "text_preview": text[:500],
                    "score": score,
                    "reason": evaluation["reason"],
                    "key_topics": evaluation["key_topics"],
                    "should_extract": True,
                    "word_count": word_count,
                    "content_type": fetch_result["content_type"],
                })
            else:
                rejected += 1

        # ── mark complete ──────────────────────────────────────────
        update_job_progress_sync(
            job_id=job_id,
            agent_name=AGENT_NAME,
            status=JobState.SUCCESS,
            progress=100,
            result={
                "evaluated": total,
                "accepted": accepted,
                "rejected": rejected,
            },
            worker_id=worker_id,
        )

        logger.info(
            f"✅ EvaluatorAgent done | job_id={job_id} "
            f"| accepted={accepted}/{total}"
        )

        return {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "evaluated": total,
            "accepted": accepted,
            "rejected": rejected,
            "docs": evaluated_docs,
        }

    except SoftTimeLimitExceeded:
        msg = "EvaluatorAgent hit the 4-minute soft time limit"
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
            f"❌ EvaluatorAgent failed | job_id={job_id} | error={msg}",
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
                "job_id": job_id,
                "session_id": session_id,
                "query": query,
                "evaluated": 0,
                "accepted": 0,
                "rejected": 0,
                "docs": [],
                "error": msg,
            }
