"""
ExtractorAgent — Celery worker task (Step 7).

Responsibilities
----------------
1. Receive evaluated docs from EvaluatorAgent (prev_result)
2. For each accepted doc:
   a. Run trafilatura over raw HTML → clean main-content text + metadata
   b. Convert to markdown with markdownify (fallback if trafilatura fails)
   c. Strip residual noise (nav, ads, cookie banners)
   d. Extract: title, author, publish date, word count
   e. Use LLM to generate a 3-sentence summary
   f. Persist full extraction to MongoDB `documents` collection
3. Update job_status progress and broadcast WebSocket updates
4. Return extracted docs list → chains to ProcessorAgent

Chain position
--------------
    evaluator_task → extractor_task(prev_result, job_id, session_id)
                         → processor_task(prev_result, job_id, session_id)

prev_result shape (from evaluator)
------------------------------------
{
    "job_id": str, "session_id": str, "query": str,
    "evaluated": int, "accepted": int, "rejected": int,
    "docs": [
        {"url": str, "title": str, "raw_html": str,
         "score": float, "key_topics": [str], ...}
    ]
}

Return shape (forwarded to processor)
---------------------------------------
{
    "job_id": str,
    "session_id": str,
    "query": str,
    "extracted": int,
    "docs": [
        {
            "url": str,
            "title": str,
            "author": str,
            "date": str,
            "content_markdown": str,
            "summary": str,
            "word_count": int,
            "key_topics": [str],
            "score": float,
            "extracted_at": str   # ISO datetime
        }
    ]
}
"""
from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import markdownify as md
import trafilatura
from bs4 import BeautifulSoup
from celery.exceptions import SoftTimeLimitExceeded
from langchain_core.messages import HumanMessage

from app.celery_config import QUEUE_EXTRACTOR, celery_app
from app.models.job_status import JobState
from app.utils.job_manager import update_job_progress_sync
from app.utils.llm import get_llm_sync

logger = logging.getLogger(__name__)

# ── constants ──────────────────────────────────────────────────────

AGENT_NAME = "ExtractorAgent"

# Max markdown length forwarded to the processor (keeps MongoDB docs reasonable)
MAX_MARKDOWN_CHARS = 150_000

# Max text given to LLM for summarisation
MAX_TEXT_FOR_SUMMARY = 4_000

# CSS-style noise patterns stripped from markdown output
_NOISE_PATTERNS = [
    r"(?i)(cookie|privacy) (policy|notice|consent|banner)[^\n]*\n?",
    r"(?i)subscribe (to|for) (our )?(newsletter|updates)[^\n]*\n?",
    r"(?i)sign (up|in) (to|for) [^\n]*\n?",
    r"(?i)advertisement\n?",
    r"\[!\[.*?\]\(.*?\)\]\(.*?\)",    # badge/image links [[img](url)](url)
    r"\n{3,}",                         # 3+ blank lines → 2
]


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


# ── extraction pipeline ────────────────────────────────────────────

def _extract_with_trafilatura(html: str, url: str) -> Dict[str, Any]:
    """
    Primary extractor: trafilatura for main-content extraction + metadata.

    Returns
    -------
    {
        "text": str, "title": str, "author": str,
        "date": str, "success": bool
    }
    """
    try:
        # include_comments=False, include_tables=True
        result = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            no_fallback=False,
            favor_recall=True,       # keep more content, fewer false negatives
            output_format="txt",
        )

        if not result or len(result.strip()) < 50:
            return {"text": "", "title": "", "author": "", "date": "", "success": False}

        # Extract metadata separately
        meta = trafilatura.extract_metadata(html, default_url=url)
        title  = (meta.title  if meta and meta.title  else "") or ""
        author = (meta.author if meta and meta.author else "") or ""
        date   = (meta.date   if meta and meta.date   else "") or ""

        return {
            "text": result.strip(),
            "title": title,
            "author": author,
            "date": date,
            "success": True,
        }

    except Exception as exc:
        logger.warning(f"trafilatura failed for {url}: {exc}")
        return {"text": "", "title": "", "author": "", "date": "", "success": False}


def _extract_with_bs4_fallback(html: str, title_hint: str = "") -> Dict[str, Any]:
    """
    Fallback extractor: BeautifulSoup plain-text extraction.
    Used when trafilatura returns nothing.
    """
    try:
        soup = BeautifulSoup(html, "lxml")

        for tag in soup(["script", "style", "nav", "footer", "header",
                         "aside", "form", "noscript", "iframe"]):
            tag.decompose()

        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else title_hint

        body = (
            soup.find("article")
            or soup.find("main")
            or soup.find("div", {"id": re.compile(r"content|main|article", re.I)})
            or soup.find("body")
            or soup
        )
        text = re.sub(r"\s+", " ", body.get_text(separator=" ", strip=True))

        # Look for author / date in meta tags
        author = ""
        date = ""
        for m in soup.find_all("meta"):
            name = (m.get("name") or m.get("property") or "").lower()
            content = m.get("content", "")
            if "author" in name:
                author = content
            elif "date" in name or "published" in name:
                date = content

        return {
            "text": text,
            "title": title,
            "author": author,
            "date": date,
            "success": bool(text),
        }
    except Exception as exc:
        logger.warning(f"BS4 fallback failed: {exc}")
        return {"text": "", "title": title_hint, "author": "", "date": "", "success": False}


def _html_to_markdown(html: str) -> str:
    """
    Convert HTML to markdown using markdownify, then strip noise patterns.
    """
    try:
        # markdownify settings: keep links + headings, strip images
        markdown = md.markdownify(
            html,
            heading_style=md.ATX,
            strip=["img", "script", "style", "nav", "footer",
                   "header", "aside", "form", "iframe"],
            newline_style=md.BACKSLASH,
            bullets="-",
        )
    except Exception as exc:
        logger.warning(f"markdownify failed: {exc}")
        return ""

    # Clean noise
    for pattern in _NOISE_PATTERNS:
        markdown = re.sub(pattern, "\n\n" if pattern == r"\n{3,}" else "", markdown)

    # Collapse 3+ newlines → 2
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return markdown.strip()[:MAX_MARKDOWN_CHARS]


def _count_words(text: str) -> int:
    return len(text.split()) if text else 0


# ── LLM summary ───────────────────────────────────────────────────

def _generate_summary(
    url: str,
    title: str,
    text: str,
    query: str,
) -> str:
    """
    Ask the LLM for a 3-sentence summary of the document
    focused on its relevance to the original query.
    Falls back to a truncated text snippet if the LLM fails.
    """
    snippet = text[:MAX_TEXT_FOR_SUMMARY]

    prompt = f"""You are summarising a web document for a research pipeline.

Original query: "{query}"
Document URL: {url}
Document Title: {title}
Document Text (excerpt):
---
{snippet}
---

Write EXACTLY 3 sentences that:
1. State what the document is about
2. Explain its relevance to the query
3. Highlight the most important fact or finding

Return ONLY the 3 sentences. No headers, no bullet points, no extra text."""

    try:
        llm = get_llm_sync(temperature=0.3, max_tokens=200)
        response = llm.invoke([HumanMessage(content=prompt)])
        summary = response.content.strip()
        # Ensure it's not empty and not excessively long
        if summary and len(summary) > 20:
            return summary[:600]
    except Exception as exc:
        logger.warning(f"LLM summary failed for {url}: {exc}")

    # Fallback: first 300 chars of clean text
    return (text[:300] + "...") if len(text) > 300 else text


# ── MongoDB update ─────────────────────────────────────────────────

def _save_extraction_to_mongo(
    url: str,
    session_id: str,
    fields: Dict[str, Any],
) -> None:
    """Update the document record with extracted content."""
    from app.database.db import db_manager

    async def _update():
        await db_manager.documents.update_one(
            {"source_url": url, "session_id": session_id},
            {"$set": {**fields, "updated_at": _now()}},
            upsert=False,
        )

    _run_async(_update())


# ── Main Celery task ───────────────────────────────────────────────

@celery_app.task(
    name="app.agents.extractor_agent.extractor_task",
    bind=True,
    queue=QUEUE_EXTRACTOR,
    max_retries=3,
    default_retry_delay=5,
)
def extractor_task(
    self,
    prev_result: dict,
    job_id: str,
    session_id: str,
) -> dict:
    """
    Convert accepted HTML documents to clean markdown + metadata + summary.

    Parameters
    ----------
    prev_result  : evaluator_task return dict (contains docs list + query)
    job_id       : UUID from SupervisorAgent
    session_id   : browser session ID

    Returns
    -------
    dict — forwarded as prev_result to processor_task
    """
    worker_id = self.request.hostname
    query = prev_result.get("query", "")
    docs  = prev_result.get("docs", [])

    logger.info(
        f"📄 ExtractorAgent started | job_id={job_id} | docs={len(docs)}"
    )

    # ── mark started ──────────────────────────────────────────────
    update_job_progress_sync(
        job_id=job_id,
        agent_name=AGENT_NAME,
        status=JobState.STARTED,
        progress=5,
        worker_id=worker_id,
    )

    # Edge case: nothing to extract
    if not docs:
        logger.warning(f"ExtractorAgent: no docs to extract | job_id={job_id}")
        update_job_progress_sync(
            job_id=job_id, agent_name=AGENT_NAME,
            status=JobState.SUCCESS, progress=100,
            result={"extracted": 0},
            worker_id=worker_id,
        )
        return {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "extracted": 0,
            "docs": [],
        }

    try:
        extracted_docs: List[Dict[str, Any]] = []
        total = len(docs)

        for idx, doc in enumerate(docs, 1):
            url      = doc.get("url", "")
            raw_html = doc.get("raw_html", "")
            title_hint = doc.get("title", "")
            score      = doc.get("score", 0.0)
            key_topics = doc.get("key_topics", [])

            # Incremental progress 5 → 85%
            progress = 5 + int((idx - 1) / total * 80)
            update_job_progress_sync(
                job_id=job_id, agent_name=AGENT_NAME,
                status=JobState.RUNNING, progress=progress, worker_id=worker_id,
            )

            logger.info(f"  [{idx}/{total}] Extracting: {url[:80]}")

            if not raw_html:
                logger.warning(f"  No HTML for {url[:60]}, skipping")
                continue

            # ── step A: trafilatura (primary) ──────────────────────
            traf = _extract_with_trafilatura(raw_html, url)

            if traf["success"]:
                clean_text = traf["text"]
                title  = traf["title"]  or title_hint
                author = traf["author"]
                date   = traf["date"]
                logger.debug(f"  trafilatura OK: {len(clean_text)} chars")
            else:
                # ── step B: BS4 fallback ───────────────────────────
                logger.debug(f"  trafilatura returned nothing, using BS4 fallback")
                bs4_result = _extract_with_bs4_fallback(raw_html, title_hint)
                clean_text = bs4_result["text"]
                title  = bs4_result["title"] or title_hint
                author = bs4_result["author"]
                date   = bs4_result["date"]

            if not clean_text or _count_words(clean_text) < 50:
                logger.warning(f"  Extraction yielded too little text for {url[:60]}")
                _save_extraction_to_mongo(url, session_id, {
                    "status": "failed",
                    "error_message": "Extraction yielded insufficient text",
                })
                continue

            # ── step C: HTML → markdown ────────────────────────────
            content_markdown = _html_to_markdown(raw_html)

            # If markdownify produced almost nothing, use plain text
            if _count_words(content_markdown) < 30:
                logger.debug("  Markdown too short, using plain text as fallback")
                content_markdown = clean_text

            word_count = _count_words(content_markdown)

            # ── step D: LLM summary ────────────────────────────────
            summary = _generate_summary(url, title, clean_text, query)
            logger.info(f"  Summary generated ({len(summary)} chars)")

            extracted_at = _now()

            # ── step E: persist to MongoDB ─────────────────────────
            from app.models.documents import DocumentStatus

            _save_extraction_to_mongo(url, session_id, {
                "title": title,
                "markdown_content": content_markdown,
                "status": DocumentStatus.EXTRACTED.value,
                "error_message": None,
                "metadata": {
                    "author": author,
                    "date": date,
                    "word_count": word_count,
                    "summary": summary,
                    "key_topics": key_topics,
                    "extracted_at": extracted_at.isoformat(),
                },
            })

            extracted_docs.append({
                "url": url,
                "title": title,
                "author": author,
                "date": date,
                "content_markdown": content_markdown,
                "summary": summary,
                "word_count": word_count,
                "key_topics": key_topics,
                "score": score,
                "extracted_at": extracted_at.isoformat(),
            })

            logger.info(
                f"  Done: '{title[:50]}' | {word_count} words | "
                f"summary={len(summary)} chars"
            )

        # ── mark complete ──────────────────────────────────────────
        update_job_progress_sync(
            job_id=job_id,
            agent_name=AGENT_NAME,
            status=JobState.SUCCESS,
            progress=100,
            result={"extracted": len(extracted_docs), "total": total},
            worker_id=worker_id,
        )

        logger.info(
            f"✅ ExtractorAgent done | job_id={job_id} "
            f"| extracted={len(extracted_docs)}/{total}"
        )

        return {
            "job_id": job_id,
            "session_id": session_id,
            "query": query,
            "extracted": len(extracted_docs),
            "docs": extracted_docs,
        }

    except SoftTimeLimitExceeded:
        msg = "ExtractorAgent hit the 4-minute soft time limit"
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
            f"❌ ExtractorAgent failed | job_id={job_id} | error={msg}",
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
                "extracted": 0,
                "docs": [],
                "error": msg,
            }
