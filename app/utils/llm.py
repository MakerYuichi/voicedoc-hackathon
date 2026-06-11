"""
LLM factory for VoiceDoc Intelligence.

Priority
--------
1. Groq (dev/test)  — if GROQ_API_KEY is set in the environment
2. Gemini           — always used in production / hackathon submission

All agents call get_llm() (or get_llm_sync()) instead of
instantiating ChatGoogleGenerativeAI directly, so swapping the
backend requires only a single .env change.

Usage
-----
    from app.utils.llm import get_llm, get_llm_sync

    # async context (FastAPI / LangGraph nodes)
    llm = get_llm()
    response = await llm.ainvoke([HumanMessage(content="hello")])

    # sync context (Celery workers)
    llm = get_llm_sync()
    response = llm.invoke([HumanMessage(content="hello")])

Notes
-----
- get_llm() and get_llm_sync() return the same object type
  (both support .invoke and .ainvoke).
- For scoring / extraction tasks that need low temperature,
  pass temperature=0.1 to get_llm().
- The active_provider() helper is exposed for logging / health
  checks — it returns "groq" or "gemini".
"""
from __future__ import annotations

import logging
from typing import Literal, Union

from langchain_core.language_models import BaseChatModel

from app.config import settings

logger = logging.getLogger(__name__)

# Type alias so callers don't need to import the concrete classes
LLMBackend = Literal["gemini", "groq"]


def active_provider() -> LLMBackend:
    """
    Return which LLM backend will be used.

    Groq is preferred when GROQ_API_KEY is non-empty.
    Falls back to Gemini unconditionally.
    """
    if settings.groq_api_key:
        return "groq"
    return "gemini"


def get_llm(
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> BaseChatModel:
    """
    Return the appropriate LangChain chat model.

    Parameters
    ----------
    temperature : override settings.temperature when provided
    max_tokens  : override settings.max_tokens when provided

    Returns
    -------
    BaseChatModel  — supports both .invoke() and .ainvoke()
    """
    temp = temperature if temperature is not None else settings.temperature
    tokens = max_tokens if max_tokens is not None else settings.max_tokens

    provider = active_provider()

    if provider == "groq":
        return _build_groq(temperature=temp, max_tokens=tokens)
    return _build_gemini(temperature=temp, max_tokens=tokens)


# Alias — identical signature, kept for clarity in sync contexts
get_llm_sync = get_llm


# ── private builders ───────────────────────────────────────────────

def _build_gemini(temperature: float, max_tokens: int) -> BaseChatModel:
    """Instantiate ChatGoogleGenerativeAI."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    logger.debug(f"LLM → Gemini | model={settings.gemini_model} | temp={temperature}")
    return ChatGoogleGenerativeAI(
        model=settings.gemini_model,
        google_api_key=settings.google_api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
    )


def _build_groq(temperature: float, max_tokens: int) -> BaseChatModel:
    """
    Instantiate ChatOpenAI pointed at Groq's OpenAI-compatible endpoint.

    Groq exposes the same REST API as OpenAI, so langchain-openai works
    without modification — just override base_url and api_key.
    """
    from langchain_openai import ChatOpenAI

    logger.debug(
        f"LLM → Groq | model={settings.groq_model} "
        f"| base_url={settings.groq_base_url} | temp={temperature}"
    )
    return ChatOpenAI(
        model=settings.groq_model,
        api_key=settings.groq_api_key,          # type: ignore[arg-type]
        base_url=settings.groq_base_url,
        temperature=temperature,
        max_tokens=max_tokens,
    )


# ── health check ───────────────────────────────────────────────────

async def llm_health_check() -> dict:
    """
    Cheap connectivity check — sends a tiny prompt and verifies a
    non-empty response.  Used by the /health endpoint.
    """
    from langchain_core.messages import HumanMessage

    provider = active_provider()
    try:
        llm = get_llm(temperature=0.0, max_tokens=10)
        response = await llm.ainvoke([HumanMessage(content="ping")])
        ok = bool(response.content)
        return {"llm": "healthy" if ok else "unhealthy", "provider": provider}
    except Exception as exc:
        logger.warning(f"LLM health check failed ({provider}): {exc}")
        return {"llm": "unhealthy", "provider": provider, "error": str(exc)[:120]}
