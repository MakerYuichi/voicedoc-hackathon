"""
Multi-agent system for VoiceDoc Intelligence.

Agents
------
SupervisorAgent  — LangGraph graph, orchestrates the pipeline
ScannerAgent     — Celery task, fetches web documents      (Step 5)
EvaluatorAgent   — Celery task, scores relevance           (Step 6)
ExtractorAgent   — Celery task, converts HTML → markdown   (Step 7)
ProcessorAgent   — Celery task, chunks + embeds + stores   (Step 8)
QueryAgent       — standalone RAG agent                    (Step 9)
"""
from app.agents.supervisor_agent import SupervisorAgent, supervisor_agent

__all__ = ["SupervisorAgent", "supervisor_agent"]
