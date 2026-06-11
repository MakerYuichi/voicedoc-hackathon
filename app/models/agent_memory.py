"""
Pydantic models for the `agent_memory` collection.

Stores per-session conversation history so the QueryAgent can maintain context
across follow-up questions.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class Message(BaseModel):
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentMemoryBase(BaseModel):
    session_id: str = Field(..., description="Unique session identifier")
    agent_name: str = Field(..., description="Which agent owns this memory, e.g. QueryAgent")
    messages: List[Message] = Field(default_factory=list)
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary key-value context the agent needs between turns",
    )
    document_ids: List[str] = Field(
        default_factory=list,
        description="Document IDs available in this session",
    )


class AgentMemoryCreate(AgentMemoryBase):
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentMemoryUpdate(BaseModel):
    messages: Optional[List[Message]] = None
    context: Optional[Dict[str, Any]] = None
    document_ids: Optional[List[str]] = None
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class AgentMemoryInDB(AgentMemoryCreate):
    id: Optional[str] = Field(None, alias="_id")

    model_config = {"populate_by_name": True}
