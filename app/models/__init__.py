"""
Pydantic models for every MongoDB collection.
"""
from app.models.documents import (
    DocumentStatus,
    DocumentBase,
    DocumentCreate,
    DocumentUpdate,
    DocumentInDB,
)
from app.models.chunks import ChunkBase, ChunkCreate, ChunkInDB
from app.models.agent_memory import (
    MessageRole,
    Message,
    AgentMemoryBase,
    AgentMemoryCreate,
    AgentMemoryUpdate,
    AgentMemoryInDB,
)
from app.models.job_status import (
    JobState,
    JobStatusBase,
    JobStatusCreate,
    JobStatusUpdate,
    JobStatusInDB,
)
from app.models.query_logs import (
    SourceReference,
    QueryLogBase,
    QueryLogCreate,
    QueryLogInDB,
)

__all__ = [
    # documents
    "DocumentStatus", "DocumentBase", "DocumentCreate", "DocumentUpdate", "DocumentInDB",
    # chunks
    "ChunkBase", "ChunkCreate", "ChunkInDB",
    # agent_memory
    "MessageRole", "Message", "AgentMemoryBase", "AgentMemoryCreate",
    "AgentMemoryUpdate", "AgentMemoryInDB",
    # job_status
    "JobState", "JobStatusBase", "JobStatusCreate", "JobStatusUpdate", "JobStatusInDB",
    # query_logs
    "SourceReference", "QueryLogBase", "QueryLogCreate", "QueryLogInDB",
]
