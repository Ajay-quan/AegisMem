"""FastAPI API schemas for stateful.ai endpoints."""
from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1)
    user_id: str
    session_id: str = ""
    agent_id: str = ""
    memory_type: str = "observation"
    source_type: str = "user_message"
    metadata: dict[str, Any] = Field(default_factory=dict)


class IngestResponse(BaseModel):
    memory_id: str
    user_id: str
    memory_type: str
    importance_score: float
    content_preview: str
    created_at: datetime


class BatchIngestRequest(BaseModel):
    items: list[IngestRequest]


class BatchIngestResponse(BaseModel):
    ingested: int
    failed: int
    memory_ids: list[str]


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    user_id: str
    agent_id: str = ""
    namespace: str = ""
    top_k: int = Field(default=5, ge=1, le=50)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)
    memory_types: list[str] = Field(default_factory=list)
    time_range_start: datetime | None = None
    time_range_end: datetime | None = None
    include_archived: bool = False


class MemorySnippet(BaseModel):
    memory_id: str
    content: str
    memory_type: str
    importance_score: float
    semantic_score: float
    composite_score: float
    rank: int
    created_at: datetime


class RetrieveResponse(BaseModel):
    query: str
    results: list[MemorySnippet]
    total_found: int
    latency_ms: float
    context_window: str = ""
    # Echo back to POST /feedback to close the continual-learning loop. Empty
    # string when continual learning is disabled.
    query_id: str = ""


# ---------------------------------------------------------------------------
# Feedback / continual learning
# ---------------------------------------------------------------------------


class FeedbackRequest(BaseModel):
    query_id: str = Field(..., description="query_id returned by /retrieve")
    memory_id: str = Field(..., description="which served memory the feedback is about")
    useful: bool | None = Field(default=None, description="coarse explicit signal")
    score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="fine grade in [0,1] (overrides 'useful')",
    )
    outcome: str = Field(default="", description="'success' | 'failure' | ''")


class FeedbackResponse(BaseModel):
    recorded: bool
    reward: float
    namespace: str
    policy_updates: int
    weights: dict[str, float] = Field(default_factory=dict)
    message: str = ""


class LearningStatsResponse(BaseModel):
    enabled: bool
    replay: dict[str, Any] = Field(default_factory=dict)
    policy: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


class MemoryDetailResponse(BaseModel):
    memory_id: str
    namespace: str
    user_id: str
    agent_id: str
    memory_type: str
    content: str
    source_type: str
    created_at: datetime
    updated_at: datetime
    importance_score: float
    recency_score: float
    confidence_score: float
    access_count: int
    version: int
    status: str
    contradiction_status: str
    tags: list[str]
    metadata: dict[str, Any]


class ListMemoriesRequest(BaseModel):
    user_id: str
    namespace: str = ""
    memory_type: str = ""
    status: str = "active"
    limit: int = Field(default=20, ge=1, le=100)
    offset: int = Field(default=0, ge=0)


class UpdateRequest(BaseModel):
    user_id: str
    new_content: str
    namespace: str = ""
    agent_id: str = ""


class UpdateResponse(BaseModel):
    memory_id: str
    action: str
    reason: str
    previous_memory_id: str
    content_preview: str


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


class ReflectRequest(BaseModel):
    user_id: str
    namespace: str = ""
    agent_id: str = ""
    force: bool = False


class ReflectResponse(BaseModel):
    reflections_generated: int
    reflections: list[dict[str, Any]]


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------


class ContradictionScanRequest(BaseModel):
    memory_id: str
    user_id: str


class ContradictionScanResponse(BaseModel):
    memory_id: str
    contradictions_found: int
    reports: list[dict[str, Any]]


class ContradictionListResponse(BaseModel):
    contradictions: list[dict[str, Any]]
    total: int


# ---------------------------------------------------------------------------
# Health / Status
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    status: str
    version: str
    components: dict[str, str]


class StatsResponse(BaseModel):
    user_id: str
    total_memories: int
    namespaces: list[str]
    by_type: dict[str, int] = Field(default_factory=dict)
    by_status: dict[str, int] = Field(default_factory=dict)
    avg_importance: float = 0.0
    total_access_count: int = 0
    unresolved_contradictions: int = 0
    last_memory_at: str = ""
