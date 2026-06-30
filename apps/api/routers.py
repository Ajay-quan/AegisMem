"""FastAPI routers for memory operations."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import (
    get_ingest_service, get_retrieve_service, get_update_service,
    get_reflect_service, get_contradiction_service, get_db_store,
    get_feedback_service,
)
from apps.api.schemas import (
    IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse,
    RetrieveRequest, RetrieveResponse, MemorySnippet,
    MemoryDetailResponse, ListMemoriesRequest,
    UpdateRequest, UpdateResponse,
    ReflectRequest, ReflectResponse,
    ContradictionScanRequest, ContradictionScanResponse, ContradictionListResponse,
    StatsResponse,
    FeedbackRequest, FeedbackResponse, LearningStatsResponse,
)
from core.schemas.memory import Observation, SourceType, RetrievalQuery
from core.exceptions import MemoryNotFoundError

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


@router.post("/ingest", response_model=IngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_memory(
    request: IngestRequest,
    svc=Depends(get_ingest_service),
):
    """Ingest a single observation into memory."""
    try:
        from core.schemas.memory import MemoryType
        mem_type = MemoryType(request.memory_type)
        src_type = SourceType(request.source_type)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    memory = await svc.ingest_text(
        text=request.text,
        user_id=request.user_id,
        session_id=request.session_id,
        agent_id=request.agent_id,
        memory_type=mem_type,
        source_type=src_type,
        metadata=request.metadata,
    )

    return IngestResponse(
        memory_id=memory.memory_id,
        user_id=memory.user_id,
        memory_type=memory.memory_type if isinstance(memory.memory_type, str) else memory.memory_type.value,
        importance_score=memory.importance_score,
        content_preview=memory.content[:200],
        created_at=memory.created_at,
    )


@router.post("/ingest/batch", response_model=BatchIngestResponse, status_code=status.HTTP_201_CREATED)
async def ingest_batch(
    request: BatchIngestRequest,
    svc=Depends(get_ingest_service),
):
    """Batch ingest multiple observations."""
    ingested = 0
    failed = 0
    memory_ids = []

    for item in request.items:
        try:
            memory = await svc.ingest_text(
                text=item.text,
                user_id=item.user_id,
                session_id=item.session_id,
                agent_id=item.agent_id,
                metadata=item.metadata,
            )
            memory_ids.append(memory.memory_id)
            ingested += 1
        except Exception as e:
            logger.error(f"Batch ingest item failed: {e}")
            failed += 1

    return BatchIngestResponse(ingested=ingested, failed=failed, memory_ids=memory_ids)


# ---------------------------------------------------------------------------
# Retrieve
# ---------------------------------------------------------------------------


@router.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_memories(
    request: RetrieveRequest,
    svc=Depends(get_retrieve_service),
):
    """Retrieve memories using hybrid semantic + symbolic search."""
    from core.schemas.memory import MemoryType
    types = []
    for t in request.memory_types:
        try:
            types.append(MemoryType(t))
        except ValueError:
            pass

    query = RetrievalQuery(
        query_text=request.query,
        user_id=request.user_id,
        agent_id=request.agent_id,
        namespace=request.namespace,
        top_k=request.top_k,
        min_importance=request.min_importance,
        memory_types=types,
        time_range_start=request.time_range_start,
        time_range_end=request.time_range_end,
        include_archived=request.include_archived,
    )

    result = await svc.retrieve(query)

    snippets = []
    for c in result.candidates:
        m = c.memory
        snippets.append(MemorySnippet(
            memory_id=m.memory_id,
            content=m.content,
            memory_type=m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value,
            importance_score=m.importance_score,
            semantic_score=c.semantic_score,
            composite_score=c.composite_score,
            rank=c.rank,
            created_at=m.created_at,
        ))

    # Build context window
    context = "\n".join([f"[{s.rank}] {s.content}" for s in snippets]) if snippets else ""

    return RetrieveResponse(
        query=request.query,
        results=snippets,
        total_found=result.total_found,
        latency_ms=result.latency_ms,
        context_window=context,
        query_id=result.query_id,
    )


# ---------------------------------------------------------------------------
# Feedback / continual learning (Stateful-CL)
# ---------------------------------------------------------------------------


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(
    request: FeedbackRequest,
    svc=Depends(get_feedback_service),
):
    """Report whether a retrieved memory was useful, driving online learning.

    Echo the ``query_id`` from a prior /retrieve response plus the ``memory_id``
    being judged. When continual learning is enabled this shapes a reward and
    updates the per-namespace ranking policy; otherwise it is a safe no-op.
    """
    result = await svc.record(
        query_id=request.query_id,
        memory_id=request.memory_id,
        useful=request.useful,
        score=request.score,
        outcome=request.outcome,
    )
    return FeedbackResponse(
        recorded=result.recorded,
        reward=result.reward,
        namespace=result.namespace,
        policy_updates=result.policy_updates,
        weights=result.weights,
        message=result.message,
    )


@router.get("/audit")
async def audit_log(limit: int = 100):
    """Recent mutating operations with the principal that performed them.

    Auth-protected (lives under /api/). Newest first. Bounded ring buffer; wire
    to a durable sink for long-term retention in production.
    """
    from core.security.audit import get_audit_log
    log = get_audit_log()
    return {"total": len(log), "entries": log.list(limit=limit)}


@router.get("/learning/stats", response_model=LearningStatsResponse)
async def learning_stats():
    """Inspect the continual-learning state: replay buffer + policy per namespace."""
    from core.config.settings import settings as _settings
    if not _settings.continual_learning_enabled:
        return LearningStatsResponse(enabled=False, replay={}, policy={})
    from domain.learning.registry import get_ranking_policy, get_replay_buffer
    return LearningStatsResponse(
        enabled=True,
        replay=get_replay_buffer().stats(),
        policy=get_ranking_policy().stats(),
    )


# ---------------------------------------------------------------------------
# Memory CRUD
# ---------------------------------------------------------------------------


@router.get("/memories/{memory_id}", response_model=MemoryDetailResponse)
async def get_memory(memory_id: str, db=Depends(get_db_store)):
    """Get a single memory by ID."""
    try:
        memory = await db.get_memory(memory_id)
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail=f"Memory {memory_id} not found")

    return MemoryDetailResponse(
        memory_id=memory.memory_id,
        namespace=memory.namespace,
        user_id=memory.user_id,
        agent_id=memory.agent_id,
        memory_type=memory.memory_type if isinstance(memory.memory_type, str) else memory.memory_type.value,
        content=memory.content,
        source_type=memory.source_type if isinstance(memory.source_type, str) else memory.source_type.value,
        created_at=memory.created_at,
        updated_at=memory.updated_at,
        importance_score=memory.importance_score,
        recency_score=memory.recency_score,
        confidence_score=memory.confidence_score,
        access_count=memory.access_count,
        version=memory.version,
        status=memory.status if isinstance(memory.status, str) else memory.status.value,
        contradiction_status=memory.contradiction_status if isinstance(memory.contradiction_status, str) else memory.contradiction_status.value,
        tags=memory.tags,
        metadata=memory.metadata,
    )


@router.post("/memories/list", response_model=list[MemoryDetailResponse])
async def list_memories(request: ListMemoriesRequest, db=Depends(get_db_store)):
    """List memories for a user."""
    memories = await db.list_memories(
        user_id=request.user_id,
        namespace=request.namespace,
        memory_type=request.memory_type,
        status=request.status,
        limit=request.limit,
        offset=request.offset,
    )
    return [
        MemoryDetailResponse(
            memory_id=m.memory_id,
            namespace=m.namespace,
            user_id=m.user_id,
            agent_id=m.agent_id,
            memory_type=m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value,
            content=m.content,
            source_type=m.source_type if isinstance(m.source_type, str) else m.source_type.value,
            created_at=m.created_at,
            updated_at=m.updated_at,
            importance_score=m.importance_score,
            recency_score=m.recency_score,
            confidence_score=m.confidence_score,
            access_count=m.access_count,
            version=m.version,
            status=m.status if isinstance(m.status, str) else m.status.value,
            contradiction_status=m.contradiction_status if isinstance(m.contradiction_status, str) else m.contradiction_status.value,
            tags=m.tags,
            metadata=m.metadata,
        )
        for m in memories
    ]


@router.delete("/memories/{memory_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_memory(memory_id: str, user_id: str, db=Depends(get_db_store)):
    """Soft-delete a memory."""
    await db.delete_memory(memory_id, user_id)


# ---------------------------------------------------------------------------
# Update
# ---------------------------------------------------------------------------


@router.post("/update", response_model=UpdateResponse)
async def update_memory(request: UpdateRequest, svc=Depends(get_update_service)):
    """Smart update: creates, updates, merges, or supersedes based on LLM decision."""
    memory, decision = await svc.update_or_create(
        user_id=request.user_id,
        new_content=request.new_content,
        namespace=request.namespace,
        agent_id=request.agent_id,
    )
    return UpdateResponse(
        memory_id=memory.memory_id,
        action=decision.action,
        reason=decision.reason,
        previous_memory_id=decision.existing_memory_id,
        content_preview=memory.content[:200],
    )


# ---------------------------------------------------------------------------
# Reflection
# ---------------------------------------------------------------------------


@router.post("/reflect", response_model=ReflectResponse)
async def generate_reflections(request: ReflectRequest, svc=Depends(get_reflect_service)):
    """Generate higher-level reflections from stored memories."""
    namespace = request.namespace or f"user:{request.user_id}"
    reflections = await svc.run_reflection_cycle(
        user_id=request.user_id,
        namespace=namespace,
        agent_id=request.agent_id,
        force=request.force,
    )
    return ReflectResponse(
        reflections_generated=len(reflections),
        reflections=[
            {
                "reflection_id": r.reflection_id,
                "content": r.content,
                "confidence": r.confidence,
                "derivation_ids": r.derivation_ids,
            }
            for r in reflections
        ],
    )


# ---------------------------------------------------------------------------
# Contradiction
# ---------------------------------------------------------------------------


@router.post("/contradictions/scan", response_model=ContradictionScanResponse)
async def scan_contradictions(
    request: ContradictionScanRequest,
    db=Depends(get_db_store),
    svc=Depends(get_contradiction_service),
):
    """Scan a memory for contradictions with existing memories."""
    try:
        memory = await db.get_memory(request.memory_id)
    except MemoryNotFoundError:
        raise HTTPException(status_code=404, detail="Memory not found")

    reports = await svc.scan_for_contradictions(memory)

    return ContradictionScanResponse(
        memory_id=request.memory_id,
        contradictions_found=len(reports),
        reports=[
            {
                "report_id": r.report_id,
                "memory_a_id": r.memory_a_id,
                "memory_b_id": r.memory_b_id,
                "description": r.contradiction_description,
                "confidence": r.confidence,
            }
            for r in reports
        ],
    )


@router.get("/contradictions", response_model=ContradictionListResponse)
async def list_contradictions(resolved: bool = False, db=Depends(get_db_store)):
    """List all detected contradictions."""
    contradictions = await db.list_contradictions(resolved=resolved)
    return ContradictionListResponse(contradictions=contradictions, total=len(contradictions))


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@router.get("/stats", response_model=StatsResponse)
async def memory_stats(user_id: str, namespace: str = "", db=Depends(get_db_store)):
    """Aggregate statistics for a user's memory store.

    Powers operational dashboards: counts by type and status, importance
    distribution, access totals, and unresolved contradiction count.
    """
    memories = await db.list_memories(
        user_id=user_id, namespace=namespace, status="", limit=10_000, offset=0,
    )
    by_type: dict[str, int] = {}
    by_status: dict[str, int] = {}
    namespaces: set[str] = set()
    total_access = 0
    importance_sum = 0.0
    last_at = ""
    for m in memories:
        mtype = m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value
        mstatus = m.status if isinstance(m.status, str) else m.status.value
        by_type[mtype] = by_type.get(mtype, 0) + 1
        by_status[mstatus] = by_status.get(mstatus, 0) + 1
        namespaces.add(m.namespace)
        total_access += m.access_count
        importance_sum += m.importance_score
        created = m.created_at.isoformat() if hasattr(m.created_at, "isoformat") else str(m.created_at)
        last_at = max(last_at, created)

    try:
        contradictions = await db.list_contradictions(resolved=False)
    except Exception:
        contradictions = []

    return StatsResponse(
        user_id=user_id,
        total_memories=len(memories),
        namespaces=sorted(n for n in namespaces if n),
        by_type=by_type,
        by_status=by_status,
        avg_importance=round(importance_sum / len(memories), 4) if memories else 0.0,
        total_access_count=total_access,
        unresolved_contradictions=len(contradictions),
        last_memory_at=last_at,
    )
