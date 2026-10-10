"""FastAPI routers for memory operations."""
from __future__ import annotations

import logging
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from apps.api.dependencies import (
    get_ingest_service, get_retrieve_service, get_update_service,
    get_reflect_service, get_contradiction_service, get_db_store,
    get_feedback_service, get_handoff_service, get_learner,
    get_erasure_service, get_vector_store, get_outcome_service, get_calibrator,
    get_risk_model,
)
from apps.api.schemas import (
    IngestRequest, IngestResponse, BatchIngestRequest, BatchIngestResponse,
    RetrieveRequest, RetrieveResponse, MemorySnippet,
    MemoryDetailResponse, ListMemoriesRequest,
    UpdateRequest, UpdateResponse,
    ReflectRequest, ReflectResponse,
    ContradictionScanRequest, ContradictionScanResponse, ContradictionListResponse,
    HandoffRequest, FeedbackRequest, FeedbackResponse, LearningStateResponse,
    ForgetRequest, ErasureReceipt, ReceiptVerifyResponse,
    QuarantineItem, QuarantineListResponse,
    CorrectionRequest, CorrectionResponse, TrustSourcesResponse,
)
from core.config.settings import settings
from services.outcome_service import OutcomeError
from services.erasure_service import ErasureError
from services.feedback_service import FeedbackError
from services.handoff_service import HandoffPacket
from domain.learning.weight_learner import default_weights
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
        status=memory.status if isinstance(memory.status, str) else memory.status.value,
        quarantine_reason=(memory.metadata.get("quarantine") or {}).get("reason"),
        trust=(memory.metadata.get("provenance") or {}).get("trust"),
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
        retrieval_id=result.retrieval_id,
        query=request.query,
        results=snippets,
        total_found=result.total_found,
        latency_ms=result.latency_ms,
        context_window=context,
        guard_report=result.guard_report,
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
# Handoff
# ---------------------------------------------------------------------------


@router.post("/handoff", response_model=HandoffPacket)
async def build_handoff(request: HandoffRequest, svc=Depends(get_handoff_service)):
    """Build a token-budgeted, cited context packet for a new agent, model or session."""
    return await svc.build(
        user_id=request.user_id,
        namespace=request.namespace,
        task=request.task,
        since=request.since,
        token_budget=request.token_budget,
        max_per_section=request.max_per_section,
    )


# ---------------------------------------------------------------------------
# Feedback / continual learning
# ---------------------------------------------------------------------------


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest, svc=Depends(get_feedback_service)):
    """Report which memories from a retrieval were used. Trains per-user ranking weights."""
    try:
        return await svc.submit(
            retrieval_id=request.retrieval_id,
            user_id=request.user_id,
            used_memory_ids=request.used_memory_ids,
        )
    except FeedbackError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/learning/{user_id}", response_model=LearningStateResponse)
async def get_learning_state(user_id: str, learner=Depends(get_learner)):
    """Inspect what the system has learned about ranking for this user."""
    state = await learner.state(user_id)
    defaults = default_weights()
    if state is None:
        return LearningStateResponse(
            user_id=user_id, learning_active=False, events=0, positives=0,
            weights=defaults, default_weights=defaults,
        )
    return LearningStateResponse(
        user_id=user_id,
        learning_active=state.events >= learner.min_events,
        events=state.events,
        positives=state.positives,
        weights={k: round(v, 4) for k, v in state.weights.items()},
        default_weights=defaults,
    )


@router.delete("/learning/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def reset_learning_state(user_id: str, learner=Depends(get_learner)):
    """Discard learned ranking weights for this user (reverts to defaults)."""
    await learner.reset(user_id)


# ---------------------------------------------------------------------------
# Integrity: verifiable erasure
# ---------------------------------------------------------------------------


@router.post("/forget", response_model=ErasureReceipt, tags=["integrity"])
async def forget(request: ForgetRequest, svc=Depends(get_erasure_service)):
    """Erase memories plus prior versions, derived memories and duplicates; return a signed receipt."""
    try:
        return await svc.forget(
            user_id=request.user_id,
            memory_ids=request.memory_ids,
            all_for_user=request.all_for_user,
            include_duplicates=request.include_duplicates,
            reason=request.reason,
        )
    except ErasureError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/receipts/{receipt_id}", response_model=ErasureReceipt, tags=["integrity"])
async def get_receipt(receipt_id: str, svc=Depends(get_erasure_service)):
    receipt = await svc.get_receipt(receipt_id)
    if receipt is None:
        raise HTTPException(status_code=404, detail="receipt not found")
    return receipt


@router.post("/receipts/verify", response_model=ReceiptVerifyResponse, tags=["integrity"])
async def verify_receipt(receipt: ErasureReceipt, svc=Depends(get_erasure_service)):
    """Check that a receipt was issued by this deployment and has not been altered."""
    return ReceiptVerifyResponse(valid=svc.verify_receipt(receipt.model_dump()))


# ---------------------------------------------------------------------------
# Integrity: quarantine review
# ---------------------------------------------------------------------------


@router.get("/quarantine/{user_id}", response_model=QuarantineListResponse, tags=["integrity"])
async def list_quarantine(user_id: str, db=Depends(get_db_store)):
    """Memories held back at ingest (low-trust, instruction-like). Never served until released."""
    held = await db.list_memories(user_id=user_id, status="pending", limit=200)
    items = [
        QuarantineItem(
            memory_id=m.memory_id,
            content=m.content,
            reason=(m.metadata.get("quarantine") or {}).get("reason", ""),
            trust=float((m.metadata.get("provenance") or {}).get("trust", 0.0)),
            source=str((m.metadata.get("provenance") or {}).get("source", "")),
            created_at=m.created_at,
        )
        for m in held
    ]
    return QuarantineListResponse(items=items, total=len(items))


@router.post("/quarantine/{memory_id}/release", tags=["integrity"])
async def release_quarantine(memory_id: str, user_id: str, svc=Depends(get_outcome_service)):
    """Approve held content after review. It becomes retrievable, and its origin earns trust."""
    try:
        return await svc.release(user_id=user_id, memory_id=memory_id)
    except OutcomeError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.post("/quarantine/{memory_id}/reject", tags=["integrity"])
async def reject_quarantine(memory_id: str, user_id: str, svc=Depends(get_outcome_service)):
    """Reject held content. Its origin loses trust deployment-wide, and the content is erased with a receipt."""
    try:
        return await svc.reject(user_id=user_id, memory_id=memory_id)
    except OutcomeError as e:
        raise HTTPException(status_code=409, detail=str(e))


# ---------------------------------------------------------------------------
# Continual learning from outcomes
# ---------------------------------------------------------------------------


@router.post("/corrections", response_model=CorrectionResponse, tags=["learning"])
async def submit_correction(request: CorrectionRequest, svc=Depends(get_outcome_service)):
    """Correct a memory. ``wrong`` teaches distrust of its origin; ``outdated`` does not.
    Memories derived from it are flagged as suspect and returned."""
    try:
        return await svc.correct(
            user_id=request.user_id,
            memory_id=request.memory_id,
            kind=request.kind,
            corrected_text=request.corrected_text,
            note=request.note,
        )
    except OutcomeError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.post("/memories/{memory_id}/confirm", tags=["learning"])
async def confirm_memory(memory_id: str, user_id: str, svc=Depends(get_outcome_service)):
    """User or reviewer confirms a memory is correct; its origin earns trust."""
    try:
        return await svc.confirm(user_id=user_id, memory_id=memory_id)
    except OutcomeError as e:
        raise HTTPException(status_code=422, detail=str(e))


@router.get("/trust/sources", response_model=TrustSourcesResponse, tags=["learning"])
async def trust_sources(calibrator=Depends(get_calibrator)):
    """Learned reliability of every source/origin, least trusted first."""
    rows = await calibrator.snapshot()
    return TrustSourcesResponse(sources=rows, guard_min_trust=settings.guard_min_trust)


@router.get("/risk-model", tags=["learning"])
async def risk_model_stats(model=Depends(get_risk_model)):
    """How the write-risk model has adapted: reviewed holds, precision, live thresholds, top risk signals."""
    return model.stats()
