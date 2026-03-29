"""Memory scoring — importance, recency, lexical, composite retrieval score.

All weights are driven by ``core.config.settings`` so nothing is hardcoded.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone

from core.config.settings import settings
from core.schemas.memory import MemoryItem, RetrievalCandidate


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ---------------------------------------------------------------------------
# Individual signal functions
# ---------------------------------------------------------------------------


def compute_recency_score(
    memory: MemoryItem,
    decay_hours: float | None = None,
) -> float:
    """Exponential decay recency score.

    Formula: exp(-Δhours / decay_hours).
    Default comes from ``settings.recency_decay_hours``.
    """
    dh = decay_hours or settings.recency_decay_hours
    ref_time = memory.updated_at or memory.created_at
    if ref_time.tzinfo is None:
        ref_time = ref_time.replace(tzinfo=timezone.utc)
    hours_elapsed = max((utcnow() - ref_time).total_seconds() / 3600.0, 0.0)
    # decay = exp(-t / decay_hours)
    return math.exp(-hours_elapsed / dh) if dh > 0 else 1.0


def compute_importance_heuristic(content: str) -> float:
    """Simple keyword-driven importance score based on content signals."""
    content_lower = content.lower()
    score = 0.4  # base

    high_signals = [
        "prefer", "always", "never", "important", "critical",
        "love", "hate", "goal", "objective", "plan", "deadline",
        "must", "need", "require", "password", "address", "phone",
        "birthday", "allerg", "medical", "work at", "job",
    ]
    medium_signals = [
        "like", "dislike", "want", "hope", "think", "believe",
        "usually", "often", "sometimes",
    ]

    for signal in high_signals:
        if signal in content_lower:
            score += 0.08

    for signal in medium_signals:
        if signal in content_lower:
            score += 0.04

    # Length bonus for detailed content
    if len(content) > 200:
        score += 0.05
    if len(content) > 500:
        score += 0.05

    return min(score, 1.0)


# ---------------------------------------------------------------------------
# Composite scoring
# ---------------------------------------------------------------------------


def compute_composite_score(
    semantic_score: float,
    recency_score: float,
    importance_score: float,
    access_count: int = 0,
    contradiction_penalty: float = 0.0,
    weights: dict[str, float] | None = None,
) -> float:
    """Hybrid retrieval composite score combining multiple signals.

    Default weights come from ``settings`` but can be overridden per call.
    """
    w = weights or {
        "semantic": settings.weight_semantic,
        "recency": settings.weight_recency,
        "importance": settings.weight_importance,
        "access": settings.weight_access,
    }

    access_score = math.log1p(access_count) / math.log1p(100)  # normalize

    score = (
        w.get("semantic", 0.40) * semantic_score
        + w.get("recency", 0.25) * recency_score
        + w.get("importance", 0.25) * importance_score
        + w.get("access", 0.10) * access_score
    )

    # Apply contradiction penalty 
    if contradiction_penalty > 0:
        score -= contradiction_penalty

    return max(0.0, min(1.0, score))


def compute_type_boost(memory: MemoryItem) -> float:
    """Small scoring boost based on memory layer and type.

    Semantic-layer memories (consolidated facts, procedures) receive a
    configurable boost since they represent higher-confidence knowledge.
    """
    boost = 0.0
    layer = getattr(memory, 'memory_layer', 'episodic')
    if (layer == 'semantic' or
            (hasattr(layer, 'value') and layer.value == 'semantic')):
        boost += settings.type_boost_semantic
    return boost


def rank_candidates(candidates: list[RetrievalCandidate]) -> list[RetrievalCandidate]:
    """Sort and assign ranks to retrieval candidates."""
    sorted_candidates = sorted(candidates, key=lambda c: c.composite_score, reverse=True)
    for i, c in enumerate(sorted_candidates):
        c.rank = i + 1
    return sorted_candidates


# ---------------------------------------------------------------------------
# High-level scoring entry point
# ---------------------------------------------------------------------------


def score_memory_for_retrieval(
    memory: MemoryItem,
    semantic_score: float,
    query_text: str = "",
) -> RetrievalCandidate:
    """Compute all signals and return a fully-scored RetrievalCandidate.

    Parameters
    ----------
    memory : MemoryItem
        The memory record to score.
    semantic_score : float
        Dense similarity returned by the vector store.
    query_text : str
        The original query string — needed for lexical scoring.
    """
    recency = compute_recency_score(memory)
    importance = memory.importance_score

    # Contradiction penalty: only apply when confidence exceeds threshold.
    contradiction_penalty = 0.0
    if memory.contradiction_status == "confirmed":
        confidence = memory.metadata.get("contradiction_confidence", 1.0)
        if confidence >= settings.contradiction_confidence_threshold:
            contradiction_penalty = settings.contradiction_penalty_weight

    composite = compute_composite_score(
        semantic_score=semantic_score,
        recency_score=recency,
        importance_score=importance,
        access_count=memory.access_count,
        contradiction_penalty=contradiction_penalty,
    )

    # Apply memory-type boost.
    type_boost = compute_type_boost(memory)
    composite = min(1.0, composite + type_boost)

    return RetrievalCandidate(
        memory=memory,
        semantic_score=semantic_score,
        recency_score=recency,
        importance_score=importance,
        composite_score=composite,
    )
