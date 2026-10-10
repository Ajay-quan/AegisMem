"""Second-stage reranking and diversity filtering for retrieval candidates.

Architecture:
    BaseReranker          — abstract interface for pluggable rerankers
    HeuristicReranker     — config-driven composite reranking (default)
    CrossEncoderReranker  — stub for future cross-encoder integration
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from core.schemas.memory import RetrievalCandidate

logger = logging.getLogger(__name__)


class BaseReranker(ABC):
    """Abstract reranker interface."""

    @abstractmethod
    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query_text: str,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        """Rerank candidates and return the top-k."""
        ...


class HeuristicReranker(BaseReranker):
    """Config-driven heuristic reranker with diversity filtering.

    1. Sort by composite_score (already computed by scoring module).
    2. Apply diversity filtering to suppress near-duplicate content.
    3. Assign final ranks and truncate to top_k.
    """

    def __init__(self, diversity_threshold: float = 0.92) -> None:
        self._diversity_threshold = diversity_threshold

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query_text: str,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        # Sort by composite score (descending).
        sorted_candidates = sorted(
            candidates, key=lambda c: c.composite_score, reverse=True,
        )

        # Diversity filter: greedy selection, skip near-duplicates.
        selected = self._diversity_filter(sorted_candidates)

        # Assign ranks and truncate.
        for i, c in enumerate(selected):
            c.rank = i + 1
        return selected[:top_k]

    def _diversity_filter(
        self, candidates: list[RetrievalCandidate],
    ) -> list[RetrievalCandidate]:
        """Greedy diversity selection — skip candidates too similar to already-selected ones."""
        if not candidates:
            return []

        selected: list[RetrievalCandidate] = []
        selected_texts: list[str] = []

        for c in candidates:
            if self._is_too_similar(c.memory.content, selected_texts):
                logger.debug(
                    f"Diversity filter: suppressed memory {c.memory.memory_id} "
                    f"(score={c.composite_score:.3f})"
                )
                continue
            selected.append(c)
            selected_texts.append(c.memory.content)

        return selected

    def _is_too_similar(self, content: str, existing_texts: list[str]) -> bool:
        """Check if content is too similar to any already-selected text."""
        if not existing_texts:
            return False

        content_tokens = set(content.lower().split())
        if not content_tokens:
            return False

        for existing in existing_texts:
            existing_tokens = set(existing.lower().split())
            if not existing_tokens:
                continue
            # Jaccard similarity on raw word tokens.
            intersection = len(content_tokens & existing_tokens)
            union = len(content_tokens | existing_tokens)
            if union > 0 and (intersection / union) > self._diversity_threshold:
                return True
        return False


class CrossEncoderReranker(BaseReranker):
    """Stub for future cross-encoder reranking (e.g. ms-marco-MiniLM).

    Not implemented — exists as an architecture hook for when heavier
    reranking is justified by the deployment context.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        self._model_name = model_name

    def rerank(
        self,
        candidates: list[RetrievalCandidate],
        query_text: str,
        top_k: int,
    ) -> list[RetrievalCandidate]:
        raise NotImplementedError(
            "CrossEncoderReranker is a placeholder. Install sentence-transformers "
            "and implement cross-encoder scoring to use this."
        )
