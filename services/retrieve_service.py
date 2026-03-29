"""Memory retrieval service — hybrid semantic + lexical + reranking pipeline.

Pipeline stages:
    1. Semantic search → broad candidate pool (top_n_candidates)
    2. Enrich from relational store + compute multi-signal scores
    3. Apply symbolic filters (time, type, importance)
    4. Second-stage reranking with diversity filtering
    5. Return top_k results
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any

from core.config.settings import settings
from core.schemas.memory import (
    MemoryItem, RetrievalQuery, RetrievalResult, RetrievalCandidate,
)
from domain.memory.scoring import score_memory_for_retrieval, rank_candidates
from domain.memory.reranker import HeuristicReranker

logger = logging.getLogger(__name__)


class RetrievalService:
    """Hybrid memory retrieval combining semantic search + lexical signals + reranking."""

    def __init__(
        self,
        relational_store: Any,
        vector_store: Any,
        embedding_backend: Any,
        graph_store: Any | None = None,
    ) -> None:
        self._db = relational_store
        self._vs = vector_store
        self._embed = embedding_backend
        self._graph = graph_store
        self._reranker = HeuristicReranker(
            diversity_threshold=settings.diversity_threshold,
        )

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Main retrieval entry point — full multi-stage pipeline."""
        start = time.time()

        # Stage 1: Broad semantic retrieval (over-retrieve).
        semantic_candidates = await self._semantic_search(query)

        # Stage 2: Enrich with full records + multi-signal scoring.
        enriched = await self._enrich_candidates(
            semantic_candidates, query,
        )

        # Stage 3: Symbolic filters (time range, type, importance).
        filtered = self._apply_filters(enriched, query)

        # Stage 4: Second-stage reranking with diversity filtering.
        reranked = self._reranker.rerank(
            candidates=filtered,
            query_text=query.query_text,
            top_k=query.top_k,
        )

        # Stage 5: Update access counts on returned results.
        for candidate in reranked:
            candidate.memory.bump_access()
            try:
                await self._db.update_memory(candidate.memory)
            except Exception:
                pass

        latency_ms = (time.time() - start) * 1000

        logger.info(
            f"Retrieved {len(reranked)}/{len(filtered)} memories for user={query.user_id} "
            f"in {latency_ms:.1f}ms (candidates={len(enriched)})"
        )

        return RetrievalResult(
            query=query,
            candidates=reranked,
            total_found=len(filtered),
            latency_ms=latency_ms,
        )

    async def _semantic_search(
        self, query: RetrievalQuery,
    ) -> list[tuple[str, float]]:
        """Run semantic search — return (memory_id, score) pairs."""
        try:
            query_embedding = await self._embed.embed_single(query.query_text)

            # Build vector filter.
            vector_filter: dict[str, Any] = {"user_id": query.user_id}
            if not query.include_archived:
                vector_filter["status"] = "active"

            # Fetch a broad candidate pool — refined by reranker later.
            n_candidates = max(
                settings.retrieval_top_n_candidates,
                query.top_k * 4,
            )

            results = await self._vs.search(
                query_vector=query_embedding,
                top_k=min(n_candidates, 50),
                filter=vector_filter,
            )
            return [(r.id, r.score) for r in results]
        except Exception as e:
            logger.warning(f"Semantic search failed, falling back to DB: {e}")
            return []

    async def _enrich_candidates(
        self,
        semantic_hits: list[tuple[str, float]],
        query: RetrievalQuery,
    ) -> list[RetrievalCandidate]:
        """Load full memory records from DB, compute multi-signal scores."""
        candidates = []
        for memory_id, semantic_score in semantic_hits:
            try:
                memory = await self._db.get_memory(memory_id)
                candidate = score_memory_for_retrieval(
                    memory, semantic_score, query_text=query.query_text,
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Could not enrich memory {memory_id}: {e}")

        # Fallback: if semantic search returned nothing, query DB directly.
        if not candidates:
            memories = await self._db.list_memories(
                user_id=query.user_id,
                namespace=query.namespace,
                limit=query.top_k * 2,
            )
            for memory in memories:
                candidate = score_memory_for_retrieval(
                    memory, 0.5, query_text=query.query_text,
                )
                candidates.append(candidate)

        return candidates

    def _apply_filters(
        self,
        candidates: list[RetrievalCandidate],
        query: RetrievalQuery,
    ) -> list[RetrievalCandidate]:
        """Apply symbolic filters to the candidate set."""
        filtered = []
        for c in candidates:
            m = c.memory

            # Importance threshold.
            if m.importance_score < query.min_importance:
                continue

            # Memory type filter.
            if query.memory_types and m.memory_type not in [
                mt.value if hasattr(mt, "value") else mt for mt in query.memory_types
            ]:
                continue

            # Memory layer filter.
            if query.memory_layers:
                layer_vals = [
                    ml.value if hasattr(ml, "value") else ml
                    for ml in query.memory_layers
                ]
                mem_layer = m.memory_layer if isinstance(m.memory_layer, str) else (
                    m.memory_layer.value if hasattr(m.memory_layer, 'value') else str(m.memory_layer)
                )
                if mem_layer not in layer_vals:
                    continue

            # Time range filter.
            if query.time_range_start and m.event_time:
                event_time = m.event_time
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                start = query.time_range_start
                if start.tzinfo is None:
                    start = start.replace(tzinfo=timezone.utc)
                if event_time < start:
                    continue

            if query.time_range_end and m.event_time:
                event_time = m.event_time
                if event_time.tzinfo is None:
                    event_time = event_time.replace(tzinfo=timezone.utc)
                end = query.time_range_end
                if end.tzinfo is None:
                    end = end.replace(tzinfo=timezone.utc)
                if event_time > end:
                    continue

            # Namespace filter.
            if query.namespace and m.namespace and not m.namespace.startswith(query.namespace):
                continue

            filtered.append(c)

        return filtered

    async def get_context_window(
        self,
        user_id: str,
        query_text: str,
        top_k: int = 5,
        namespace: str = "",
    ) -> str:
        """Retrieve memories and format them as a context window string."""
        result = await self.retrieve(
            RetrievalQuery(
                query_text=query_text,
                user_id=user_id,
                namespace=namespace,
                top_k=top_k,
            )
        )

        if not result.candidates:
            return ""

        lines = ["# Relevant Memories\n"]
        for i, c in enumerate(result.candidates, 1):
            m = c.memory
            lines.append(
                f"{i}. [{m.memory_type}] {m.content} "
                f"(importance={m.importance_score:.2f}, score={c.composite_score:.2f})"
            )

        return "\n".join(lines)
