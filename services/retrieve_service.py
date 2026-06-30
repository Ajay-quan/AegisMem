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
import uuid
from datetime import datetime, timezone
from typing import Any

from core.config.settings import settings
from core.schemas.memory import (
    MemoryItem, RetrievalQuery, RetrievalResult, RetrievalCandidate,
)
from domain.memory.scoring import score_memory_for_retrieval, rank_candidates
from domain.memory.lexical import (
    BM25Index, reciprocal_rank_fusion, normalize_scores,
)
from domain.memory.reranker import build_reranker
from core.observability import metrics

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
        self._reranker = build_reranker(settings)

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Main retrieval entry point — full multi-stage pipeline."""
        start = time.time()
        query_id = uuid.uuid4().hex

        # Stateful-CL: when enabled, pull the learned per-namespace ranking weights
        # so the composite score adapts to feedback. Disabled => weights=None =>
        # static settings weights => original behavior.
        learned_weights = self._learned_weights(query)

        # Stage 1a: Broad dense semantic retrieval (over-retrieve).
        semantic_candidates = await self._semantic_search(query)

        # Stage 1b: Sparse lexical retrieval (BM25) when hybrid mode is on.
        lexical_candidates: list[tuple[str, float]] = []
        if settings.hybrid_retrieval_enabled:
            lexical_candidates = await self._lexical_search(query)

        # Stage 1c: Fuse dense + sparse rankings with Reciprocal Rank Fusion.
        fused_ids, semantic_map, lexical_map = self._fuse(
            semantic_candidates, lexical_candidates, query,
        )

        # Stage 2: Enrich with full records + multi-signal scoring.
        enriched = await self._enrich_candidates(
            fused_ids, semantic_map, lexical_map, query, weights=learned_weights,
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

        # Stage 6 (Stateful-CL): log the served interaction to the replay buffer so
        # a later /feedback call can turn it into labeled training examples.
        self._log_interaction(query_id, query, reranked)

        latency_s = time.time() - start
        latency_ms = latency_s * 1000

        mode = "hybrid" if (settings.hybrid_retrieval_enabled and lexical_candidates) else "dense"
        metrics.observe_retrieval(mode, latency_s)
        metrics.observe_retrieval_results(len(reranked), len(filtered))
        logger.info(
            f"Retrieved {len(reranked)}/{len(filtered)} memories for user={query.user_id} "
            f"in {latency_ms:.1f}ms (mode={mode}, dense={len(semantic_candidates)}, "
            f"lexical={len(lexical_candidates)}, fused={len(enriched)})"
        )

        return RetrievalResult(
            query=query,
            candidates=reranked,
            total_found=len(filtered),
            latency_ms=latency_ms,
            query_id=query_id if settings.continual_learning_enabled else "",
        )

    # ------------------------------------------------------------- Stateful-CL
    def _learned_weights(self, query: RetrievalQuery) -> dict[str, float] | None:
        """Fetch learned per-namespace ranking weights when CL is enabled."""
        if not settings.continual_learning_enabled:
            return None
        try:
            from domain.learning.registry import get_ranking_policy
            namespace = query.namespace or f"user:{query.user_id}"
            return get_ranking_policy().weights(namespace)
        except Exception as e:  # never let learning break serving
            logger.debug(f"Learned-weights lookup failed: {e}")
            return None

    def _log_interaction(
        self,
        query_id: str,
        query: RetrievalQuery,
        served: list[RetrievalCandidate],
    ) -> None:
        """Record served candidates + features to the replay buffer."""
        if not settings.continual_learning_enabled or not served:
            return
        try:
            from domain.learning.registry import get_replay_buffer
            from domain.learning.features import extract_features
            from domain.learning.replay import CandidateRecord

            namespace = query.namespace or f"user:{query.user_id}"
            records = [
                CandidateRecord(
                    memory_id=c.memory.memory_id,
                    features=extract_features(c),
                    served_rank=c.rank,
                    score=c.composite_score,
                )
                for c in served
            ]
            get_replay_buffer().log(
                query_id=query_id,
                user_id=query.user_id,
                namespace=namespace,
                query_text=query.query_text,
                candidates=records,
            )
        except Exception as e:  # never let learning break serving
            logger.debug(f"Replay logging failed: {e}")

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

    async def _lexical_search(
        self, query: RetrievalQuery,
    ) -> list[tuple[str, float]]:
        """Sparse BM25 search over the user's memory corpus.

        Returns ``(memory_id, bm25_score)`` ranked high→low. Pure-Python and
        infra-free; complements dense search on rare tokens, names, and IDs.
        """
        try:
            memories = await self._db.list_memories(
                user_id=query.user_id,
                namespace=query.namespace,
                limit=settings.lexical_candidate_pool,
            )
            if not memories:
                return []
            index = BM25Index.build(
                [(m.memory_id, m.content) for m in memories],
                k1=settings.bm25_k1,
                b=settings.bm25_b,
            )
            pool = max(settings.retrieval_top_n_candidates, query.top_k * 4)
            return index.search(query.query_text, top_k=pool)
        except Exception as e:
            logger.warning(f"Lexical search failed: {e}")
            return []

    def _fuse(
        self,
        semantic_hits: list[tuple[str, float]],
        lexical_hits: list[tuple[str, float]],
        query: RetrievalQuery,
    ) -> tuple[list[str], dict[str, float], dict[str, float]]:
        """Fuse dense + sparse rankings with RRF.

        Returns the fused ordered id list plus per-id semantic and normalized
        lexical score maps for downstream multi-signal scoring.
        """
        semantic_map = {mid: s for mid, s in semantic_hits}
        lexical_map = normalize_scores(lexical_hits)

        if not lexical_hits:
            return [mid for mid, _ in semantic_hits], semantic_map, lexical_map

        fused = reciprocal_rank_fusion(
            [
                [mid for mid, _ in semantic_hits],
                [mid for mid, _ in lexical_hits],
            ],
            k=settings.rrf_k,
        )
        fused_ids = [mid for mid, _ in fused]
        return fused_ids, semantic_map, lexical_map

    async def _enrich_candidates(
        self,
        fused_ids: list[str],
        semantic_map: dict[str, float],
        lexical_map: dict[str, float],
        query: RetrievalQuery,
        weights: dict[str, float] | None = None,
    ) -> list[RetrievalCandidate]:
        """Load full memory records from DB, compute multi-signal scores.

        ``weights`` overrides the static ranking weights when supplied (the
        learned Stateful-CL per-namespace policy); ``None`` keeps static behavior.
        """
        candidates = []
        for memory_id in fused_ids:
            try:
                memory = await self._db.get_memory(memory_id)
                candidate = score_memory_for_retrieval(
                    memory,
                    semantic_map.get(memory_id, 0.0),
                    query_text=query.query_text,
                    lexical_score=lexical_map.get(memory_id, 0.0),
                    weights=weights,
                )
                candidates.append(candidate)
            except Exception as e:
                logger.debug(f"Could not enrich memory {memory_id}: {e}")

        # Fallback: if both retrieval arms returned nothing, query DB directly.
        if not candidates:
            memories = await self._db.list_memories(
                user_id=query.user_id,
                namespace=query.namespace,
                limit=query.top_k * 2,
            )
            for memory in memories:
                candidate = score_memory_for_retrieval(
                    memory, 0.5, query_text=query.query_text, weights=weights,
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
