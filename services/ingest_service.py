"""Memory ingestion service - process observations into stored memories."""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from core.schemas.memory import (
    MemoryItem, MemoryType, MemoryLayer, SourceType, Observation, MemoryStatus,
)
from domain.memory.scoring import compute_importance_heuristic

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class IngestionService:
    """Processes raw observations into structured MemoryItems."""

    def __init__(
        self,
        relational_store: Any,
        vector_store: Any,
        embedding_backend: Any,
        graph_store: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self._db = relational_store
        self._vs = vector_store
        self._embed = embedding_backend
        self._graph = graph_store
        self._llm = llm_client

    async def ingest_observation(
        self,
        observation: Observation,
        memory_type: MemoryType = MemoryType.OBSERVATION,
        importance_override: float | None = None,
    ) -> MemoryItem:
        """Convert an observation into a stored memory."""
        importance = importance_override or compute_importance_heuristic(observation.content)

        memory = MemoryItem(
            namespace=f"user:{observation.user_id}",
            user_id=observation.user_id,
            agent_id=observation.agent_id,
            memory_type=memory_type,
            memory_layer=self._classify_layer(memory_type),
            content=observation.content,
            source_type=observation.source_type,
            source_ref=observation.session_id,
            event_time=observation.observed_at,
            importance_score=importance,
            metadata=observation.metadata,
        )

        # Save to relational store
        await self._db.save_memory(memory)

        # Embed and save to vector store
        try:
            embedding = await self._embed.embed_single(observation.content)
            memory.content_embedding = embedding
            await self._vs.upsert(
                id=memory.memory_id,
                vector=embedding,
                payload={
                    "user_id": memory.user_id,
                    "namespace": memory.namespace,
                    "memory_type": memory.memory_type,
                    "memory_layer": memory.memory_layer,
                    "content": memory.content[:500],
                    "importance_score": memory.importance_score,
                    "status": "active",
                },
            )
        except Exception as e:
            logger.warning(f"Embedding failed for memory {memory.memory_id}: {e}")

        # Create graph node
        if self._graph and self._graph.is_available():
            try:
                await self._graph.create_memory_node(
                    memory.memory_id,
                    {
                        "user_id": memory.user_id,
                        "memory_type": memory.memory_type,
                        "importance": memory.importance_score,
                    },
                )
            except Exception as e:
                logger.warning(f"Graph node creation failed: {e}")

        logger.info(
            f"Ingested memory {memory.memory_id} for user {memory.user_id} "
            f"(type={memory.memory_type}, importance={importance:.2f})"
        )
        return memory

    async def ingest_batch(
        self,
        observations: list[Observation],
    ) -> list[MemoryItem]:
        """Batch ingest multiple observations."""
        memories = []
        for obs in observations:
            try:
                mem = await self.ingest_observation(obs)
                memories.append(mem)
            except Exception as e:
                logger.error(f"Failed to ingest observation {obs.observation_id}: {e}")
        return memories

    async def ingest_text(
        self,
        text: str,
        user_id: str,
        session_id: str = "",
        agent_id: str = "",
        memory_type: MemoryType = MemoryType.OBSERVATION,
        source_type: SourceType = SourceType.USER_MESSAGE,
        metadata: dict[str, Any] | None = None,
        importance_override: float | None = None,
    ) -> MemoryItem:
        """Convenience method to ingest plain text."""
        obs = Observation(
            session_id=session_id,
            user_id=user_id,
            agent_id=agent_id,
            content=text,
            source_type=source_type,
            metadata=metadata or {},
        )
        return await self.ingest_observation(obs, memory_type=memory_type, importance_override=importance_override)

    @staticmethod
    def _classify_layer(memory_type: MemoryType) -> MemoryLayer:
        """Auto-classify memory layer based on type."""
        semantic_types = {
            MemoryType.FACT, MemoryType.PROCEDURE,
            MemoryType.REFLECTION, MemoryType.SUMMARY,
        }
        mt = memory_type if isinstance(memory_type, MemoryType) else MemoryType(memory_type)
        return MemoryLayer.SEMANTIC if mt in semantic_types else MemoryLayer.EPISODIC
