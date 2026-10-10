"""Retrieval feedback: closes the loop between retrieval and continual learning.

Flow:
    /retrieve  -> RetrievalLedger.record() stores the signals each memory was scored with
    /feedback  -> FeedbackService.submit() looks those signals up, updates per-memory
                  usefulness counters, and trains the per-user OnlineWeightLearner.

Scoring the signals the user actually saw, rather than recomputing them later, keeps
training labels consistent even after recency or access counts change.
"""
from __future__ import annotations

import logging
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from core.schemas.memory import RetrievalResult
from domain.learning import OnlineWeightLearner
from domain.learning.trust_calibrator import memory_source_origin

logger = logging.getLogger(__name__)


@dataclass
class LedgerEntry:
    user_id: str
    signals: dict[str, dict[str, float]]  # memory_id -> signal vector
    consumed: bool = False


class RetrievalLedger:
    """Bounded LRU of recent retrievals, keyed by retrieval_id."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._entries: OrderedDict[str, LedgerEntry] = OrderedDict()

    def record(self, result: RetrievalResult) -> None:
        signals = {
            c.memory.memory_id: {
                "semantic": c.semantic_score,
                "recency": c.recency_score,
                "importance": c.importance_score,
                "access": min(1.0, math.log1p(c.memory.access_count) / math.log1p(100)),
            }
            for c in result.candidates
        }
        self._entries[result.retrieval_id] = LedgerEntry(result.query.user_id, signals)
        self._entries.move_to_end(result.retrieval_id)
        while len(self._entries) > self._capacity:
            self._entries.popitem(last=False)

    def get(self, retrieval_id: str) -> LedgerEntry | None:
        return self._entries.get(retrieval_id)


class FeedbackError(ValueError):
    pass


class FeedbackService:
    def __init__(
        self,
        relational_store: Any,
        ledger: RetrievalLedger,
        learner: OnlineWeightLearner,
        trust_calibrator: Any | None = None,
    ) -> None:
        self._db = relational_store
        self._ledger = ledger
        self._learner = learner
        self._calibrator = trust_calibrator

    async def submit(
        self,
        retrieval_id: str,
        user_id: str,
        used_memory_ids: list[str],
    ) -> dict[str, Any]:
        """Record which retrieved memories were used. Others count as not used.

        Each retrieval accepts feedback once, so retries cannot double-train.
        """
        entry = self._ledger.get(retrieval_id)
        if entry is None:
            raise FeedbackError("unknown or expired retrieval_id")
        if entry.user_id != user_id:
            raise FeedbackError("retrieval_id does not belong to this user")
        if entry.consumed:
            raise FeedbackError("feedback already submitted for this retrieval")

        used = set(used_memory_ids)
        unknown = used - entry.signals.keys()
        if unknown:
            raise FeedbackError(f"memory ids not in this retrieval: {sorted(unknown)}")

        examples = []
        for memory_id, signals in entry.signals.items():
            was_used = memory_id in used
            examples.append((signals, was_used))
            try:
                memory = await self._db.get_memory(memory_id)
                memory.retrieval_count += 1
                if was_used:
                    memory.successful_retrieval_count += 1
                await self._db.update_memory(memory)
                if was_used and self._calibrator is not None and "provenance" in memory.metadata:
                    source, origin = memory_source_origin(memory.metadata, memory.source_type)
                    await self._calibrator.record(source, origin, "used")
            except Exception as e:
                logger.warning(f"Feedback counter update failed for {memory_id}: {e}")

        entry.consumed = True
        state = await self._learner.observe(user_id, examples)
        active = state.events >= self._learner.min_events
        return {
            "accepted": len(examples),
            "used": len(used),
            "learning_active": active,
            "weights": state.to_dict()["weights"],
        }
