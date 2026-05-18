"""Latency-oriented in-process indexes.

HotMemoryIndex demonstrates concrete use of a priority queue, hash map, and
recency tree for retrieval optimization.
"""
from __future__ import annotations

import heapq
from bisect import insort
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class HotMemory:
    memory_id: str
    priority: float
    updated_at: str
    payload: dict[str, Any]


class HotMemoryIndex:
    """Hot-memory cache using heapq, dict hash maps, and a sorted recency tree."""

    def __init__(self, capacity: int = 512) -> None:
        self.capacity = capacity
        self._by_id: dict[str, HotMemory] = {}
        self._priority_queue: list[tuple[float, str]] = []
        self._recency_tree: list[tuple[str, str]] = []

    def upsert(self, memory_id: str, *, priority: float, updated_at: str, payload: dict[str, Any]) -> None:
        item = HotMemory(memory_id=memory_id, priority=priority, updated_at=updated_at, payload=payload)
        self._by_id[memory_id] = item
        heapq.heappush(self._priority_queue, (priority, memory_id))
        insort(self._recency_tree, (updated_at, memory_id))
        self._evict_if_needed()

    def get(self, memory_id: str) -> HotMemory | None:
        return self._by_id.get(memory_id)

    def recent(self, limit: int = 10) -> list[HotMemory]:
        results: list[HotMemory] = []
        for _, memory_id in reversed(self._recency_tree):
            item = self._by_id.get(memory_id)
            if item and item not in results:
                results.append(item)
            if len(results) >= limit:
                break
        return results

    def delete(self, memory_id: str) -> None:
        self._by_id.pop(memory_id, None)

    def _evict_if_needed(self) -> None:
        while len(self._by_id) > self.capacity and self._priority_queue:
            _, memory_id = heapq.heappop(self._priority_queue)
            if memory_id in self._by_id:
                self._by_id.pop(memory_id, None)
