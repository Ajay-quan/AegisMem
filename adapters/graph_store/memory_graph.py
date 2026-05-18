"""Persistent local memory graph with weighted breadth-first traversal."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path
from threading import RLock
from typing import Any


class LocalMemoryGraph:
    """Small persistent graph for related-memory traversal in the demo deployment."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._nodes: dict[str, dict[str, Any]] = {}
        self._edges: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._load()

    def add_memory(self, memory_id: str, properties: dict[str, Any]) -> None:
        with self._lock:
            self._nodes[memory_id] = properties
            self._persist()

    def connect(self, source_id: str, target_id: str, relation: str = "RELATED", weight: float = 1.0) -> None:
        with self._lock:
            self._edges[source_id].append({"target": target_id, "relation": relation, "weight": weight})
            self._edges[target_id].append({"target": source_id, "relation": relation, "weight": weight})
            self._persist()

    def traverse(self, start_id: str, depth: int = 2, relation: str | None = None) -> list[dict[str, Any]]:
        """Weighted BFS traversal returning reachable memories ordered by path score."""
        with self._lock:
            if start_id not in self._nodes:
                return []
            seen = {start_id}
            queue = deque([(start_id, 0, 1.0, [])])
            results: list[dict[str, Any]] = []
            while queue:
                current, distance, score, path = queue.popleft()
                if distance >= depth:
                    continue
                for edge in self._edges.get(current, []):
                    if relation and edge["relation"] != relation:
                        continue
                    target = edge["target"]
                    if target in seen:
                        continue
                    seen.add(target)
                    next_path = [*path, {"from": current, **edge}]
                    next_score = score * float(edge.get("weight", 1.0))
                    results.append(
                        {
                            "memory_id": target,
                            "distance": distance + 1,
                            "path_score": next_score,
                            "relation": edge["relation"],
                            "properties": self._nodes.get(target, {}),
                            "path": next_path,
                        }
                    )
                    queue.append((target, distance + 1, next_score, next_path))
            return sorted(results, key=lambda item: (item["distance"], -item["path_score"]))

    def delete(self, memory_id: str) -> None:
        with self._lock:
            self._nodes.pop(memory_id, None)
            self._edges.pop(memory_id, None)
            for edges in self._edges.values():
                edges[:] = [edge for edge in edges if edge["target"] != memory_id]
            self._persist()

    def _load(self) -> None:
        if not self.path.exists():
            self._persist()
            return
        raw = json.loads(self.path.read_text() or "{}")
        self._nodes = dict(raw.get("nodes", {}))
        self._edges = defaultdict(list, {k: list(v) for k, v in raw.get("edges", {}).items()})

    def _persist(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"nodes": self._nodes, "edges": self._edges}, indent=2, sort_keys=True))
        tmp.replace(self.path)
