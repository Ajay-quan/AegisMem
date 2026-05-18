"""Persistent FAISS vector store for local semantic retrieval."""
from __future__ import annotations

import math
import pickle
from pathlib import Path
from threading import RLock
from typing import Any

from adapters.vector_store.base import VectorSearchResult


class FaissVectorStore:
    """FAISS-backed vector database persisted to local disk.

    Docker installs faiss-cpu. On developer machines without faiss/numpy, this
    falls back to exact pure-Python cosine search while preserving the same API.
    """

    def __init__(self, index_path: str | Path, metadata_path: str | Path) -> None:
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._index: Any = None
        self._ids: list[str] = []
        self._payloads: dict[str, dict[str, Any]] = {}
        self._dimension = 0
        self._faiss: Any = None
        self._numpy: Any = None

    def initialize(self, dimension: int) -> None:
        with self._lock:
            self._dimension = dimension
            try:
                import faiss
                import numpy as np
                self._faiss = faiss
                self._numpy = np
            except Exception:
                self._faiss = None
                self._numpy = None
            if self._faiss and self.index_path.exists() and self.metadata_path.exists():
                self._index = self._faiss.read_index(str(self.index_path))
                self._load_metadata()
            else:
                self._index = self._faiss.IndexFlatIP(dimension) if self._faiss else None
                self._load_metadata()
                self._persist()

    def upsert(self, id: str, vector: list[float], payload: dict[str, Any]) -> None:
        with self._lock:
            if id in self._payloads:
                self.delete(id)
            payload = {**payload, "_vector": vector}
            if self._faiss:
                self._index.add(self._normalize_np([vector]))
            self._ids.append(id)
            self._payloads[id] = payload
            self._persist()

    def search(self, query_vector: list[float], top_k: int = 10, filter: dict[str, Any] | None = None) -> list[VectorSearchResult]:
        with self._lock:
            if not self._ids:
                return []
            if self._faiss and self._index.ntotal:
                scores, indices = self._index.search(self._normalize_np([query_vector]), min(top_k * 4, self._index.ntotal))
                pairs = [(float(score), int(idx)) for score, idx in zip(scores[0], indices[0])]
            else:
                pairs = sorted(
                    (
                        (self._cosine(query_vector, self._payloads[memory_id]["_vector"]), idx)
                        for idx, memory_id in enumerate(self._ids)
                    ),
                    key=lambda item: item[0],
                    reverse=True,
                )[: top_k * 4]
            results: list[VectorSearchResult] = []
            for score, idx in pairs:
                if idx < 0 or idx >= len(self._ids):
                    continue
                memory_id = self._ids[idx]
                payload = self._payloads.get(memory_id, {})
                if filter and any(payload.get(k) != v for k, v in filter.items()):
                    continue
                results.append(VectorSearchResult(id=memory_id, score=score, payload=payload))
                if len(results) >= top_k:
                    break
            return results

    def get(self, id: str) -> VectorSearchResult | None:
        payload = self._payloads.get(id)
        if payload is None:
            return None
        return VectorSearchResult(id=id, score=1.0, payload=payload)

    def delete(self, id: str) -> None:
        with self._lock:
            if id not in self._payloads:
                return
            kept_ids = [memory_id for memory_id in self._ids if memory_id != id]
            kept_payloads = {memory_id: self._payloads[memory_id] for memory_id in kept_ids}
            self._ids = []
            self._payloads = {}
            self._index = self._faiss.IndexFlatIP(self._dimension) if self._faiss else None
            for memory_id in kept_ids:
                payload = kept_payloads[memory_id]
                if self._faiss:
                    self._index.add(self._normalize_np([payload["_vector"]]))
                self._ids.append(memory_id)
                self._payloads[memory_id] = payload
            self._persist()

    def _normalize_np(self, vectors: list[list[float]]) -> Any:
        array = self._numpy.asarray(vectors, dtype="float32")
        norms = self._numpy.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def _cosine(self, left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left)) or 1.0
        right_norm = math.sqrt(sum(b * b for b in right)) or 1.0
        return dot / (left_norm * right_norm)

    def _load_metadata(self) -> None:
        if self.metadata_path.exists():
            with self.metadata_path.open("rb") as handle:
                metadata = pickle.load(handle)
            self._ids = metadata.get("ids", [])
            self._payloads = metadata.get("payloads", {})

    def _persist(self) -> None:
        if self._faiss and self._index is not None:
            self._faiss.write_index(self._index, str(self.index_path))
        with self.metadata_path.open("wb") as handle:
            pickle.dump({"ids": self._ids, "payloads": self._payloads}, handle)
