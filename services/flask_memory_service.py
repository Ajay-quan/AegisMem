"""Synchronous Flask-facing memory service."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from adapters.embeddings.langchain_adapter import StatefulEmbeddings
from adapters.graph_store.memory_graph import LocalMemoryGraph
from adapters.relational_store.json_store import JsonMemoryStore, StoredMemory
from adapters.vector_store.faiss_store import FaissVectorStore
from services.cache_index import HotMemoryIndex


class DeterministicEmbeddingBackend:
    """Small lexical hashing embedding backend for zero-key local demos.

    This avoids network downloads while still making similar queries and memories
    share dimensions through normalized token hashing.
    """

    dimension = 384

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    async def embed_single(self, text: str) -> list[float]:
        return self._embed_text(text)

    def _embed_text(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        vector = [0.0] * self.dimension
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = sum(value * value for value in vector) ** 0.5 or 1.0
        return [value / norm for value in vector]

    def _tokens(self, text: str) -> list[str]:
        cleaned = []
        current = []
        for char in text.lower():
            if char.isalnum():
                current.append(char)
            elif current:
                cleaned.append("".join(current))
                current = []
        if current:
            cleaned.append("".join(current))
        stopwords = {"a", "an", "and", "for", "in", "is", "of", "on", "the", "to", "with"}
        return [token for token in cleaned if token not in stopwords]


class FlaskMemoryService:
    """Coordinates ingestion, lifecycle updates, semantic search, and graph traversal."""

    def __init__(
        self,
        data_dir: str | Path = "data",
        embedding_backend: str = "mock",
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        vector_backend: str = "faiss",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        if embedding_backend == "sentence_transformers":
            from adapters.embeddings.backend import SentenceTransformerBackend
            backend: Any = SentenceTransformerBackend(embedding_model)
        else:
            backend = DeterministicEmbeddingBackend()
        self.embeddings = StatefulEmbeddings(backend)
        self.store = JsonMemoryStore(self.data_dir / "memories.json")
        if vector_backend == "chroma":
            from adapters.vector_store.chroma_store import ChromaVectorStore
            self.vector_store = ChromaVectorStore(self.data_dir / "chroma")
        else:
            self.vector_store = FaissVectorStore(
                self.data_dir / "faiss" / "memories.index",
                self.data_dir / "faiss" / "metadata.pkl",
            )
        self.vector_store.initialize(backend.dimension)
        self.graph = LocalMemoryGraph(self.data_dir / "memory_graph.json")
        self.hot_index = HotMemoryIndex()

    def ingest(
        self,
        *,
        content: str,
        user_id: str,
        key: str | None = None,
        related_memory_ids: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance_score: float = 0.5,
    ) -> StoredMemory:
        memory = self.store.create(
            content=content,
            user_id=user_id,
            key=key,
            metadata=metadata,
            tags=tags,
            importance_score=importance_score,
        )
        vector = self.embeddings.embed_query(content)
        self.vector_store.upsert(
            memory.memory_id,
            vector,
            {
                "user_id": user_id,
                "content": content,
                "key": memory.key,
                "status": "active",
                "importance_score": memory.importance_score,
            },
        )
        self.graph.add_memory(memory.memory_id, {"user_id": user_id, "content": content, "key": memory.key})
        for related_id in related_memory_ids or []:
            self.graph.connect(memory.memory_id, related_id, relation="RELATED", weight=0.85)
        self._cache(memory)
        return memory

    def retrieve(self, *, query: str, user_id: str, top_k: int = 5) -> list[dict[str, Any]]:
        vector = self.embeddings.embed_query(query)
        hits = self.vector_store.search(vector, top_k=top_k, filter={"user_id": user_id, "status": "active"})
        results = []
        for rank, hit in enumerate(hits, start=1):
            memory = self.store.get(hit.id)
            if not memory:
                continue
            self.store.bump_access(memory.memory_id)
            self._cache(memory)
            results.append(
                {
                    "rank": rank,
                    "memory_id": memory.memory_id,
                    "content": memory.content,
                    "score": hit.score,
                    "key": memory.key,
                    "metadata": memory.metadata,
                    "tags": memory.tags,
                }
            )
        return results

    def get_by_key(self, *, user_id: str, key: str) -> StoredMemory | None:
        memory = self.store.get_by_key(user_id, key)
        if memory:
            self.store.bump_access(memory.memory_id)
            self._cache(memory)
        return memory

    def update(self, memory_id: str, **changes: Any) -> StoredMemory | None:
        memory = self.store.update(memory_id, **changes)
        if not memory:
            return None
        vector = self.embeddings.embed_query(memory.content)
        self.vector_store.upsert(
            memory.memory_id,
            vector,
            {
                "user_id": memory.user_id,
                "content": memory.content,
                "key": memory.key,
                "status": "active",
                "importance_score": memory.importance_score,
            },
        )
        self.graph.add_memory(memory.memory_id, {"user_id": memory.user_id, "content": memory.content, "key": memory.key})
        self._cache(memory)
        return memory

    def versions(self, memory_id: str) -> list[dict[str, Any]]:
        return self.store.versions(memory_id)

    def export_payload(self, include_deleted: bool = True) -> dict[str, Any]:
        return self.store.export_payload(include_deleted=include_deleted)

    def import_payload(self, payload: dict[str, Any], replace: bool = False) -> int:
        imported = self.store.import_payload(payload, replace=replace)
        for memory in self.store.list_all(include_deleted=False):
            vector = self.embeddings.embed_query(memory.content)
            self.vector_store.upsert(
                memory.memory_id,
                vector,
                {
                    "user_id": memory.user_id,
                    "content": memory.content,
                    "key": memory.key,
                    "status": "active",
                    "importance_score": memory.importance_score,
                },
            )
            self.graph.add_memory(memory.memory_id, {"user_id": memory.user_id, "content": memory.content, "key": memory.key})
            self._cache(memory)
        return imported

    def delete(self, memory_id: str) -> bool:
        deleted = self.store.delete(memory_id)
        if deleted:
            self.vector_store.delete(memory_id)
            self.graph.delete(memory_id)
            self.hot_index.delete(memory_id)
        return deleted

    def traverse(self, memory_id: str, depth: int = 2) -> list[dict[str, Any]]:
        return self.graph.traverse(memory_id, depth=depth)

    def _cache(self, memory: StoredMemory) -> None:
        priority = memory.importance_score + min(memory.access_count / 10.0, 1.0)
        self.hot_index.upsert(
            memory.memory_id,
            priority=priority,
            updated_at=memory.updated_at,
            payload={"key": memory.key, "content": memory.content},
        )
