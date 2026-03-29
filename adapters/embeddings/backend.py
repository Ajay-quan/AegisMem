"""Embedding backends for AegisMem - swappable by configuration."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import Any

import numpy as np

from core.exceptions import EmbeddingError

logger = logging.getLogger(__name__)


class EmbeddingBackend(ABC):
    """Base class for embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

    async def embed_single(self, text: str) -> list[float]:
        results = await self.embed([text])
        return results[0]

    def cosine_similarity(self, a: list[float], b: list[float]) -> float:
        va, vb = np.array(a), np.array(b)
        norm_a, norm_b = np.linalg.norm(va), np.linalg.norm(vb)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(va, vb) / (norm_a * norm_b))


class SentenceTransformerBackend(EmbeddingBackend):
    """Local sentence-transformers embedding backend."""

    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5", device: str = "cpu") -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None
        self._dim: int = 0

    def _load(self) -> None:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                logger.info(f"Loading embedding model: {self._model_name}")
                self._model = SentenceTransformer(self._model_name, device=self._device)
                self._dim = self._model.get_sentence_embedding_dimension() or 1024
                logger.info(f"Embedding model loaded, dim={self._dim}")
            except Exception as e:
                raise EmbeddingError(f"Failed to load sentence-transformer: {e}") from e

    @property
    def dimension(self) -> int:
        if not self._dim:
            self._load()
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        self._load()
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                lambda: self._model.encode(
                    texts,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist(),
            )
            return embeddings
        except Exception as e:
            raise EmbeddingError(f"Embedding failed: {e}") from e


class MockEmbeddingBackend(EmbeddingBackend):
    """Deterministic mock embedding for tests."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import hashlib, random
        results = []
        for text in texts:
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**32)
            rng = random.Random(seed)
            vec = [rng.gauss(0, 1) for _ in range(self._dim)]
            norm = sum(x**2 for x in vec) ** 0.5 or 1.0
            results.append([x / norm for x in vec])
        return results


@lru_cache(maxsize=1)
def get_embedding_backend(backend: str = "mock", model_name: str = "") -> EmbeddingBackend:
    """Factory for embedding backends (cached singleton)."""
    if backend == "sentence_transformers":
        model = model_name or "BAAI/bge-large-en-v1.5"
        return SentenceTransformerBackend(model_name=model)
    elif backend == "mock":
        return MockEmbeddingBackend()
    else:
        raise EmbeddingError(f"Unknown embedding backend: {backend}")
