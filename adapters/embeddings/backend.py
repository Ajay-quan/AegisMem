"""Embedding backends for stateful.ai - swappable by configuration."""
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


class OpenAIEmbeddingBackend(EmbeddingBackend):
    """OpenAI embeddings backend (e.g. text-embedding-3-small/large).

    Lazily constructs the client so importing this module never requires the
    ``openai`` package or a key; both are only needed when this backend is
    actually selected and used.
    """

    _MODEL_DIMS = {
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
        "text-embedding-ada-002": 1536,
    }

    def __init__(self, model_name: str = "text-embedding-3-small",
                 api_key: str | None = None) -> None:
        self._model_name = model_name
        self._api_key = api_key
        self._client: Any = None
        self._dim = self._MODEL_DIMS.get(model_name, 1536)

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI
            except Exception as e:  # pragma: no cover - optional dep guard
                raise EmbeddingError(
                    "OpenAI embedding backend requires the 'openai' package "
                    "(pip install \"stateful_ai[llm]\" or pip install openai)."
                ) from e
            self._client = OpenAI(api_key=self._api_key) if self._api_key else OpenAI()
        return self._client

    @property
    def dimension(self) -> int:
        return self._dim

    async def embed(self, texts: list[str]) -> list[list[float]]:
        import asyncio
        client = self._get_client()
        try:
            loop = asyncio.get_event_loop()
            resp = await loop.run_in_executor(
                None,
                lambda: client.embeddings.create(model=self._model_name, input=texts),
            )
            return [d.embedding for d in resp.data]
        except Exception as e:  # pragma: no cover - network path
            raise EmbeddingError(f"OpenAI embedding failed: {e}") from e


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


@lru_cache(maxsize=4)
def get_embedding_backend(backend: str = "mock", model_name: str = "") -> EmbeddingBackend:
    """Factory for embedding backends (cached per (backend, model))."""
    if backend == "sentence_transformers":
        model = model_name or "BAAI/bge-large-en-v1.5"
        return SentenceTransformerBackend(model_name=model)
    elif backend == "openai":
        from core.config.settings import settings
        return OpenAIEmbeddingBackend(
            model_name=settings.openai_embedding_model,
            api_key=settings.openai_api_key or None,
        )
    elif backend == "mock":
        return MockEmbeddingBackend()
    else:
        raise EmbeddingError(f"Unknown embedding backend: {backend}")
