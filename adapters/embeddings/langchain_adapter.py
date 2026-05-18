"""LangChain integration for AegisMem embeddings."""
from __future__ import annotations

import asyncio
from typing import Any

try:
    from langchain_core.embeddings import Embeddings
except Exception:
    class Embeddings:  # type: ignore[no-redef]
        """Minimal fallback so local tests run before requirements are installed."""

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise NotImplementedError

        def embed_query(self, text: str) -> list[float]:
            raise NotImplementedError


class AegisMemEmbeddings(Embeddings):
    """Expose any AegisMem embedding backend through LangChain Embeddings."""

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._run(self.backend.embed(texts))

    def embed_query(self, text: str) -> list[float]:
        return self._run(self.backend.embed_single(text))

    def _run(self, coroutine: Any) -> Any:
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        if loop.is_running():
            raise RuntimeError("AegisMemEmbeddings sync API cannot run inside an active event loop")
        return loop.run_until_complete(coroutine)
