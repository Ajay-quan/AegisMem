"""Dependency injection for FastAPI - wires up all services."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Annotated

from fastapi import Depends

from core.config import settings
from adapters.embeddings.backend import get_embedding_backend, EmbeddingBackend
from adapters.llm.factory import create_llm_client
from adapters.llm.base import LLMClient
from adapters.vector_store.qdrant_store import InMemoryVectorStore, QdrantStore
from adapters.graph_store.neo4j_store import MockGraphStore, GraphStore
from adapters.relational_store.memory_store import InMemoryRelationalStore
# PostgresStore (and its SQLAlchemy dependency) is imported lazily so the
# zero-infra in-memory default has no hard dependency on a SQL stack.
from services.ingest_service import IngestionService
from services.retrieve_service import RetrievalService
from services.update_service import UpdateService
from services.reflect_service import ReflectionService
from services.contradiction_service import ContradictionService
from services.feedback_service import FeedbackService

logger = logging.getLogger(__name__)

# Global singletons (initialized at startup)
_db_store: InMemoryRelationalStore | None = None
_vector_store: InMemoryVectorStore | None = None
_graph_store: MockGraphStore | None = None


def _fallback_or_raise(component: str, exc: Exception) -> None:
    """Decide whether an unreachable configured store should fail or fall back.

    In production a configured external store that is unavailable is a fatal
    misconfiguration — we must not silently downgrade to a non-durable
    in-memory store and pretend everything is fine. In development/staging we
    keep the friendly zero-infra fallback.
    """
    if settings.strict_stores:
        raise RuntimeError(
            f"{component} is configured but unavailable in production "
            f"(app_env=production): {exc}. Refusing to fall back to the "
            f"in-memory store. Fix the connection or change the *_STORE setting."
        ) from exc
    logger.warning(
        f"{component} unavailable ({exc}); falling back to in-memory "
        f"(app_env={settings.app_env}). This is non-durable — do not use in production."
    )


async def get_db_store() -> InMemoryRelationalStore:
    """Return the relational store.

    Defaults to the zero-infra in-memory store. Set ``RELATIONAL_STORE=postgres``
    for the production store. If Postgres is selected but unreachable: in
    production we raise (no silent data-loss fallback); otherwise we fall back
    to in-memory with a warning.
    """
    global _db_store
    if _db_store is None:
        if settings.relational_store == "postgres":
            try:
                from adapters.relational_store.postgres_store import PostgresStore
                store = PostgresStore(settings.postgres_url)
                await store.initialize()
                _db_store = store
                logger.info("Using PostgreSQL relational store")
            except Exception as e:
                _fallback_or_raise("PostgreSQL relational store", e)
                store = InMemoryRelationalStore(settings.data_dir)
                await store.initialize()
                _db_store = store
        else:
            store = InMemoryRelationalStore(settings.data_dir)
            await store.initialize()
            _db_store = store
    return _db_store


async def get_vector_store() -> InMemoryVectorStore:
    """Return the vector store, gated by ``VECTOR_STORE`` (memory|qdrant).

    Only attempts Qdrant when explicitly selected, so the zero-infra default
    never probes localhost:6333. Production raises on an unreachable Qdrant.
    """
    global _vector_store
    if _vector_store is None:
        if settings.vector_store == "qdrant":
            try:
                qdrant = QdrantStore(
                    host=settings.qdrant_host,
                    port=settings.qdrant_port,
                    collection_name=settings.qdrant_collection,
                )
                embed_backend = get_embedding_backend(
                    settings.embedding_backend, settings.embedding_model
                )
                await qdrant.initialize(embed_backend.dimension)
                _vector_store = qdrant  # type: ignore
                logger.info("Using Qdrant vector store")
                return _vector_store
            except Exception as e:
                _fallback_or_raise("Qdrant vector store", e)
        store = InMemoryVectorStore()
        embed_backend = get_embedding()
        await store.initialize(embed_backend.dimension)
        _vector_store = store
    return _vector_store


async def get_graph_store() -> MockGraphStore:
    """Return the graph store, gated by ``GRAPH_STORE`` (memory|neo4j).

    Only attempts Neo4j when explicitly selected. Production raises on an
    unreachable Neo4j.
    """
    global _graph_store
    if _graph_store is None:
        if settings.graph_store == "neo4j":
            try:
                from adapters.graph_store.neo4j_store import GraphStore as Neo4jStore
                store = Neo4jStore(
                    uri=settings.neo4j_uri,
                    user=settings.neo4j_user,
                    password=settings.neo4j_password,
                )
                await store.connect()
                _graph_store = store  # type: ignore
                logger.info("Using Neo4j graph store")
                return _graph_store
            except Exception as e:
                _fallback_or_raise("Neo4j graph store", e)
        mock = MockGraphStore()
        await mock.connect()
        _graph_store = mock
    return _graph_store


def get_embedding() -> EmbeddingBackend:
    """Resolve the configured embedding backend.

    Only API-backed backends require a key; local ones (sentence_transformers)
    and the deterministic ``mock`` backend do not. We therefore fall back to
    ``mock`` solely when an API-backed backend is selected without its key, so
    setting ``EMBEDDING_BACKEND=sentence_transformers`` actually runs local
    embeddings as the docs claim.
    """
    backend = settings.embedding_backend
    if backend == "openai" and not settings.openai_api_key:
        if settings.strict_stores:
            raise RuntimeError(
                "EMBEDDING_BACKEND=openai requires OPENAI_API_KEY in production."
            )
        logger.warning(
            "Embedding backend 'openai' needs OPENAI_API_KEY but none is set; "
            "falling back to 'mock'."
        )
        backend = "mock"
    return get_embedding_backend(backend, settings.embedding_model)


def get_llm() -> LLMClient:
    try:
        return create_llm_client()
    except Exception as e:
        logger.warning(f"LLM client unavailable ({e}), using mock")
        from adapters.llm.mock_client import MockLLMClient
        return MockLLMClient()


async def get_ingest_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    graph: Annotated[MockGraphStore, Depends(get_graph_store)],
) -> IngestionService:
    return IngestionService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        graph_store=graph,
        llm_client=get_llm(),
    )


async def get_retrieve_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
) -> RetrievalService:
    return RetrievalService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
    )


async def get_update_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    ingest: Annotated[IngestionService, Depends(get_ingest_service)],
) -> UpdateService:
    return UpdateService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        llm_client=get_llm(),
        ingest_service=ingest,
    )


async def get_reflect_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
    ingest: Annotated[IngestionService, Depends(get_ingest_service)],
) -> ReflectionService:
    return ReflectionService(
        relational_store=db,
        ingest_service=ingest,
        llm_client=get_llm(),
    )


async def get_contradiction_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    graph: Annotated[MockGraphStore, Depends(get_graph_store)],
) -> ContradictionService:
    return ContradictionService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        llm_client=get_llm(),
        graph_store=graph,
    )


async def get_feedback_service(
    db: Annotated[InMemoryRelationalStore, Depends(get_db_store)],
) -> FeedbackService:
    return FeedbackService(relational_store=db)
