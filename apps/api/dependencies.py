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
from adapters.relational_store.postgres_store import PostgresStore
from services.ingest_service import IngestionService
from services.retrieve_service import RetrievalService
from services.update_service import UpdateService
from services.reflect_service import ReflectionService
from services.contradiction_service import ContradictionService

logger = logging.getLogger(__name__)

# Global singletons (initialized at startup)
_db_store: PostgresStore | None = None
_vector_store: InMemoryVectorStore | None = None
_graph_store: MockGraphStore | None = None


async def get_db_store() -> PostgresStore:
    global _db_store
    if _db_store is None:
        _db_store = PostgresStore(settings.postgres_url)
        await _db_store.initialize()
    return _db_store


async def get_vector_store() -> InMemoryVectorStore:
    global _vector_store
    if _vector_store is None:
        # Try Qdrant first, fall back to in-memory
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
        except Exception as e:
            logger.warning(f"Qdrant unavailable ({e}), using in-memory vector store")
            store = InMemoryVectorStore()
            embed_backend = get_embedding_backend("mock")
            await store.initialize(embed_backend.dimension)
            _vector_store = store
    return _vector_store


async def get_graph_store() -> MockGraphStore:
    global _graph_store
    if _graph_store is None:
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
        except Exception as e:
            logger.warning(f"Neo4j unavailable ({e}), using mock graph store")
            mock = MockGraphStore()
            await mock.connect()
            _graph_store = mock
    return _graph_store


def get_embedding() -> EmbeddingBackend:
    backend = settings.embedding_backend
    # In production: use sentence_transformers. In test/dev: use mock
    if not settings.openai_api_key and not settings.anthropic_api_key:
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
    db: Annotated[PostgresStore, Depends(get_db_store)],
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
    db: Annotated[PostgresStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
) -> RetrievalService:
    return RetrievalService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
    )


async def get_update_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
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
    db: Annotated[PostgresStore, Depends(get_db_store)],
    ingest: Annotated[IngestionService, Depends(get_ingest_service)],
) -> ReflectionService:
    return ReflectionService(
        relational_store=db,
        ingest_service=ingest,
        llm_client=get_llm(),
    )


async def get_contradiction_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
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
