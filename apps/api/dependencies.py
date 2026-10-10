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
from services.feedback_service import FeedbackService, RetrievalLedger
from services.handoff_service import HandoffService
from services.erasure_service import ErasureService, InMemoryReceiptStore
from domain.learning import OnlineWeightLearner
from domain.learning.trust_calibrator import TrustCalibrator
from domain.learning.risk_model import OnlineRiskModel
from services.guard_service import MemoryGuard
from services.outcome_service import OutcomeService

logger = logging.getLogger(__name__)

# Global singletons (initialized at startup)
_db_store: PostgresStore | None = None
_vector_store: InMemoryVectorStore | None = None
_graph_store: MockGraphStore | None = None

# Continual-learning state (process-local until the Postgres store lands; see ROADMAP P1).
_learner = OnlineWeightLearner()
_ledger = RetrievalLedger()
# Erasure receipts (process-local until persisted; see project_log.md).
_receipts = InMemoryReceiptStore()
# Outcome-calibrated source trust, shared across users (process-local until persisted).
_calibrator = TrustCalibrator()


def get_calibrator() -> TrustCalibrator:
    return _calibrator


# Self-improving write-risk model trained by reviewer decisions (process-local until persisted).
_risk_model = OnlineRiskModel()


def get_risk_model() -> OnlineRiskModel:
    return _risk_model


def get_learner() -> OnlineWeightLearner:
    return _learner


def get_ledger() -> RetrievalLedger:
    return _ledger


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
    cal: Annotated[TrustCalibrator, Depends(get_calibrator)],
    risk: Annotated[OnlineRiskModel, Depends(get_risk_model)],
) -> IngestionService:
    return IngestionService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        graph_store=graph,
        llm_client=get_llm(),
        trust_calibrator=cal,
        risk_model=risk,
    )


async def get_retrieve_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    cal: Annotated[TrustCalibrator, Depends(get_calibrator)],
) -> RetrievalService:
    return RetrievalService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        learner=get_learner(),
        ledger=get_ledger(),
        guard=MemoryGuard(db, trust_calibrator=cal),
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


async def get_feedback_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
    cal: Annotated[TrustCalibrator, Depends(get_calibrator)],
) -> FeedbackService:
    return FeedbackService(
        relational_store=db, ledger=get_ledger(), learner=get_learner(),
        trust_calibrator=cal,
    )


async def get_handoff_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
    retrieve: Annotated[RetrievalService, Depends(get_retrieve_service)],
) -> HandoffService:
    return HandoffService(relational_store=db, retrieval_service=retrieve)


async def get_erasure_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    graph: Annotated[MockGraphStore, Depends(get_graph_store)],
) -> ErasureService:
    return ErasureService(
        relational_store=db,
        vector_store=vs,
        embedding_backend=get_embedding(),
        graph_store=graph,
        receipt_store=_receipts,
    )


async def get_outcome_service(
    db: Annotated[PostgresStore, Depends(get_db_store)],
    vs: Annotated[InMemoryVectorStore, Depends(get_vector_store)],
    ingest: Annotated[IngestionService, Depends(get_ingest_service)],
    erasure: Annotated[ErasureService, Depends(get_erasure_service)],
    cal: Annotated[TrustCalibrator, Depends(get_calibrator)],
    risk: Annotated[OnlineRiskModel, Depends(get_risk_model)],
) -> OutcomeService:
    return OutcomeService(
        relational_store=db, vector_store=vs, ingest_service=ingest,
        trust_calibrator=cal, erasure_service=erasure, risk_model=risk,
    )
