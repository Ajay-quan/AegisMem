"""End-to-end API tests for the Stateful-CL feedback loop."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from apps.api.main import app
from apps.api.dependencies import (
    get_db_store, get_vector_store, get_graph_store,
    get_ingest_service, get_retrieve_service, get_feedback_service,
)
from tests.fixtures.conftest import MockPostgresStore
from adapters.vector_store.qdrant_store import InMemoryVectorStore
from adapters.graph_store.neo4j_store import MockGraphStore
from adapters.embeddings.backend import MockEmbeddingBackend
from adapters.llm.mock_client import MockLLMClient
from services.ingest_service import IngestionService
from services.retrieve_service import RetrievalService
from services.feedback_service import FeedbackService
from core.config.settings import settings
from domain.learning.registry import reset_learning_state


@pytest.fixture
def cl_enabled():
    """Turn on continual learning with a clean learning state for the test."""
    prev = settings.continual_learning_enabled
    settings.continual_learning_enabled = True
    reset_learning_state()
    try:
        yield
    finally:
        settings.continual_learning_enabled = prev
        reset_learning_state()


@pytest.fixture
async def client():
    db = MockPostgresStore()
    embed = MockEmbeddingBackend(dim=384)
    vs = InMemoryVectorStore()
    await vs.initialize(384)
    graph = MockGraphStore()
    llm = MockLLMClient()

    ingest = IngestionService(db, vs, embed, graph, llm)
    retrieve = RetrievalService(db, vs, embed)
    feedback = FeedbackService(db)

    app.dependency_overrides[get_db_store] = lambda: db
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.dependency_overrides[get_graph_store] = lambda: graph
    app.dependency_overrides[get_ingest_service] = lambda: ingest
    app.dependency_overrides[get_retrieve_service] = lambda: retrieve
    app.dependency_overrides[get_feedback_service] = lambda: feedback

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def _ingest_and_retrieve(client):
    await client.post("/api/v1/ingest", json={
        "text": "Alice prefers Python and FAISS for local vector search.",
        "user_id": "alice", "memory_type": "fact",
    })
    resp = await client.post("/api/v1/retrieve", json={
        "query": "local vector search", "user_id": "alice", "top_k": 3,
    })
    return resp.json()


async def test_retrieve_returns_query_id_when_enabled(client, cl_enabled):
    body = await _ingest_and_retrieve(client)
    assert body["query_id"]  # non-empty
    assert body["results"], "expected at least one served memory"


async def test_feedback_updates_policy(client, cl_enabled):
    body = await _ingest_and_retrieve(client)
    query_id = body["query_id"]
    memory_id = body["results"][0]["memory_id"]

    resp = await client.post("/api/v1/feedback", json={
        "query_id": query_id, "memory_id": memory_id, "useful": True, "outcome": "success",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["recorded"] is True
    assert data["reward"] >= 0.5
    assert data["policy_updates"] >= 1
    assert set(data["weights"]) == {"semantic", "lexical", "recency", "importance", "access"}

    # learning stats endpoint reflects activity
    stats = (await client.get("/api/v1/learning/stats")).json()
    assert stats["enabled"] is True
    assert stats["replay"]["labeled"] >= 1


async def test_feedback_unknown_query_is_safe(client, cl_enabled):
    resp = await client.post("/api/v1/feedback", json={
        "query_id": "does-not-exist", "memory_id": "m", "useful": True,
    })
    assert resp.status_code == 200
    assert resp.json()["recorded"] is False


async def test_feedback_noop_when_disabled(client):
    # cl_enabled fixture NOT used => continual learning is off
    body = await _ingest_and_retrieve(client)
    assert body["query_id"] == ""  # no correlation id when disabled
    resp = await client.post("/api/v1/feedback", json={
        "query_id": "x", "memory_id": "y", "useful": True,
    })
    assert resp.status_code == 200
    assert resp.json()["recorded"] is False
