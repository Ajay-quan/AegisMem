"""End-to-end tests for scoped API-key auth and the audit log."""
from __future__ import annotations

import pytest
from httpx import AsyncClient, ASGITransport

from apps.api.main import app
from apps.api.dependencies import (
    get_db_store, get_vector_store, get_graph_store, get_ingest_service,
)
from core.config.settings import settings
from core.security.keys import reset_key_registry
from core.security.audit import get_audit_log
from tests.fixtures.conftest import MockPostgresStore
from adapters.vector_store.qdrant_store import InMemoryVectorStore
from adapters.graph_store.neo4j_store import MockGraphStore
from adapters.embeddings.backend import MockEmbeddingBackend
from adapters.llm.mock_client import MockLLMClient
from services.ingest_service import IngestionService


@pytest.fixture
def scoped_keys():
    """Configure one named, tenanted key and start with a clean audit log."""
    prev = settings.api_keys
    settings.api_keys = "svc-a:secretA:tenantA"
    reset_key_registry()
    get_audit_log().clear()
    try:
        yield
    finally:
        settings.api_keys = prev
        reset_key_registry()
        get_audit_log().clear()


@pytest.fixture
async def client():
    # Mock the heavy data dependencies so the test isolates auth + audit
    # (which run as app middleware, independent of the data layer).
    db = MockPostgresStore()
    embed = MockEmbeddingBackend(dim=384)
    vs = InMemoryVectorStore()
    await vs.initialize(384)
    graph = MockGraphStore()
    ingest = IngestionService(db, vs, embed, graph, MockLLMClient())

    app.dependency_overrides[get_db_store] = lambda: db
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.dependency_overrides[get_graph_store] = lambda: graph
    app.dependency_overrides[get_ingest_service] = lambda: ingest

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


async def test_request_without_key_is_rejected(client, scoped_keys):
    r = await client.post("/api/v1/ingest", json={
        "text": "hello", "user_id": "alice", "memory_type": "fact"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "unauthorized"


async def test_valid_key_allows_and_is_audited(client, scoped_keys):
    headers = {"X-API-Key": "secretA"}
    r = await client.post("/api/v1/ingest", headers=headers, json={
        "text": "Alice prefers Python.", "user_id": "alice", "memory_type": "fact"})
    assert r.status_code == 201

    audit = (await client.get("/api/v1/audit", headers=headers)).json()
    assert audit["total"] >= 1
    top = audit["entries"][0]
    assert top["principal"] == "svc-a"
    assert top["tenant"] == "tenantA"
    assert top["method"] == "POST"
    assert top["path"] == "/api/v1/ingest"


async def test_wrong_key_rejected(client, scoped_keys):
    r = await client.get("/api/v1/audit", headers={"X-API-Key": "WRONG"})
    assert r.status_code == 401
