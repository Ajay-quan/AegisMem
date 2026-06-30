"""Tests for API-key auth, rate limiting, security headers, and stats."""
import pytest
from httpx import AsyncClient, ASGITransport

from apps.api.main import app
from apps.api import security
from apps.api.dependencies import get_db_store, get_vector_store, get_graph_store
from core.config import settings
from tests.fixtures.conftest import MockPostgresStore
from adapters.vector_store.qdrant_store import InMemoryVectorStore
from adapters.graph_store.neo4j_store import MockGraphStore


@pytest.fixture
async def client():
    db = MockPostgresStore()
    vs = InMemoryVectorStore()
    await vs.initialize(384)
    graph = MockGraphStore()
    app.dependency_overrides[get_db_store] = lambda: db
    app.dependency_overrides[get_vector_store] = lambda: vs
    app.dependency_overrides[get_graph_store] = lambda: graph
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


class TestSecurityHeaders:
    async def test_headers_present(self, client):
        resp = await client.get("/health")
        assert resp.headers["X-Content-Type-Options"] == "nosniff"
        assert resp.headers["X-Frame-Options"] == "DENY"
        assert resp.headers["Referrer-Policy"] == "no-referrer"

    async def test_api_routes_no_store(self, client):
        resp = await client.get("/api/v1/stats", params={"user_id": "u1"})
        assert resp.headers.get("Cache-Control") == "no-store"


class TestApiKeyAuth:
    async def test_open_when_no_key_configured(self, client):
        resp = await client.get("/api/v1/stats", params={"user_id": "u1"})
        assert resp.status_code == 200

    async def test_rejects_missing_key(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "s3cret")
        resp = await client.get("/api/v1/stats", params={"user_id": "u1"})
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "unauthorized"

    async def test_rejects_wrong_key(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "s3cret")
        resp = await client.get(
            "/api/v1/stats", params={"user_id": "u1"}, headers={"X-API-Key": "nope"}
        )
        assert resp.status_code == 401

    async def test_accepts_valid_key(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "s3cret")
        resp = await client.get(
            "/api/v1/stats", params={"user_id": "u1"}, headers={"X-API-Key": "s3cret"}
        )
        assert resp.status_code == 200

    async def test_health_stays_public(self, client, monkeypatch):
        monkeypatch.setattr(settings, "api_key", "s3cret")
        resp = await client.get("/health")
        assert resp.status_code == 200


class TestRateLimiting:
    async def test_burst_then_429(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_enabled", True)
        monkeypatch.setattr(security, "limiter", security.TokenBucketLimiter(60, burst=2))
        ok1 = await client.get("/api/v1/stats", params={"user_id": "u1"})
        ok2 = await client.get("/api/v1/stats", params={"user_id": "u1"})
        blocked = await client.get("/api/v1/stats", params={"user_id": "u1"})
        assert ok1.status_code == 200 and ok2.status_code == 200
        assert blocked.status_code == 429
        assert blocked.json()["error"]["code"] == "rate_limited"
        assert "Retry-After" in blocked.headers

    async def test_public_paths_not_limited(self, client, monkeypatch):
        monkeypatch.setattr(settings, "rate_limit_enabled", True)
        monkeypatch.setattr(security, "limiter", security.TokenBucketLimiter(60, burst=1))
        for _ in range(5):
            resp = await client.get("/health")
            assert resp.status_code == 200


class TestBodySizeLimit:
    async def test_oversized_body_rejected(self, client, monkeypatch):
        monkeypatch.setattr(settings, "max_request_bytes", 100)
        resp = await client.post(
            "/api/v1/ingest",
            json={"user_id": "u1", "text": "x" * 500},
        )
        assert resp.status_code == 413
        assert resp.json()["error"]["code"] == "payload_too_large"


class TestStats:
    async def test_stats_aggregates(self, client):
        for text, mtype in [
            ("Alice prefers FAISS", "fact"),
            ("Alice works at Acme", "observation"),
            ("Alice asked about Qdrant", "observation"),
        ]:
            resp = await client.post(
                "/api/v1/ingest",
                json={"user_id": "stats_user", "text": text, "memory_type": mtype},
            )
            assert resp.status_code == 201

        resp = await client.get("/api/v1/stats", params={"user_id": "stats_user"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_memories"] == 3
        assert body["by_type"].get("observation") == 2
        assert body["by_type"].get("fact") == 1
        assert 0.0 <= body["avg_importance"] <= 1.0
        assert body["last_memory_at"]
