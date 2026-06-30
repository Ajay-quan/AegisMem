"""Flask lifecycle API tests for the single-node demo deployment."""
from __future__ import annotations

import importlib


def make_client(tmp_path, monkeypatch, api_key: str = ""):
    monkeypatch.setenv("STATEFUL_AI_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("STATEFUL_AI_EMBEDDING_BACKEND", "mock")
    monkeypatch.setenv("STATEFUL_AI_VECTOR_STORE", "faiss")
    if api_key:
        monkeypatch.setenv("STATEFUL_AI_API_KEY", api_key)
    else:
        monkeypatch.delenv("STATEFUL_AI_API_KEY", raising=False)
    module = importlib.import_module("apps.flask_app")
    return module.create_app().test_client()


def test_flask_memory_lifecycle(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)

    first = client.post(
        "/api/v1/memories",
        json={
            "user_id": "alice",
            "key": "python-pref",
            "content": "Alice prefers Python and FAISS for local vector search.",
            "importance_score": 0.9,
        },
    )
    assert first.status_code == 201
    first_id = first.get_json()["memory"]["memory_id"]

    second = client.post(
        "/api/v1/memories",
        json={
            "user_id": "alice",
            "key": "aws-pref",
            "content": "Alice wants AWS Free Tier deployments with no managed databases.",
            "related_memory_ids": [first_id],
        },
    )
    assert second.status_code == 201

    retrieve = client.post("/api/v1/retrieve", json={"user_id": "alice", "query": "local vector search"})
    assert retrieve.status_code == 200
    assert retrieve.get_json()["total_found"] >= 1

    detail = client.get(f"/api/v1/memories/{first_id}")
    assert detail.status_code == 200

    listing = client.get("/api/v1/memories?user_id=alice")
    assert listing.status_code == 200
    assert listing.get_json()["total"] == 2

    exact = client.get("/api/v1/memories/key/alice/python-pref")
    assert exact.status_code == 200
    assert exact.get_json()["lookup"] == "sha256_hash_index"

    graph = client.get(f"/api/v1/graph/{first_id}?depth=2")
    assert graph.status_code == 200
    assert graph.get_json()["related"]

    patch = client.patch(f"/api/v1/memories/{first_id}", json={"content": "Alice prefers Flask APIs."})
    assert patch.status_code == 200
    assert patch.get_json()["memory"]["version"] == 2

    versions = client.get(f"/api/v1/memories/{first_id}/versions")
    assert versions.status_code == 200
    assert versions.get_json()["total"] == 2

    exported = client.get("/api/v1/export")
    assert exported.status_code == 200
    assert len(exported.get_json()["records"]) == 2

    delete = client.delete(f"/api/v1/memories/{first_id}")
    assert delete.status_code == 200


def test_import_export_replace(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    payload = {
        "replace": True,
        "records": [
            {
                "memory_id": "fixed-id",
                "user_id": "alice",
                "content": "Imported memory about Chroma and FAISS.",
                "key": "imported",
                "namespace": "user:alice",
                "metadata": {},
                "tags": ["import"],
                "importance_score": 0.7,
                "access_count": 0,
                "version": 1,
                "status": "active",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:00:00+00:00",
            }
        ],
    }
    imported = client.post("/api/v1/import", json=payload)
    assert imported.status_code == 200
    assert imported.get_json()["imported"] == 1
    exact = client.get("/api/v1/memories/key/alice/imported")
    assert exact.status_code == 200
    assert exact.get_json()["memory"]["memory_id"] == "fixed-id"


def test_validation_errors_are_structured(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch)
    response = client.post("/api/v1/retrieve", json={"user_id": "alice", "query": "x", "top_k": 1000})
    assert response.status_code == 400
    assert response.get_json()["error"]["code"] == "bad_request"


def test_api_key_auth(tmp_path, monkeypatch):
    client = make_client(tmp_path, monkeypatch, api_key="secret")
    unauthorized = client.post("/api/v1/retrieve", json={"user_id": "alice", "query": "x"})
    assert unauthorized.status_code == 401
    authorized = client.post(
        "/api/v1/retrieve",
        json={"user_id": "alice", "query": "x"},
        headers={"X-API-Key": "secret"},
    )
    assert authorized.status_code == 200
