"""Flask REST API for the single-node stateful.ai demonstration deployment."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, request

from apps.landing_page import DEMO_HTML, LANDING_HTML
from services.flask_memory_service import FlaskMemoryService


class ApiError(ValueError):
    """Request validation error with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_app() -> Flask:
    """Create the Flask app with lifecycle memory routes."""
    app = Flask(__name__)
    # Default to a project-local path so the app boots anywhere; the Docker
    # image overrides this with STATEFUL_AI_DATA_DIR=/data/stateful_ai.
    data_dir = os.getenv("STATEFUL_AI_DATA_DIR", "./data/stateful_ai")
    embedding_backend = os.getenv("STATEFUL_AI_EMBEDDING_BACKEND", "mock")
    vector_backend = os.getenv("STATEFUL_AI_VECTOR_STORE", "faiss")
    api_key = os.getenv("STATEFUL_AI_API_KEY", "")
    service = FlaskMemoryService(
        data_dir=data_dir,
        embedding_backend=embedding_backend,
        vector_backend=vector_backend,
    )
    app.config["memory_service"] = service

    @app.after_request
    def security_headers(response):
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        if request.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response

    @app.before_request
    def require_api_key():
        if not api_key or request.path in {"/", "/demo", "/health"}:
            return None
        if request.headers.get("X-API-Key") != api_key:
            return error_response("invalid or missing API key", 401, code="unauthorized")
        return None

    @app.errorhandler(ApiError)
    def handle_api_error(exc: ApiError):
        return error_response(str(exc), exc.status_code)

    @app.errorhandler(404)
    def handle_not_found(_: Any):
        return error_response("route not found", 404, code="not_found")

    @app.errorhandler(Exception)
    def handle_unexpected(exc: Exception):
        return error_response(str(exc), 500, code="internal_error")

    @app.get("/")
    def index():
        return LANDING_HTML

    @app.get("/demo")
    def demo():
        return DEMO_HTML

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "stateful_ai-flask",
                "data_dir": data_dir,
                "vector_store": vector_backend,
                "auth_enabled": bool(api_key),
            }
        )

    @app.post("/api/v1/memories")
    def ingest_memory():
        payload = json_body()
        content = non_empty(payload.get("content") or payload.get("text"), "content/text")
        user_id = non_empty(payload.get("user_id"), "user_id")
        memory = service.ingest(
            content=content,
            user_id=user_id,
            key=optional_string(payload.get("key")),
            related_memory_ids=string_list(payload.get("related_memory_ids", []), "related_memory_ids"),
            metadata=object_value(payload.get("metadata", {}), "metadata"),
            tags=string_list(payload.get("tags", []), "tags"),
            importance_score=score_value(payload.get("importance_score", 0.5), "importance_score"),
        )
        return jsonify({"memory": asdict(memory)}), 201

    @app.post("/api/v1/ingest")
    def ingest_alias():
        return ingest_memory()

    @app.post("/api/v1/retrieve")
    def retrieve_memories():
        payload = json_body()
        query = non_empty(payload.get("query"), "query")
        user_id = non_empty(payload.get("user_id"), "user_id")
        top_k = int_range(payload.get("top_k", 5), "top_k", 1, 50)
        results = service.retrieve(query=query, user_id=user_id, top_k=top_k)
        return jsonify({"query": query, "results": results, "total_found": len(results)})

    @app.get("/api/v1/memories/<memory_id>")
    def get_memory(memory_id: str):
        memory = service.store.get(memory_id)
        if not memory:
            return error_response("memory not found", 404, code="memory_not_found")
        return jsonify({"memory": asdict(memory)})

    @app.get("/api/v1/memories")
    def list_memories():
        user_id = non_empty(request.args.get("user_id"), "user_id query parameter")
        include_deleted = request.args.get("include_deleted", "false").lower() == "true"
        memories = [asdict(memory) for memory in service.store.list(user_id=user_id, include_deleted=include_deleted)]
        return jsonify({"memories": memories, "total": len(memories)})

    @app.get("/api/v1/memories/key/<user_id>/<path:key>")
    def get_by_hash_key(user_id: str, key: str):
        memory = service.get_by_key(user_id=user_id, key=key)
        if not memory:
            return error_response("memory not found", 404, code="memory_not_found")
        return jsonify({"memory": asdict(memory), "lookup": "sha256_hash_index"})

    @app.get("/api/v1/memories/<memory_id>/versions")
    def memory_versions(memory_id: str):
        versions = service.versions(memory_id)
        if not versions:
            return error_response("memory not found", 404, code="memory_not_found")
        return jsonify({"memory_id": memory_id, "versions": versions, "total": len(versions)})

    @app.patch("/api/v1/memories/<memory_id>")
    def update_memory(memory_id: str):
        payload = json_body()
        allowed: dict[str, Any] = {}
        if "content" in payload:
            allowed["content"] = non_empty(payload["content"], "content")
        if "key" in payload:
            allowed["key"] = non_empty(payload["key"], "key")
        if "metadata" in payload:
            allowed["metadata"] = object_value(payload["metadata"], "metadata")
        if "tags" in payload:
            allowed["tags"] = string_list(payload["tags"], "tags")
        if "importance_score" in payload:
            allowed["importance_score"] = score_value(payload["importance_score"], "importance_score")
        if not allowed:
            raise ApiError("at least one updatable field is required")
        memory = service.update(memory_id, **allowed)
        if not memory:
            return error_response("memory not found", 404, code="memory_not_found")
        return jsonify({"memory": asdict(memory)})

    @app.delete("/api/v1/memories/<memory_id>")
    def delete_memory(memory_id: str):
        if not service.delete(memory_id):
            return error_response("memory not found", 404, code="memory_not_found")
        return jsonify({"deleted": True, "memory_id": memory_id})

    @app.get("/api/v1/graph/<memory_id>")
    def graph_traversal(memory_id: str):
        depth = int_range(request.args.get("depth", 2), "depth", 1, 5)
        return jsonify({"memory_id": memory_id, "related": service.traverse(memory_id, depth=depth)})

    @app.get("/api/v1/stats")
    def memory_stats():
        memories = service.store.list_all(include_deleted=True)
        active = [m for m in memories if m.status == "active"]
        users = {m.user_id for m in memories}
        tags: dict[str, int] = {}
        for m in active:
            for tag in m.tags:
                tags[tag] = tags.get(tag, 0) + 1
        top_tags = sorted(tags.items(), key=lambda kv: -kv[1])[:8]
        return jsonify(
            {
                "total_memories": len(memories),
                "active_memories": len(active),
                "deleted_memories": len(memories) - len(active),
                "users": len(users),
                "total_versions": sum(m.version for m in memories),
                "total_access_count": sum(m.access_count for m in active),
                "avg_importance": round(
                    sum(m.importance_score for m in active) / len(active), 4
                ) if active else 0.0,
                "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
                "last_updated": max((m.updated_at for m in memories), default=""),
            }
        )

    @app.get("/api/v1/export")
    def export_memories():
        include_deleted = request.args.get("include_deleted", "true").lower() != "false"
        return jsonify(service.export_payload(include_deleted=include_deleted))

    @app.post("/api/v1/import")
    def import_memories():
        payload = json_body()
        replace = bool(payload.get("replace", False))
        imported = service.import_payload(payload, replace=replace)
        return jsonify({"imported": imported, "replace": replace})

    return app


def json_body() -> dict[str, Any]:
    payload = request.get_json(force=True, silent=True)
    if not isinstance(payload, dict):
        raise ApiError("request body must be a JSON object")
    return payload


def non_empty(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ApiError(f"{field} is required")
    return value.strip()


def optional_string(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ApiError("key must be a string")
    return value.strip() or None


def object_value(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ApiError(f"{field} must be an object")
    return value


def string_list(value: Any, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ApiError(f"{field} must be a list of strings")
    return value


def score_value(value: Any, field: str) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{field} must be a number") from exc
    if score < 0.0 or score > 1.0:
        raise ApiError(f"{field} must be between 0 and 1")
    return score


def int_range(value: Any, field: str, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ApiError(f"{field} must be an integer") from exc
    if parsed < minimum or parsed > maximum:
        raise ApiError(f"{field} must be between {minimum} and {maximum}")
    return parsed


def error_response(message: str, status_code: int, code: str = "bad_request"):
    return jsonify({"error": {"code": code, "message": message}}), status_code


app = create_app()
