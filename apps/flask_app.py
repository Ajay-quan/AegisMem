"""Flask REST API for the single-node AegisMem demonstration deployment."""
from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any

from flask import Flask, jsonify, request

from services.flask_memory_service import FlaskMemoryService


INDEX_HTML = """
<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>AegisMem Demo</title>
  <style>
    body { font-family: system-ui, -apple-system, BlinkMacSystemFont, sans-serif; margin: 0; color: #172033; background: #f7f8fb; }
    main { max-width: 1080px; margin: 0 auto; padding: 32px 20px; }
    h1 { margin: 0 0 6px; font-size: 32px; }
    h2 { font-size: 18px; margin-top: 0; }
    .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(310px, 1fr)); gap: 16px; }
    section { background: white; border: 1px solid #d9deea; border-radius: 8px; padding: 16px; }
    label { display: block; font-size: 13px; font-weight: 650; margin: 10px 0 4px; }
    input, textarea { width: 100%; box-sizing: border-box; border: 1px solid #c7cfdd; border-radius: 6px; padding: 9px; font: inherit; }
    textarea { min-height: 82px; resize: vertical; }
    button { margin-top: 12px; border: 0; border-radius: 6px; background: #244fd6; color: white; padding: 10px 13px; font-weight: 700; cursor: pointer; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; background: #101827; color: #e7edf8; padding: 14px; border-radius: 8px; min-height: 160px; }
    .hint { color: #596579; margin: 0 0 22px; }
  </style>
</head>
<body>
<main>
  <h1>AegisMem</h1>
  <p class="hint">Local memory lifecycle demo: ingest, retrieve, exact key lookup, graph traversal, import, and export.</p>
  <div class="grid">
    <section>
      <h2>Ingest</h2>
      <label>User ID</label><input id="ingest-user" value="alice" />
      <label>Key</label><input id="ingest-key" value="python-pref" />
      <label>Content</label><textarea id="ingest-content">Alice prefers Python and FAISS for local vector search.</textarea>
      <button onclick="ingest()">Ingest</button>
    </section>
    <section>
      <h2>Retrieve</h2>
      <label>User ID</label><input id="retrieve-user" value="alice" />
      <label>Query</label><input id="retrieve-query" value="FAISS vector search" />
      <button onclick="retrieveMemories()">Retrieve</button>
    </section>
    <section>
      <h2>Exact Lookup</h2>
      <label>User ID</label><input id="lookup-user" value="alice" />
      <label>Key</label><input id="lookup-key" value="python-pref" />
      <button onclick="lookupKey()">Lookup</button>
    </section>
    <section>
      <h2>Graph</h2>
      <label>Memory ID</label><input id="graph-id" placeholder="Paste memory_id" />
      <button onclick="graph()">Traverse</button>
    </section>
    <section>
      <h2>Export / Import</h2>
      <button onclick="exportData()">Export</button>
      <label>Import JSON</label><textarea id="import-json" placeholder='{"records": [...]}'></textarea>
      <button onclick="importData()">Import</button>
    </section>
    <section>
      <h2>Output</h2>
      <pre id="output">Ready.</pre>
    </section>
  </div>
</main>
<script>
const apiKey = localStorage.getItem('aegismem_api_key') || '';
function headers() { const h = {'Content-Type': 'application/json'}; if (apiKey) h['X-API-Key'] = apiKey; return h; }
function show(data) { document.getElementById('output').textContent = JSON.stringify(data, null, 2); }
async function ingest() {
  const body = {user_id: val('ingest-user'), key: val('ingest-key'), content: val('ingest-content')};
  const res = await fetch('/api/v1/memories', {method:'POST', headers: headers(), body: JSON.stringify(body)});
  const data = await res.json(); show(data); if (data.memory) document.getElementById('graph-id').value = data.memory.memory_id;
}
async function retrieveMemories() {
  const body = {user_id: val('retrieve-user'), query: val('retrieve-query'), top_k: 5};
  show(await (await fetch('/api/v1/retrieve', {method:'POST', headers: headers(), body: JSON.stringify(body)})).json());
}
async function lookupKey() { show(await (await fetch(`/api/v1/memories/key/${encodeURIComponent(val('lookup-user'))}/${encodeURIComponent(val('lookup-key'))}`, {headers: headers()})).json()); }
async function graph() { show(await (await fetch(`/api/v1/graph/${encodeURIComponent(val('graph-id'))}?depth=2`, {headers: headers()})).json()); }
async function exportData() { const data = await (await fetch('/api/v1/export', {headers: headers()})).json(); document.getElementById('import-json').value = JSON.stringify(data, null, 2); show(data); }
async function importData() { show(await (await fetch('/api/v1/import', {method:'POST', headers: headers(), body: val('import-json')})).json()); }
function val(id) { return document.getElementById(id).value; }
</script>
</body>
</html>
"""


class ApiError(ValueError):
    """Request validation error with an HTTP status code."""

    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def create_app() -> Flask:
    """Create the Flask app with lifecycle memory routes."""
    app = Flask(__name__)
    data_dir = os.getenv("AEGISMEM_DATA_DIR", "/data/aegismem")
    embedding_backend = os.getenv("AEGISMEM_EMBEDDING_BACKEND", "mock")
    vector_backend = os.getenv("AEGISMEM_VECTOR_STORE", "faiss")
    api_key = os.getenv("AEGISMEM_API_KEY", "")
    service = FlaskMemoryService(
        data_dir=data_dir,
        embedding_backend=embedding_backend,
        vector_backend=vector_backend,
    )
    app.config["memory_service"] = service

    @app.before_request
    def require_api_key():
        if not api_key or request.path in {"/", "/health"}:
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
        return INDEX_HTML

    @app.get("/health")
    def health():
        return jsonify(
            {
                "status": "ok",
                "service": "aegismem-flask",
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
