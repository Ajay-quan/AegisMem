"""Dependency-free HTTP client for the stateful.ai FastAPI service.

Covers the full memory lifecycle plus the Stateful-CL feedback loop. Implemented on
``urllib`` so the SDK has no third-party dependencies; a single private
``_request`` method centralizes transport, which also makes the client trivial
to unit-test by patching that one method.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class StatefulError(RuntimeError):
    """Raised when the API returns a non-2xx response."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"stateful.ai API error {status}: {message}")
        self.status = status
        self.message = message


class StatefulClient:
    """Thin synchronous client for the stateful.ai REST API."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    # ------------------------------------------------------------ transport
    def _request(self, method: str, path: str, payload: dict | None = None,
                 params: dict | None = None) -> Any:
        url = f"{self.base_url}{path}"
        if params:
            from urllib.parse import urlencode
            url = f"{url}?{urlencode({k: v for k, v in params.items() if v is not None})}"
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode()
                return json.loads(body) if body else {}
        except urllib.error.HTTPError as e:  # pragma: no cover - network path
            detail = e.read().decode(errors="replace")
            raise StatefulError(e.code, detail) from e
        except urllib.error.URLError as e:  # pragma: no cover - network path
            raise StatefulError(0, f"could not reach {self.base_url} ({e.reason})") from e

    # --------------------------------------------------------------- memory
    def ingest(self, text: str, user_id: str, memory_type: str = "observation",
               metadata: dict | None = None, **kw) -> dict:
        return self._request("POST", "/api/v1/ingest", {
            "text": text, "user_id": user_id, "memory_type": memory_type,
            "metadata": metadata or {}, **kw,
        })

    def retrieve(self, query: str, user_id: str, top_k: int = 5, **kw) -> dict:
        return self._request("POST", "/api/v1/retrieve", {
            "query": query, "user_id": user_id, "top_k": top_k, **kw,
        })

    def update(self, user_id: str, new_content: str, namespace: str = "") -> dict:
        return self._request("POST", "/api/v1/update", {
            "user_id": user_id, "new_content": new_content, "namespace": namespace,
        })

    def get_memory(self, memory_id: str) -> dict:
        return self._request("GET", f"/api/v1/memories/{memory_id}")

    def list_memories(self, user_id: str, limit: int = 20, **kw) -> list:
        return self._request("POST", "/api/v1/memories/list", {
            "user_id": user_id, "limit": limit, **kw,
        })

    def delete(self, memory_id: str, user_id: str) -> Any:
        return self._request("DELETE", f"/api/v1/memories/{memory_id}",
                             params={"user_id": user_id})

    # ----------------------------------------------------- continual learning
    def feedback(self, query_id: str, memory_id: str, useful: bool | None = None,
                 score: float | None = None, outcome: str = "") -> dict:
        return self._request("POST", "/api/v1/feedback", {
            "query_id": query_id, "memory_id": memory_id,
            "useful": useful, "score": score, "outcome": outcome,
        })

    def learning_stats(self) -> dict:
        return self._request("GET", "/api/v1/learning/stats")

    # --------------------------------------------------------------- system
    def stats(self, user_id: str, namespace: str = "") -> dict:
        return self._request("GET", "/api/v1/stats",
                             params={"user_id": user_id, "namespace": namespace})

    def health(self) -> dict:
        return self._request("GET", "/health")

    # ---------------------------------------------------------- convenience
    def context_for(self, query: str, user_id: str, top_k: int = 5) -> str:
        """Return just the ready-to-inject context window string for a query."""
        return self.retrieve(query, user_id, top_k=top_k).get("context_window", "")
