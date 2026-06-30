"""Unit tests for the stateful.ai SDK and CLI (no live server)."""
from __future__ import annotations

from sdk import StatefulClient


class RecordingClient(StatefulClient):
    """Captures the last _request call instead of hitting the network."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.calls: list[tuple] = []

    def _request(self, method, path, payload=None, params=None):
        self.calls.append((method, path, payload, params))
        # Return plausible shapes so chained helpers work.
        if path.endswith("/retrieve"):
            return {"query_id": "qid", "results": [{"memory_id": "m1"}],
                    "context_window": "ctx"}
        return {"ok": True}

    @property
    def last(self):
        return self.calls[-1]


def test_ingest_builds_request():
    c = RecordingClient()
    c.ingest("hello", "alice", memory_type="fact")
    method, path, payload, _ = c.last
    assert method == "POST" and path == "/api/v1/ingest"
    assert payload["text"] == "hello" and payload["user_id"] == "alice"
    assert payload["memory_type"] == "fact"


def test_retrieve_and_context_helper():
    c = RecordingClient()
    assert c.context_for("q", "alice") == "ctx"
    method, path, payload, _ = c.last
    assert method == "POST" and path == "/api/v1/retrieve"
    assert payload["query"] == "q" and payload["top_k"] == 5


def test_feedback_builds_request():
    c = RecordingClient()
    c.feedback("qid", "m1", useful=True, outcome="success")
    method, path, payload, _ = c.last
    assert path == "/api/v1/feedback"
    assert payload == {"query_id": "qid", "memory_id": "m1", "useful": True,
                       "score": None, "outcome": "success"}


def test_delete_uses_query_params():
    c = RecordingClient()
    c.delete("m1", "alice")
    method, path, _, params = c.last
    assert method == "DELETE" and path == "/api/v1/memories/m1"
    assert params == {"user_id": "alice"}


def test_api_key_header_set():
    c = StatefulClient(api_key="secret")
    assert c.api_key == "secret"


def test_cli_recall_dispatches(monkeypatch, capsys):
    import apps.cli as cli

    captured = {}

    class FakeClient:
        def __init__(self, **kw):
            captured["init"] = kw

        def retrieve(self, query, user, top_k=5):
            captured["retrieve"] = (query, user, top_k)
            return {"results": []}

    monkeypatch.setattr(cli, "StatefulClient", FakeClient)
    rc = cli.main(["recall", "what does alice like?", "--user", "alice", "--top-k", "3"])
    assert rc == 0
    assert captured["retrieve"] == ("what does alice like?", "alice", 3)
