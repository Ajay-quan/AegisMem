"""stateful.ai Python SDK — a tiny, dependency-free client for the memory API.

    from sdk import StatefulClient
    mem = StatefulClient("http://localhost:8000")
    mem.ingest("Alice prefers Python and FAISS.", user_id="alice", memory_type="fact")
    hits = mem.retrieve("what does alice like?", user_id="alice")
    mem.feedback(hits["query_id"], hits["results"][0]["memory_id"], useful=True)

Uses only the standard library (``urllib``) so it adds zero dependencies and
works anywhere Python runs.
"""
from __future__ import annotations

from sdk.client import StatefulClient, StatefulError

__all__ = ["StatefulClient", "StatefulError"]
