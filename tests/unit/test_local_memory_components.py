"""Tests for local AegisMem demo components."""
from __future__ import annotations

from adapters.graph_store.memory_graph import LocalMemoryGraph
from adapters.relational_store.json_store import JsonMemoryStore
from adapters.vector_store.faiss_store import FaissVectorStore
from services.cache_index import HotMemoryIndex
from services.flask_memory_service import FlaskMemoryService


def test_json_store_hash_lookup_update_delete(tmp_path):
    store = JsonMemoryStore(tmp_path / "memories.json")
    memory = store.create(user_id="u1", key="pref", content="User prefers Flask")
    assert store.get_by_key("u1", "pref").memory_id == memory.memory_id
    updated = store.update(memory.memory_id, key="new-pref", content="User prefers FAISS")
    assert updated.content == "User prefers FAISS"
    assert updated.version == 2
    assert len(store.versions(memory.memory_id)) == 2
    assert store.get_by_key("u1", "pref") is None
    assert store.get_by_key("u1", "new-pref").memory_id == memory.memory_id
    assert store.delete(memory.memory_id) is True
    assert store.get(memory.memory_id) is None


def test_graph_weighted_bfs(tmp_path):
    graph = LocalMemoryGraph(tmp_path / "graph.json")
    graph.add_memory("a", {"content": "root"})
    graph.add_memory("b", {"content": "child"})
    graph.add_memory("c", {"content": "grandchild"})
    graph.connect("a", "b", weight=0.9)
    graph.connect("b", "c", weight=0.8)
    related = graph.traverse("a", depth=2)
    assert [item["memory_id"] for item in related] == ["b", "c"]
    assert related[1]["distance"] == 2


def test_hot_memory_index_uses_expected_structures():
    index = HotMemoryIndex(capacity=2)
    index.upsert("a", priority=0.1, updated_at="2026-01-01T00:00:00Z", payload={})
    index.upsert("b", priority=0.9, updated_at="2026-01-02T00:00:00Z", payload={})
    index.upsert("c", priority=0.8, updated_at="2026-01-03T00:00:00Z", payload={})
    assert index.get("a") is None
    assert [item.memory_id for item in index.recent(2)] == ["c", "b"]


def test_vector_store_persists_and_searches(tmp_path):
    store = FaissVectorStore(tmp_path / "faiss.index", tmp_path / "metadata.pkl")
    store.initialize(3)
    store.upsert("a", [1.0, 0.0, 0.0], {"user_id": "u1", "status": "active"})
    store.upsert("b", [0.0, 1.0, 0.0], {"user_id": "u1", "status": "active"})
    assert store.search([1.0, 0.0, 0.0], top_k=1)[0].id == "a"

    reloaded = FaissVectorStore(tmp_path / "faiss.index", tmp_path / "metadata.pkl")
    reloaded.initialize(3)
    assert reloaded.search([0.0, 1.0, 0.0], top_k=1)[0].id == "b"


def test_service_persistence_restart(tmp_path):
    first = FlaskMemoryService(data_dir=tmp_path, embedding_backend="mock")
    memory = first.ingest(content="Alice prefers AWS Free Tier EC2 demos", user_id="alice", key="aws")
    assert first.get_by_key(user_id="alice", key="aws").memory_id == memory.memory_id

    restarted = FlaskMemoryService(data_dir=tmp_path, embedding_backend="mock")
    assert restarted.get_by_key(user_id="alice", key="aws").memory_id == memory.memory_id
    results = restarted.retrieve(query="AWS Free Tier", user_id="alice", top_k=1)
    assert results and results[0]["memory_id"] == memory.memory_id


def test_json_store_import_export(tmp_path):
    store = JsonMemoryStore(tmp_path / "memories.json")
    memory = store.create(user_id="u1", key="portable", content="Portable memory export")
    payload = store.export_payload()

    restored = JsonMemoryStore(tmp_path / "restored.json")
    assert restored.import_payload(payload, replace=True) == 1
    assert restored.get_by_key("u1", "portable").memory_id == memory.memory_id
