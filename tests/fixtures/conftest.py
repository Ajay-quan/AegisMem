"""Shared test fixtures for AegisMem tests."""
from __future__ import annotations

import pytest
import pytest_asyncio

from adapters.llm.mock_client import MockLLMClient
from adapters.embeddings.backend import MockEmbeddingBackend
from adapters.vector_store.qdrant_store import InMemoryVectorStore
from adapters.graph_store.neo4j_store import MockGraphStore
from core.schemas.memory import MemoryItem, MemoryType, SourceType, Observation


class MockPostgresStore:
    """In-memory mock for PostgresStore."""

    def __init__(self):
        self._memories: dict[str, MemoryItem] = {}
        self._facts = []
        self._contradictions = []
        self._logs = []
        self._evals = []

    async def initialize(self):
        pass

    async def save_memory(self, memory: MemoryItem) -> MemoryItem:
        self._memories[memory.memory_id] = memory
        return memory

    async def get_memory(self, memory_id: str) -> MemoryItem:
        from core.exceptions import MemoryNotFoundError
        if memory_id not in self._memories:
            raise MemoryNotFoundError(memory_id)
        return self._memories[memory_id]

    async def update_memory(self, memory: MemoryItem) -> MemoryItem:
        self._memories[memory.memory_id] = memory
        return memory

    async def delete_memory(self, memory_id: str, user_id: str) -> None:
        if memory_id in self._memories:
            self._memories[memory_id].status = "deleted"

    async def list_memories(self, user_id: str, namespace: str = "", memory_type: str = "",
                             status: str = "active", limit: int = 50, offset: int = 0) -> list:
        results = [
            m for m in self._memories.values()
            if m.user_id == user_id
            and (not status or (m.status if isinstance(m.status, str) else m.status.value) == status)
            and (not namespace or m.namespace == namespace)
            and (not memory_type or (m.memory_type if isinstance(m.memory_type, str) else m.memory_type.value) == memory_type)
        ]
        return results[offset:offset + limit]

    async def count_memories(self, user_id: str, namespace: str = "") -> int:
        return len([m for m in self._memories.values() if m.user_id == user_id])

    async def save_fact(self, fact):
        self._facts.append(fact)
        return fact

    async def get_facts_for_user(self, user_id: str, subject: str = ""):
        return []

    async def save_contradiction(self, report_id, a_id, b_id, description, confidence):
        self._contradictions.append({"report_id": report_id, "a_id": a_id, "b_id": b_id})

    async def list_contradictions(self, resolved: bool = False):
        return []

    async def save_eval_result(self, eval_name, run_id, metrics, config):
        self._evals.append({"eval_name": eval_name, "run_id": run_id})

    async def get_operation_logs(self, memory_id: str):
        return []

    async def close(self):
        pass


@pytest.fixture
def mock_db():
    return MockPostgresStore()


@pytest.fixture
def mock_embed():
    return MockEmbeddingBackend(dim=384)


@pytest.fixture
def mock_llm():
    client = MockLLMClient()
    # Register common responses
    client.register_response(
        "contradicts",
        '{"contradicts": true, "confidence": 0.9, "description": "Conflicting location", "resolution_suggestion": "Use newer"}'
    )
    client.register_response(
        "reflection",
        '{"reflections": [{"content": "User shows strong interest in AI", "confidence": 0.85, "supporting_indices": [0, 1]}]}'
    )
    client.register_response(
        "update",
        '{"action": "supersede", "reason": "New info replaces old", "merged_content": "user moved to Austin", "confidence": 0.9}'
    )
    return client


@pytest_asyncio.fixture
async def mock_vector_store():
    store = InMemoryVectorStore()
    await store.initialize(dimension=384)
    return store


@pytest.fixture
def mock_graph():
    return MockGraphStore()


@pytest.fixture
def sample_memory(mock_db) -> MemoryItem:
    return MemoryItem(
        namespace="user:test123",
        user_id="test123",
        agent_id="agent1",
        memory_type=MemoryType.OBSERVATION,
        content="User likes Python and machine learning",
        source_type=SourceType.USER_MESSAGE,
        importance_score=0.7,
    )


@pytest.fixture
def sample_observation() -> Observation:
    return Observation(
        session_id="sess1",
        user_id="test123",
        agent_id="agent1",
        content="User mentioned they prefer remote work",
        source_type=SourceType.USER_MESSAGE,
    )
