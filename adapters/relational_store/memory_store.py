"""Async in-memory relational store with optional JSON persistence.

This is the zero-infrastructure fallback for the FastAPI service. It implements
the exact async interface of :class:`PostgresStore` so the full product —
ingestion, hybrid retrieval, updates, contradiction detection, reflection, and
evaluation — runs end-to-end with nothing but ``pip install`` and no database,
queue, or vector service required.

When ``data_dir`` is provided, records are persisted to a single JSON file so
state survives restarts (mirroring the durability story of the demo stack).
This keeps the "runs anywhere" claim honest while preserving a clean migration
path to Postgres for production.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.exceptions import MemoryNotFoundError
from core.schemas.memory import MemoryItem

logger = logging.getLogger(__name__)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class InMemoryRelationalStore:
    """Dict-backed async relational store, API-compatible with PostgresStore."""

    def __init__(self, data_dir: str | None = None) -> None:
        self._memories: dict[str, MemoryItem] = {}
        self._facts: list[dict[str, Any]] = []
        self._contradictions: list[dict[str, Any]] = []
        self._op_logs: dict[str, list[dict[str, Any]]] = {}
        self._evals: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._path = Path(data_dir) / "relational_store.json" if data_dir else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def initialize(self) -> None:
        if self._path and self._path.exists():
            try:
                payload = json.loads(self._path.read_text())
                for raw in payload.get("memories", []):
                    item = MemoryItem(**raw)
                    self._memories[item.memory_id] = item
                self._facts = payload.get("facts", [])
                self._contradictions = payload.get("contradictions", [])
                self._op_logs = payload.get("op_logs", {})
                self._evals = payload.get("evals", [])
                logger.info(f"Loaded {len(self._memories)} memories from {self._path}")
            except Exception as e:
                logger.warning(f"Could not load persisted store ({e}); starting empty.")
        logger.info("InMemoryRelationalStore ready (zero-infra mode)")

    def _persist(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "memories": [json.loads(m.model_dump_json()) for m in self._memories.values()],
                "facts": self._facts,
                "contradictions": self._contradictions,
                "op_logs": self._op_logs,
                "evals": self._evals,
            }
            self._path.write_text(json.dumps(payload, default=str))
        except Exception as e:
            logger.warning(f"Persist failed: {e}")

    def _log(self, memory_id: str, user_id: str, operation: str) -> None:
        self._op_logs.setdefault(memory_id, []).append(
            {"operation": operation, "at": utcnow().isoformat(), "details": {}}
        )

    # ------------------------------------------------------------------
    # Memory CRUD
    # ------------------------------------------------------------------

    async def save_memory(self, memory: MemoryItem) -> MemoryItem:
        async with self._lock:
            self._memories[memory.memory_id] = memory
            self._log(memory.memory_id, memory.user_id, "create")
            self._persist()
        return memory

    async def get_memory(self, memory_id: str) -> MemoryItem:
        memory = self._memories.get(memory_id)
        if memory is None or (memory.status if isinstance(memory.status, str)
                              else memory.status.value) == "deleted":
            raise MemoryNotFoundError(memory_id)
        return memory

    async def update_memory(self, memory: MemoryItem) -> MemoryItem:
        async with self._lock:
            if memory.memory_id not in self._memories:
                raise MemoryNotFoundError(memory.memory_id)
            memory.updated_at = utcnow()
            self._memories[memory.memory_id] = memory
            self._log(memory.memory_id, memory.user_id, "update")
            self._persist()
        return memory

    async def delete_memory(self, memory_id: str, user_id: str) -> None:
        async with self._lock:
            memory = self._memories.get(memory_id)
            if memory:
                memory.status = "deleted"
                memory.updated_at = utcnow()
                self._log(memory_id, user_id, "delete")
                self._persist()

    @staticmethod
    def _status_str(memory: MemoryItem) -> str:
        return memory.status if isinstance(memory.status, str) else memory.status.value

    @staticmethod
    def _type_str(memory: MemoryItem) -> str:
        mt = memory.memory_type
        return mt if isinstance(mt, str) else mt.value

    async def list_memories(
        self,
        user_id: str,
        namespace: str = "",
        memory_type: str = "",
        status: str = "active",
        limit: int = 50,
        offset: int = 0,
    ) -> list[MemoryItem]:
        items = [m for m in self._memories.values() if m.user_id == user_id]
        if namespace:
            items = [m for m in items if m.namespace == namespace]
        if memory_type:
            items = [m for m in items if self._type_str(m) == memory_type]
        if status:
            items = [m for m in items if self._status_str(m) == status]
        items.sort(key=lambda m: m.created_at, reverse=True)
        return items[offset:offset + limit]

    async def count_memories(self, user_id: str, namespace: str = "") -> int:
        items = [
            m for m in self._memories.values()
            if m.user_id == user_id and self._status_str(m) == "active"
        ]
        if namespace:
            items = [m for m in items if m.namespace == namespace]
        return len(items)

    # ------------------------------------------------------------------
    # Facts / contradictions / logs / evals
    # ------------------------------------------------------------------

    async def save_fact(self, fact: Any) -> Any:
        async with self._lock:
            self._facts.append(json.loads(fact.model_dump_json())
                               if hasattr(fact, "model_dump_json") else dict(fact))
            self._persist()
        return fact

    async def get_facts_for_user(self, user_id: str, subject: str = "") -> list[dict[str, Any]]:
        out = [f for f in self._facts if f.get("user_id") == user_id]
        if subject:
            out = [f for f in out if f.get("subject") == subject]
        return out

    async def save_contradiction(
        self, report_id: str, a_id: str, b_id: str, description: str, confidence: float,
    ) -> None:
        async with self._lock:
            self._contradictions.append({
                "report_id": report_id,
                "memory_a_id": a_id,
                "memory_b_id": b_id,
                "description": description,
                "confidence": confidence,
                "resolved": False,
                "detected_at": utcnow().isoformat(),
            })
            self._persist()

    async def list_contradictions(self, resolved: bool = False) -> list[dict[str, Any]]:
        return [c for c in self._contradictions if c.get("resolved", False) == resolved]

    async def get_operation_logs(self, memory_id: str) -> list[dict[str, Any]]:
        return list(self._op_logs.get(memory_id, []))

    async def save_eval_result(
        self, eval_name: str, run_id: str, metrics: dict[str, Any], config: dict[str, Any],
    ) -> None:
        async with self._lock:
            self._evals.append({
                "eval_name": eval_name, "run_id": run_id,
                "metrics": metrics, "config": config,
                "created_at": utcnow().isoformat(),
            })
            self._persist()

    async def close(self) -> None:
        self._persist()
