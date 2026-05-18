"""Local JSON memory store with hash-indexed exact lookup and version history.

The Flask demonstration deployment uses this store as the canonical record
store so it can run on a single AWS Free Tier EC2 instance with no managed
database cost. Mutations use an advisory file lock to reduce multi-process
Gunicorn write races on Unix systems.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any, Iterator
from uuid import uuid4

try:
    import fcntl
except Exception:  # pragma: no cover - non-Unix fallback
    fcntl = None  # type: ignore[assignment]


def utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def hash_key(key: str) -> str:
    """Return a stable SHA-256 hash for O(1) exact-key indexing."""
    return hashlib.sha256(key.strip().lower().encode("utf-8")).hexdigest()


@dataclass
class StoredMemory:
    """Serializable canonical memory record."""

    memory_id: str
    user_id: str
    content: str
    key: str
    namespace: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    importance_score: float = 0.5
    access_count: int = 0
    version: int = 1
    status: str = "active"
    created_at: str = field(default_factory=utcnow)
    updated_at: str = field(default_factory=utcnow)


class JsonMemoryStore:
    """Persistent local memory store backed by JSON and hash maps."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock_path = self.path.with_suffix(".lock")
        self._lock = RLock()
        self._records: dict[str, StoredMemory] = {}
        self._hash_index: dict[str, str] = {}
        self._versions: dict[str, list[dict[str, Any]]] = {}
        self._load()

    def create(
        self,
        *,
        content: str,
        user_id: str,
        key: str | None = None,
        namespace: str = "",
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance_score: float = 0.5,
    ) -> StoredMemory:
        memory = StoredMemory(
            memory_id=str(uuid4()),
            user_id=user_id,
            content=content,
            key=key or content,
            namespace=namespace or f"user:{user_id}",
            metadata=metadata or {},
            tags=tags or [],
            importance_score=max(0.0, min(1.0, importance_score)),
        )
        with self._locked_reload():
            self._records[memory.memory_id] = memory
            self._hash_index[self._hash_for(memory.user_id, memory.key)] = memory.memory_id
            self._versions.setdefault(memory.memory_id, [])
            self._persist_unlocked()
        return memory

    def get(self, memory_id: str, include_deleted: bool = False) -> StoredMemory | None:
        with self._lock:
            self._load_unlocked()
            memory = self._records.get(memory_id)
            if not memory:
                return None
            if not include_deleted and memory.status != "active":
                return None
            return memory

    def get_by_key(self, user_id: str, key: str) -> StoredMemory | None:
        """O(1) exact lookup through the hash-indexed path."""
        with self._lock:
            self._load_unlocked()
            memory_id = self._hash_index.get(self._hash_for(user_id, key))
            memory = self._records.get(memory_id) if memory_id else None
            return memory if memory and memory.status == "active" else None

    def list(self, user_id: str, include_deleted: bool = False) -> list[StoredMemory]:
        with self._lock:
            self._load_unlocked()
            rows = [m for m in self._records.values() if m.user_id == user_id]
        if not include_deleted:
            rows = [m for m in rows if m.status == "active"]
        return sorted(rows, key=lambda m: m.created_at, reverse=True)

    def list_all(self, include_deleted: bool = True) -> list[StoredMemory]:
        with self._lock:
            self._load_unlocked()
            rows = list(self._records.values())
        if not include_deleted:
            rows = [m for m in rows if m.status == "active"]
        return sorted(rows, key=lambda m: (m.user_id, m.created_at))

    def update(
        self,
        memory_id: str,
        *,
        content: str | None = None,
        key: str | None = None,
        metadata: dict[str, Any] | None = None,
        tags: list[str] | None = None,
        importance_score: float | None = None,
    ) -> StoredMemory | None:
        with self._locked_reload():
            memory = self._records.get(memory_id)
            if not memory or memory.status != "active":
                return None
            previous = asdict(memory)
            old_hash = self._hash_for(memory.user_id, memory.key)
            if content is not None:
                memory.content = content
            if key is not None:
                memory.key = key
            if metadata is not None:
                memory.metadata = metadata
            if tags is not None:
                memory.tags = tags
            if importance_score is not None:
                memory.importance_score = max(0.0, min(1.0, importance_score))
            memory.version += 1
            memory.updated_at = utcnow()
            self._versions.setdefault(memory_id, []).append(previous)
            self._hash_index.pop(old_hash, None)
            self._hash_index[self._hash_for(memory.user_id, memory.key)] = memory.memory_id
            self._persist_unlocked()
            return memory

    def versions(self, memory_id: str) -> list[dict[str, Any]]:
        with self._lock:
            self._load_unlocked()
            current = self._records.get(memory_id)
            history = list(self._versions.get(memory_id, []))
            if current:
                history.append(asdict(current))
            return history

    def bump_access(self, memory_id: str) -> None:
        with self._locked_reload():
            memory = self._records.get(memory_id)
            if memory:
                memory.access_count += 1
                memory.updated_at = utcnow()
                self._persist_unlocked()

    def delete(self, memory_id: str) -> bool:
        with self._locked_reload():
            memory = self._records.get(memory_id)
            if not memory or memory.status == "deleted":
                return False
            previous = asdict(memory)
            memory.status = "deleted"
            memory.version += 1
            memory.updated_at = utcnow()
            self._versions.setdefault(memory_id, []).append(previous)
            self._hash_index.pop(self._hash_for(memory.user_id, memory.key), None)
            self._persist_unlocked()
            return True

    def export_payload(self, include_deleted: bool = True) -> dict[str, Any]:
        with self._lock:
            self._load_unlocked()
            records = [asdict(memory) for memory in self._records.values()]
            if not include_deleted:
                records = [memory for memory in records if memory["status"] == "active"]
            return {"records": records, "versions": self._versions}

    def import_payload(self, payload: dict[str, Any], replace: bool = False) -> int:
        imported = 0
        with self._locked_reload():
            if replace:
                self._records = {}
                self._hash_index = {}
                self._versions = {}
            for item in payload.get("records", []):
                item = {**item}
                item.setdefault("version", 1)
                item.setdefault("status", "active")
                memory = StoredMemory(**item)
                self._records[memory.memory_id] = memory
                if memory.status == "active":
                    self._hash_index[self._hash_for(memory.user_id, memory.key)] = memory.memory_id
                imported += 1
            for memory_id, versions in payload.get("versions", {}).items():
                self._versions[memory_id] = list(versions)
            self._persist_unlocked()
        return imported

    def _hash_for(self, user_id: str, key: str) -> str:
        return hash_key(f"{user_id}:{key}")

    @contextmanager
    def _locked_reload(self) -> Iterator[None]:
        with self._lock:
            with self._file_lock():
                self._load_unlocked()
                yield

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _load(self) -> None:
        with self._lock:
            self._load_unlocked()
            if not self.path.exists():
                self._persist_unlocked()

    def _load_unlocked(self) -> None:
        if not self.path.exists():
            self._records = {}
            self._hash_index = {}
            self._versions = {}
            return
        raw = json.loads(self.path.read_text() or "{}")
        self._records = {}
        for item in raw.get("records", []):
            item.setdefault("version", 1)
            item.setdefault("status", "active")
            self._records[item["memory_id"]] = StoredMemory(**item)
        self._hash_index = dict(raw.get("hash_index", {}))
        if not self._hash_index:
            self._hash_index = {
                self._hash_for(memory.user_id, memory.key): memory.memory_id
                for memory in self._records.values()
                if memory.status == "active"
            }
        self._versions = {key: list(value) for key, value in raw.get("versions", {}).items()}

    def _persist_unlocked(self) -> None:
        payload = {
            "records": [asdict(memory) for memory in self._records.values()],
            "hash_index": self._hash_index,
            "versions": self._versions,
        }
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True))
        tmp.replace(self.path)
