"""Bounded, in-process audit log of mutating API operations.

Surfaces *who did what, when* — the missing operator-facing accountability
trail. Records every state-changing request (POST/PATCH/PUT/DELETE on the API)
with the resolved principal, method, path, status, and timestamp. Bounded ring
buffer so memory stays flat; swap for a durable sink (DB/SIEM) in production.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from threading import Lock


@dataclass
class AuditEntry:
    ts: str
    principal: str
    tenant: str
    method: str
    path: str
    status: int

    def to_dict(self) -> dict:
        return asdict(self)


class AuditLog:
    def __init__(self, capacity: int = 1000) -> None:
        self._entries: "deque[AuditEntry]" = deque(maxlen=capacity)
        self._lock = Lock()

    def record(self, principal: str, tenant: str, method: str,
               path: str, status: int) -> None:
        entry = AuditEntry(
            ts=datetime.now(timezone.utc).isoformat(),
            principal=principal, tenant=tenant,
            method=method, path=path, status=status,
        )
        with self._lock:
            self._entries.append(entry)

    def list(self, limit: int = 100) -> list[dict]:
        with self._lock:
            items = list(self._entries)[-limit:]
        return [e.to_dict() for e in reversed(items)]  # newest first

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_audit_log = AuditLog()


def get_audit_log() -> AuditLog:
    return _audit_log
