"""Prometheus metrics for AegisMem.

The ``prometheus-client`` dependency is declared in ``pyproject.toml`` but was
previously unused. This module wires it in and degrades gracefully: if the
library is not installed, every helper becomes a no-op and ``/metrics`` reports
that the exporter is disabled, so observability is opt-in without breaking the
zero-infra path.
"""
from __future__ import annotations

try:
    from prometheus_client import (
        Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST,
    )

    _ENABLED = True
except Exception:  # pragma: no cover - dependency guard
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"

if _ENABLED:
    REQUEST_COUNT = Counter(
        "aegismem_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "aegismem_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
    )
    RETRIEVAL_LATENCY = Histogram(
        "aegismem_retrieval_duration_seconds",
        "Memory retrieval latency in seconds",
        ["mode"],
    )
    MEMORIES_INGESTED = Counter(
        "aegismem_memories_ingested_total",
        "Total memories ingested",
    )


def is_enabled() -> bool:
    return _ENABLED


def observe_request(method: str, path: str, status: int, duration_s: float) -> None:
    if not _ENABLED:
        return
    REQUEST_COUNT.labels(method=method, path=path, status=str(status)).inc()
    REQUEST_LATENCY.labels(method=method, path=path).observe(duration_s)


def observe_retrieval(mode: str, duration_s: float) -> None:
    if not _ENABLED:
        return
    RETRIEVAL_LATENCY.labels(mode=mode).observe(duration_s)


def inc_ingested(n: int = 1) -> None:
    if not _ENABLED:
        return
    MEMORIES_INGESTED.inc(n)


def render() -> bytes:
    if not _ENABLED:
        return b"# AegisMem metrics exporter disabled (install prometheus-client)\n"
    return generate_latest()
