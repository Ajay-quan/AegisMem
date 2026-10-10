"""Prometheus metrics. Labels are low-cardinality by construction: route *templates*,
decision enums and tenant ids (bounded by customers), never user ids, memory ids or content."""
from __future__ import annotations

from prometheus_client import CollectorRegistry, Counter, Histogram

REGISTRY = CollectorRegistry(auto_describe=True)

HTTP_REQUESTS = Counter(
    "aegis_http_requests_total", "HTTP requests", ["method", "route", "status"], registry=REGISTRY)
HTTP_LATENCY = Histogram(
    "aegis_http_request_seconds", "HTTP request latency", ["method", "route"], registry=REGISTRY,
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5))
AUTH_FAILURES = Counter(
    "aegis_auth_failures_total", "Rejected API calls", ["reason"], registry=REGISTRY)

GUARD_DECISIONS = Counter(
    "aegis_guard_decisions_total", "Read-time guard decisions that changed what was served",
    ["tenant", "action", "reason"], registry=REGISTRY)
WRITES = Counter(
    "aegis_memory_writes_total", "Memory writes by outcome", ["tenant", "source", "outcome"],
    registry=REGISTRY)
ERASURES = Counter(
    "aegis_erasures_total", "Erasure requests", ["tenant", "verified"], registry=REGISTRY)
ERASED_ITEMS = Counter(
    "aegis_erased_items_total", "Items erased by relation", ["tenant", "relation"], registry=REGISTRY)
OUTBOX_PROCESSED = Counter(
    "aegis_outbox_ops_total", "Vector-sync outbox operations processed", ["op", "result"],
    registry=REGISTRY)
LEARNING_EVENTS = Counter(
    "aegis_learning_events_total", "Durable learning events appended", ["tenant", "component"],
    registry=REGISTRY)
