"""Prometheus metrics for stateful.ai.

The ``prometheus-client`` dependency is declared in ``pyproject.toml`` but was
previously unused. This module wires it in and degrades gracefully: if the
library is not installed, every helper becomes a no-op and ``/metrics`` reports
that the exporter is disabled, so observability is opt-in without breaking the
zero-infra path.
"""
from __future__ import annotations

try:
    from prometheus_client import (
        Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST,
    )

    _ENABLED = True
except Exception:  # pragma: no cover - dependency guard
    _ENABLED = False
    CONTENT_TYPE_LATEST = "text/plain"

if _ENABLED:
    REQUEST_COUNT = Counter(
        "stateful_ai_http_requests_total",
        "Total HTTP requests",
        ["method", "path", "status"],
    )
    REQUEST_LATENCY = Histogram(
        "stateful_ai_http_request_duration_seconds",
        "HTTP request latency in seconds",
        ["method", "path"],
    )
    RETRIEVAL_LATENCY = Histogram(
        "stateful_ai_retrieval_duration_seconds",
        "Memory retrieval latency in seconds",
        ["mode"],
    )
    MEMORIES_INGESTED = Counter(
        "stateful_ai_memories_ingested_total",
        "Total memories ingested",
    )

    # --- Retrieval result telemetry ----------------------------------
    RETRIEVAL_RESULTS = Histogram(
        "stateful_ai_retrieval_results_returned",
        "Number of memories returned per retrieval",
        buckets=(0, 1, 2, 3, 5, 8, 13, 21, 50),
    )
    RETRIEVAL_CANDIDATES = Histogram(
        "stateful_ai_retrieval_candidates_considered",
        "Number of candidates considered (post-filter) per retrieval",
        buckets=(0, 1, 5, 10, 20, 50, 100, 200),
    )
    RETRIEVAL_EMPTY = Counter(
        "stateful_ai_retrieval_empty_total",
        "Retrievals that returned zero memories",
    )

    # --- Continual-learning (Stateful-CL) telemetry ---------------------
    FEEDBACK_TOTAL = Counter(
        "stateful_ai_feedback_total",
        "Feedback submissions by recorded status and outcome",
        ["recorded", "outcome"],
    )
    FEEDBACK_REWARD = Histogram(
        "stateful_ai_feedback_reward",
        "Distribution of shaped rewards from feedback",
        buckets=(0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0),
    )
    POLICY_UPDATES = Counter(
        "stateful_ai_cl_policy_updates_total",
        "Total online ranking-policy updates applied",
    )
    REPLAY_INTERACTIONS = Gauge(
        "stateful_ai_cl_replay_interactions",
        "Current number of logged retrieval interactions in the replay buffer",
    )
    REPLAY_LABELED = Gauge(
        "stateful_ai_cl_replay_labeled_examples",
        "Current number of labeled (rewarded) examples in the replay buffer",
    )
    CL_NAMESPACES = Gauge(
        "stateful_ai_cl_policy_namespaces",
        "Number of namespaces with a learned ranking policy",
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


def observe_retrieval_results(returned: int, considered: int) -> None:
    """Record how many memories a retrieval returned vs considered."""
    if not _ENABLED:
        return
    RETRIEVAL_RESULTS.observe(returned)
    RETRIEVAL_CANDIDATES.observe(considered)
    if returned == 0:
        RETRIEVAL_EMPTY.inc()


def observe_feedback(recorded: bool, outcome: str, reward: float) -> None:
    """Record a feedback submission and its shaped reward."""
    if not _ENABLED:
        return
    FEEDBACK_TOTAL.labels(
        recorded=str(recorded).lower(),
        outcome=(outcome or "none").strip().lower() or "none",
    ).inc()
    if recorded:
        FEEDBACK_REWARD.observe(max(0.0, min(1.0, reward)))


def inc_policy_update(n: int = 1) -> None:
    if not _ENABLED:
        return
    POLICY_UPDATES.inc(n)


def set_replay_gauges(interactions: int, labeled: int, namespaces: int) -> None:
    """Snapshot the replay-buffer / policy state for scraping."""
    if not _ENABLED:
        return
    REPLAY_INTERACTIONS.set(interactions)
    REPLAY_LABELED.set(labeled)
    CL_NAMESPACES.set(namespaces)


def render() -> bytes:
    if not _ENABLED:
        return b"# stateful.ai metrics exporter disabled (install prometheus-client)\n"
    return generate_latest()
