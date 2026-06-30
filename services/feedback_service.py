"""Feedback service — closes the Stateful-CL learning loop (loop L1).

Flow:
    retrieve() logs served candidates + features to the replay buffer, keyed by
    query_id  ->  the agent later reports usefulness via POST /feedback  ->  this
    service shapes a reward, updates the online ranking policy for that
    namespace, records retrieval-feedback counters on the memory, and labels the
    replay example for future parametric loops (L2/L3).

Everything degrades safely: if continual learning is disabled, or the
interaction has expired from the bounded buffer, the call is a well-formed
no-op rather than an error.
"""
from __future__ import annotations

import logging
from typing import Any

from core.config.settings import settings
from core.observability import metrics
from domain.learning.feedback import shape_reward
from domain.learning.registry import get_ranking_policy, get_replay_buffer

logger = logging.getLogger(__name__)


class FeedbackResult:
    """Lightweight result object for the router to serialize."""

    def __init__(
        self,
        recorded: bool,
        reward: float,
        namespace: str,
        weights: dict[str, float],
        policy_updates: int,
        message: str,
    ) -> None:
        self.recorded = recorded
        self.reward = reward
        self.namespace = namespace
        self.weights = weights
        self.policy_updates = policy_updates
        self.message = message


class FeedbackService:
    """Turns a /feedback call into an online learning update."""

    def __init__(self, relational_store: Any) -> None:
        self._db = relational_store

    async def record(
        self,
        query_id: str,
        memory_id: str,
        useful: bool | None = None,
        score: float | None = None,
        outcome: str = "",
    ) -> FeedbackResult:
        if not settings.continual_learning_enabled:
            return FeedbackResult(
                recorded=False, reward=0.0, namespace="", weights={},
                policy_updates=0,
                message="continual_learning_enabled is False; feedback ignored.",
            )

        buffer = get_replay_buffer()
        policy = get_ranking_policy()

        interaction = buffer.get(query_id)
        if interaction is None:
            metrics.observe_feedback(False, outcome, 0.0)
            return FeedbackResult(
                recorded=False, reward=0.0, namespace="", weights={},
                policy_updates=0,
                message=f"No logged interaction for query_id={query_id} "
                        "(expired from buffer or unknown).",
            )

        cand = interaction.candidate(memory_id)
        if cand is None:
            metrics.observe_feedback(False, outcome, 0.0)
            return FeedbackResult(
                recorded=False, reward=0.0, namespace=interaction.namespace,
                weights=policy.weights(interaction.namespace),
                policy_updates=policy.update_count(interaction.namespace),
                message=f"memory_id={memory_id} was not served for this query.",
            )

        # Integrity signal: penalize confirmed-contradicted (stale) memories.
        contradicted = False
        try:
            memory = await self._db.get_memory(memory_id)
            cstatus = memory.contradiction_status
            cstatus = cstatus if isinstance(cstatus, str) else getattr(cstatus, "value", cstatus)
            contradicted = cstatus == "confirmed"
        except Exception:
            memory = None

        reward = shape_reward(
            useful=useful,
            score=score,
            outcome=outcome,
            contradicted=contradicted,
            success_bonus=settings.cl_reward_success_bonus,
            contradiction_penalty=settings.cl_reward_contradiction_penalty,
        )

        # Online policy update (loop L1) + label the replay example.
        policy.update(interaction.namespace, cand.features, reward)
        buffer.attach_reward(query_id, memory_id, reward)

        # Telemetry: feedback, reward distribution, policy + replay state.
        metrics.observe_feedback(True, outcome, reward)
        metrics.inc_policy_update()
        buf_stats = buffer.stats()
        metrics.set_replay_gauges(
            interactions=buf_stats.get("interactions", 0),
            labeled=buf_stats.get("labeled", 0),
            namespaces=len(policy.namespaces()),
        )

        # Record retrieval-feedback counters on the memory itself.
        if memory is not None:
            try:
                memory.retrieval_count += 1
                if reward >= 0.5:
                    memory.successful_retrieval_count += 1
                await self._db.update_memory(memory)
            except Exception as e:
                logger.debug(f"Could not persist feedback counters: {e}")

        return FeedbackResult(
            recorded=True,
            reward=round(reward, 4),
            namespace=interaction.namespace,
            weights=policy.weights(interaction.namespace),
            policy_updates=policy.update_count(interaction.namespace),
            message="feedback applied; ranking policy updated.",
        )
