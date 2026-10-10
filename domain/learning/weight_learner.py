"""Online, per-user learning of retrieval ranking weights.

Clients report whether each retrieved memory was actually used. Every report is
a labelled example ``(signals, used)``. We fit a logistic model over the ranking
signals with one SGD step per example, then map the coefficients back onto the
weight simplex used by ``compute_composite_score``.

Guardrails:
  * Learned weights are only served after ``min_events`` feedback events.
  * L2 pull toward the global default weights prevents drift on sparse data.
  * Weights are projected to ``[floor, 1]`` and renormalised to sum to 1, so no
    signal can be switched off entirely by a burst of noisy feedback.
  * State is per user and can be reset at any time.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

from core.config.settings import settings

SIGNALS: tuple[str, ...] = ("semantic", "recency", "importance", "access")


def default_weights() -> dict[str, float]:
    return {
        "semantic": settings.weight_semantic,
        "recency": settings.weight_recency,
        "importance": settings.weight_importance,
        "access": settings.weight_access,
    }


@dataclass
class UserWeightState:
    user_id: str
    weights: dict[str, float] = field(default_factory=default_weights)
    bias: float = 0.0
    events: int = 0
    positives: int = 0
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "weights": {k: round(v, 4) for k, v in self.weights.items()},
            "events": self.events,
            "positives": self.positives,
            "updated_at": self.updated_at.isoformat(),
        }


class InMemoryLearningStore:
    """Process-local state store. Swap for a Postgres-backed store in production."""

    def __init__(self) -> None:
        self._states: dict[str, UserWeightState] = {}

    async def get(self, user_id: str) -> UserWeightState | None:
        return self._states.get(user_id)

    async def put(self, state: UserWeightState) -> None:
        self._states[state.user_id] = state

    async def delete(self, user_id: str) -> None:
        self._states.pop(user_id, None)


class OnlineWeightLearner:
    def __init__(
        self,
        store: InMemoryLearningStore | None = None,
        learning_rate: float = 0.05,
        prior_strength: float = 0.02,
        floor: float = 0.05,
        min_events: int = 20,
    ) -> None:
        self._store = store or InMemoryLearningStore()
        self.lr = learning_rate
        self.prior_strength = prior_strength
        self.floor = floor
        self.min_events = min_events

    async def weights_for(self, user_id: str) -> dict[str, float] | None:
        """Learned weights for ranking, or None if not enough evidence yet."""
        state = await self._store.get(user_id)
        if state is None or state.events < self.min_events:
            return None
        return dict(state.weights)

    async def state(self, user_id: str) -> UserWeightState | None:
        return await self._store.get(user_id)

    async def reset(self, user_id: str) -> None:
        await self._store.delete(user_id)

    async def observe(
        self, user_id: str, examples: list[tuple[dict[str, float], bool]],
    ) -> UserWeightState:
        """Apply one SGD step per (signals, used) example."""
        state = await self._store.get(user_id) or UserWeightState(user_id=user_id)
        prior = default_weights()
        # Scale so weights in [0,1] produce usable logits.
        scale = 4.0

        for signals, used in examples:
            x = {s: float(signals.get(s, 0.0)) for s in SIGNALS}
            z = state.bias + scale * sum(state.weights[s] * x[s] for s in SIGNALS)
            p = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, z))))
            err = (1.0 if used else 0.0) - p
            for s in SIGNALS:
                grad = err * scale * x[s] - self.prior_strength * (state.weights[s] - prior[s])
                state.weights[s] += self.lr * grad
            state.bias += self.lr * err
            state.weights = self._project(state.weights)
            state.events += 1
            state.positives += int(used)

        state.updated_at = datetime.now(timezone.utc)
        await self._store.put(state)
        return state

    def _project(self, w: dict[str, float]) -> dict[str, float]:
        clipped = {s: min(1.0, max(self.floor, w[s])) for s in SIGNALS}
        total = sum(clipped.values())
        return {s: v / total for s, v in clipped.items()}
