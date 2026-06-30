"""Stateful-CL — the continual-learning subsystem for stateful.ai.

This package turns stateful.ai's memory corpus into a self-improving retrieval
system. It is intentionally dependency-free (pure Python / stdlib) so it
preserves the project's zero-infra promise and stays disabled by default
(``settings.continual_learning_enabled = False``).

Loops implemented here:
    * L1 — online learned ranking policy (per-namespace, EWC-anchored).
    * Replay buffer — the substrate (logged retrieval interactions).
    * Feedback reward shaping — explicit + implicit signal fusion.
    * Continual-learning metrics — Backward / Forward Transfer, forgetting.

See ``docs/continual_learning_design.md`` for the full architecture.
"""
from __future__ import annotations

from domain.learning.features import FEATURE_NAMES, extract_features
from domain.learning.online_scorer import OnlineRankingPolicy
from domain.learning.replay import (
    CandidateRecord, RetrievalInteraction, ReplayBuffer,
)
from domain.learning.feedback import FeedbackSignal, shape_reward
from domain.learning.cl_metrics import (
    average_accuracy, backward_transfer, forward_transfer,
    forgetting, summarize_matrix,
)
from domain.learning.registry import (
    get_ranking_policy, get_replay_buffer, reset_learning_state,
)

__all__ = [
    "FEATURE_NAMES",
    "extract_features",
    "OnlineRankingPolicy",
    "CandidateRecord",
    "RetrievalInteraction",
    "ReplayBuffer",
    "FeedbackSignal",
    "shape_reward",
    "average_accuracy",
    "backward_transfer",
    "forward_transfer",
    "forgetting",
    "summarize_matrix",
    "get_ranking_policy",
    "get_replay_buffer",
    "reset_learning_state",
]
