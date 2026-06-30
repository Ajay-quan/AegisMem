# Stateful-CL Continual-Learning Evaluation

Generated: 2026-06-30T01:44:28.845724+00:00

Task-incremental protocol: a single shared ranking policy is trained on a
stream of tasks (each rewards a different retrieval signal) and evaluated on
all tasks after each one. We report Average Accuracy (P@1), Backward Transfer
(BWT; negative = forgetting), Forward Transfer (FWT), and Forgetting.

## Headline

- Static (no learning) average P@1: **0.2675**
- Stateful-CL (EWC) average P@1: **0.3538**  (lift vs static: **+0.0863**)
- BWT with EWC: **+0.1967**  vs  without EWC: **-0.6367**
- Forgetting reduced by EWC: **+0.6367**

## Per-arm metrics

| Arm | Avg Acc | BWT | FWT | Forgetting |
| --- | ---: | ---: | ---: | ---: |
| static | 0.2675 | +0.0 | +0.2567 | 0.0 |
| cl_no_ewc | 0.335 | -0.6367 | +0.15 | 0.6367 |
| cl_ewc | 0.3538 | +0.1967 | +0.2217 | 0.0 |

Interpretation: the learned policy should beat the static baseline on average
accuracy, and the EWC arm should show BWT >= the no-EWC arm — empirical evidence
that anchoring mitigates catastrophic forgetting across the task stream.
