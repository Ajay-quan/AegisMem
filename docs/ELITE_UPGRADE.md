# stateful.ai — Elite Upgrade Dossier

A senior-level diagnosis of stateful.ai and the roadmap to take it from a strong
portfolio project to a research-grade, production-grade, adoption-ready
open-source memory system. This document is deliberately honest: it challenges
the existing design rather than flattering it.

---

## 1. Diagnosis — what stateful.ai is today

A persistent memory layer for long-running LLM agents with a genuinely clean
spine: `API → services → domain → adapters`, every backend behind a swappable
adapter, and a zero-infra default (in-memory stores + deterministic mock
embeddings) so it boots anywhere. Retrieval is real hybrid search — dense + BM25
fused with Reciprocal Rank Fusion, multi-signal composite scoring, and a
heuristic/cross-encoder reranker. It ships an MCP server, Prometheus metrics,
structured logs, API-key auth, rate limiting, ADRs, an architecture paper, and a
test suite. As of v0.3.0 it also has a working **continual-learning loop**
(online EWC-anchored ranking policy + replay buffer + feedback API), **telemetry
for that loop**, and **PII redaction at ingest**.

Verdict: the bones are better than 90% of "AI memory" repos. The gaps are about
*credibility of claims*, *parametric depth*, *scale proof*, and *adoption
surface* — not architecture.

---

## 2. Gap analysis (challenging every decision)

**Research / evaluation gaps.**
- The headline benchmark is synthetic (70 memories, 10 queries, P@1=1.0). That
  number persuades no reviewer. *Fix:* run LoCoMo and LongMemEval; publish
  dense-vs-hybrid-vs-reranker and static-vs-learned ablations.
- No external baseline comparison (Mem0 / Zep / Letta / A-MEM) on a shared
  metric. *Fix:* a comparison harness + table.

**AI/ML depth gaps.**
- Learning today is non-parametric (ranking weights). The reranker and embedder
  are still frozen. *Fix:* L2 — replay+EWC LoRA fine-tuning of the cross-encoder;
  L3 — per-namespace embedding projection heads.
- Importance and forgetting are heuristic, not learned. *Fix:* learn importance
  from outcomes; learned per-type decay (adaptive forgetting).
- No temporal reasoning: `valid_from`/`valid_to` exist but are passive. *Fix:*
  bi-temporal belief revision (Zep's strongest capability).

**Architecture / scale gaps.**
- Single-node story dominates; the Postgres/Qdrant/Neo4j path isn't proven under
  concurrency. *Fix:* multi-process consistency tests; connection pooling;
  per-namespace isolation under concurrent writes.
- No async background worker actually running (sleep-time consolidation is
  scaffolded, not scheduled). *Fix:* wire `apps/worker` to a real queue.
- Retrieval has no caching layer. *Fix:* request-scoped + embedding cache.

**Security gaps.**
- One global API key, not per-tenant scoped keys/JWT. *Fix:* scoped keys,
  per-key namespaces. (PII redaction now shipped.)
- No audit log of write/delete operations surfaced to operators.

**Product / adoption gaps (now partially closed).**
- DX: shipped a zero-dep **Python SDK** + **CLI** in v0.3.0. Next: a hosted
  remote MCP server (one-URL install in Claude/ChatGPT) and a `.mcpb` bundle.
- Onboarding: README is thorough but long; needs a 30-second "why + try" top.
- No live hosted demo / no comparison table (addressed in README upgrade).

**Maintainability / OSS-readiness gaps.**
- `version` is inconsistent (`pyproject` says 0.1.0, app says 0.2.0). *Fix:*
  single source of truth.
- No CONTRIBUTING, issue templates, or coverage gate in CI.

---

## 3. Research grounding (what the upgrades are built on)

| Decision | Chosen approach | Alternatives considered | Why |
|---|---|---|---|
| Online ranking | Per-namespace convex policy + online EWC | full RL; static weights | gradient-free, anti-forgetting, multi-tenant safe |
| Anti-forgetting | Experience replay (corpus) + EWC/Fisher + LoRA (L2) | naive fine-tune | replay is the strongest LM defense; EWC cut forgetting ~46% in KG evals; LoRA reduces forgetting vs full FT |
| Temporal facts | Bi-temporal validity + belief revision | overwrite; dup facts | matches Zep/Graphiti's best-in-class temporal correctness |
| Memory structure | Hybrid (vector+BM25+graph) + A-MEM dynamic linking | dense-only RAG | embeddings alone miss identifiers, structure, conflict handling |
| Privacy | Regex + Luhn redaction at ingest | model-based PII NER | deterministic, zero-dep, auditable; NER can be a later opt-in |
| DX | stdlib SDK + CLI + MCP | SDK with heavy deps | zero install friction; widest reach |

Sources: Mem0, Zep/Graphiti, Letta (sleep-time compute), A-MEM (NeurIPS 2025),
EWC (Kirkpatrick 2017), GEM/BWT-FWT (Lopez-Paz 2017), experience replay & LoRA
continual-learning literature.

---

## 4. Target architecture (production-grade)

```mermaid
flowchart TD
    subgraph Edge
      C[Agent / MCP client / SDK / CLI] --> GW[FastAPI gateway: auth, rate-limit, PII redact]
    end
    GW --> ING[Ingest] --> STORES
    GW --> RET[Retrieve: dense+BM25+RRF -> learned scorer -> reranker]
    RET --> STORES[(Relational | Vector | Graph)]
    RET --> CACHE[(Embedding + result cache)]
    GW --> FB[Feedback] --> CL
    subgraph CL [Stateful-CL learning plane - background]
      BUF[(Replay buffer)] --> L1[online ranker]
      BUF --> L2[LoRA reranker + replay + EWC]
      BUF --> L3[embedding heads]
      WORK[Sleep-time worker: consolidate/reflect/forget] --> STORES
      EVAL{Continual eval: P@k, BWT/FWT} -->|promote on win| RET
    end
    GW --> OBS[/Prometheus + JSON logs/]
```

Principles: API-first, adapter-per-backend, learning strictly in the background,
every learned component degrades to a safe heuristic default, promotion gated by
eval. Nothing in the serving path blocks on learning.

---

## 5. Feature prioritization

1. **Must-have credibility:** LoCoMo/LongMemEval harness + baseline comparison.
2. **Depth:** L2 LoRA reranker (replay+EWC); temporal belief revision.
3. **Scale:** concurrency tests + caching + real background worker.
4. **Security:** scoped per-tenant keys; operation audit log. *(PII redaction done.)*
5. **Adoption:** remote MCP server + `.mcpb` bundle; hosted demo. *(SDK + CLI done.)*
6. **Polish:** version single-source; CONTRIBUTING + CI coverage gate.

---

## 6. Shipped in this upgrade cycle (v0.3.0)

- Continual-learning subsystem (`domain/learning/`): online EWC ranking policy,
  replay buffer, reward shaping, BWT/FWT metrics, registry.
- Feedback API (`/feedback`, `/learning/stats`) + retrieval logging + learned
  weights in the scorer.
- Continual-eval harness (`scripts/continual_eval.py`) — EWC turns BWT −0.64 into
  +0.20 while beating the static baseline; doubles as a promotion gate.
- Telemetry: feedback/reward/policy/replay/retrieval-result Prometheus series.
- **PII redaction at ingest** (`domain/privacy/`) — email/phone/card(Luhn)/SSN/
  IP/secret, config-gated, reported.
- **Zero-dependency Python SDK** (`sdk/`) + **CLI** (`apps/cli.py`).
- 135 passing tests.

---

## 7. Testing & evaluation strategy

- Unit: scoring, policy (convexity, anti-forgetting, isolation), replay, metrics,
  reward, redaction, SDK/CLI.
- Integration: ingest→retrieve, hybrid retrieval, feedback loop end-to-end.
- Continual eval: task-incremental BWT/FWT promotion gate.
- Next: real-corpus eval (LoCoMo/LongMemEval), concurrency/consistency tests,
  CI coverage threshold, load test for p95 under concurrency.

---

## 8. Deployment strategy

- Dev: zero-infra FastAPI (`uvicorn apps.api.main:app`).
- Single-node demo: Dockerized Flask + FAISS/JSON on AWS Free Tier (runbook in
  README).
- Production: docker-compose `observability`/backends profiles → Postgres +
  Qdrant + Neo4j + Redis + Prometheus/Grafana; horizontal scale behind the
  stateless API; learning plane on a worker.
- Distribution: remote MCP connector (Claude/ChatGPT) + `.mcpb` one-click bundle.

---

## 9. Self-critique (honest)

- The continual-learning win is currently demonstrated on a *synthetic*
  task-incremental benchmark. It is methodologically correct and shows the right
  dynamics, but real-corpus validation is required before claiming SOTA.
- Learning is still non-parametric; the LoRA reranker (L2) is the next real
  depth jump and is not yet implemented.
- Multi-tenant isolation is correct *by construction* (per-namespace), but not
  yet proven under concurrent load.
- The product is adoption-ready for developers (SDK/CLI/MCP-local) but not yet
  for non-technical users (needs the hosted remote MCP).

None of these are hand-waved — each is tracked above with a concrete fix. The
honesty is the point: this reads like an engineering plan, not a pitch.

---

## 10. Future roadmap (next cycles)

1. Real-corpus eval + public leaderboard table vs Mem0/Zep/Letta.
2. L2 LoRA reranker and L3 embedding heads (parametric continual learning).
3. Temporal knowledge graph + bi-temporal belief revision.
4. Remote MCP server + `.mcpb` + hosted demo.
5. Scoped multi-tenant auth, audit log, concurrency hardening, caching.
6. Sleep-time worker scheduling (consolidation/reflection/adaptive forgetting).
