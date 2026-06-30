# Changelog

## [0.3.0] — hardening round 2 (security, evaluation, OSS readiness)

### Security & multi-tenancy
- **Scoped, revocable, named API keys** (`core/security/keys.py`): configure many
  keys via `STATEFUL_AI_API_KEYS` (`name:secret[:tenant]`), each attributable and
  independently revocable. Backward-compatible with the single `STATEFUL_AI_API_KEY`
  (maps to principal `default`). Constant-time matching; per-principal namespace
  scoping helper.
- **Audit log** (`core/security/audit.py`, `GET /api/v1/audit`): every mutating
  API call is recorded with principal, tenant, method, path, status, timestamp
  (bounded ring buffer; newest-first).

### Evaluation credibility
- **Standard IR metrics** (`domain/evaluations/ir_metrics.py`): hit@k,
  precision@k, recall@k, MRR, nDCG@k — replacing the synthetic-only P@1.
- **Runnable IR benchmark** (`scripts/benchmark_memory.py`) over the real
  ingest+retrieval stack with a **dense-only vs hybrid ablation** and a bundled
  LoCoMo-style sample (`docs/benchmarks/sample_dataset.json`); swap in real
  LoCoMo/LongMemEval via the same schema. On the sample, hybrid lifts recall@5
  by **+0.22** and MRR by **+0.34** over dense-only.

### OSS readiness & hygiene
- `CONTRIBUTING.md`, issue templates, and a PR template under `.github/`.
- CI now runs the full unit/integration/API suite with coverage reporting.
- Fixed a stale `reranker.py` docstring that called the (fully implemented)
  cross-encoder a "stub".
- Tests: `test_keys`, `test_scoped_auth`, `test_ir_metrics` added.
  Full suite: **163 passing**; `tsc --noEmit` clean.

## [0.3.0] — production-readiness hardening (cleanup pass)

### Branding & versioning
- Unified product branding around **stateful.ai** (engine name **stateful.ai**
  retained for the Python package/modules to avoid risky renames): API title,
  root response, OpenAPI metadata, MCP server name, and docs.
- **Single source of truth for version** (`core/version.py`, `__version__=0.3.0`);
  API `/`, `/health`, OpenAPI, and `pyproject.toml` now all agree (was 0.1.0 /
  0.2.0 drift).

### Backend production safety
- **No more silent fallback to in-memory stores in production.** Store fallback
  is now environment-aware: development/staging fall back with a warning;
  `APP_ENV=production` **raises** when a configured Postgres/Qdrant/Neo4j store
  is unavailable (`apps/api/dependencies.py`), so `/health/ready` reports `503`
  instead of pretending durability.
- **Explicit store selectors** `VECTOR_STORE` and `GRAPH_STORE` (memory|qdrant /
  memory|neo4j) — the zero-infra default no longer probes localhost:6333/7687.
- **Stable Qdrant point IDs**: replaced process-salted `hash()` (which orphaned
  vectors and broke delete/get after every restart) with a deterministic
  **UUIDv5** mapping (`point_id_for`), so upsert/delete/get are stable across
  restarts and machines.

### Config / implementation alignment
- **Implemented the OpenAI embedding backend** (`OpenAIEmbeddingBackend`, lazy
  client, dimension-aware) and **removed the unsupported `voyage`** option from
  settings/docs so advertised backends match reality. `.env.example`,
  `settings.py`, adapters, and README now agree.
- Removed stale generated `.next-*` references from `tsconfig.json`; `.gitignore`
  now covers all generated Next build dirs (`.next-*/`), `*.tsbuildinfo`, and
  `next-env.d.ts`.

### API / MCP consistency
- MCP `remember` docstring now matches the real `MemoryType` enum, and invalid
  memory types return a clear error listing valid types instead of silently
  coercing to `observation`.

### Tests
- Added `test_hardening` (deterministic Qdrant IDs, embedding factory, config
  validation/strictness, enum contract) and `test_metadata` (version/brand
  consistency, env-aware store fallback). Suite: **149 passing**;
  `tsc --noEmit` clean.

## [0.3.0] — Stateful-CL: continual learning

stateful.ai becomes a *self-improving* memory system. The memory corpus is reused
as a continual-learning replay buffer, and retrieval ranking now adapts online
from feedback — disabled by default, so the zero-infra path and all prior
behavior are unchanged. See `docs/continual_learning_design.md`.

### Added
- **`domain/learning/` package** (pure stdlib, no new deps):
  - `online_scorer.OnlineRankingPolicy` — per-namespace, convex (simplex)
    learned ranking weights over the five retrieval signals, updated online with
    an **EWC-style anchor** (accumulated Fisher diagonal + protected snapshot via
    `consolidate()`) to resist catastrophic forgetting. Per-namespace isolation
    gives multi-tenant personalization with zero cross-talk.
  - `replay.ReplayBuffer` — bounded, thread-safe store of served retrieval
    interactions and labeled (features, reward) examples; uniform `sample()` for
    experience replay; optional JSON persistence.
  - `feedback.shape_reward` — fuses explicit grade, coarse usefulness, downstream
    outcome, and a stale/contradicted-memory penalty into one reward in [0, 1].
  - `cl_metrics` — Average Accuracy, **Backward Transfer (BWT)**, Forward
    Transfer, and Forgetting over a task×checkpoint performance matrix.
  - `features.extract_features` — single source of truth for serving/training
    feature parity; `registry` — process-wide policy + buffer singletons.
- **`POST /api/v1/feedback`** — report whether a retrieved memory was useful;
  shapes a reward and updates the per-namespace ranking policy online.
- **`GET /api/v1/learning/stats`** — inspect replay buffer + per-namespace policy.
- **Retrieval feedback loop wiring**: `/retrieve` now returns a `query_id`;
  served candidates + features are logged to the replay buffer; the learned
  per-namespace weights drive the composite score when enabled.
- **Continual-eval harness** (`scripts/continual_eval.py`): task-incremental
  protocol with a static / no-EWC / EWC ablation, writing `BWT/FWT/forgetting`
  to `docs/benchmarks/continual_eval.{json,md}` and acting as a promotion gate.
  Result: EWC turns BWT ≈ **−0.64** (catastrophic forgetting) into **+0.20**
  while lifting average P@1 over the static baseline.
- **Telemetry** (`core/observability/metrics.py`): new Prometheus series on
  `/metrics` — `stateful_ai_feedback_total{recorded,outcome}`,
  `stateful_ai_feedback_reward` (histogram), `stateful_ai_cl_policy_updates_total`,
  `stateful_ai_cl_replay_interactions` / `_labeled_examples` /
  `stateful_ai_cl_policy_namespaces` (gauges), plus retrieval-result telemetry
  (`stateful_ai_retrieval_results_returned`, `_candidates_considered`,
  `stateful_ai_retrieval_empty_total`). Wired into the retrieve and feedback
  services; degrades to no-ops when `prometheus-client` is absent.
- **Tests**: `test_online_scorer`, `test_replay_buffer`, `test_cl_metrics`,
  `test_feedback_reward`, `test_telemetry`, and end-to-end `test_feedback`
  (API loop + disabled no-op path). Full suite: 122 passing.

- **PII redaction at ingest** (`domain/privacy/redaction.py`): config-gated
  (`PII_REDACTION_ENABLED`) detection + typed-placeholder redaction of emails,
  phone numbers, credit cards (Luhn-validated), SSNs, IPv4s, and provider
  secrets — applied before content is embedded, stored, or retrievable; reports
  per-category counts on memory metadata. Wired into `ingest_service`.
- **Zero-dependency Python SDK** (`sdk/`, stdlib `urllib`) covering the full
  lifecycle plus the feedback loop, and a **CLI** (`apps/cli.py`).
- **Elite upgrade dossier** (`docs/ELITE_UPGRADE.md`): senior-level diagnosis,
  gap analysis, research-grounded roadmap, target architecture, and self-critique.

### Settings
- `CONTINUAL_LEARNING_ENABLED` (default `false`), `CL_LEARNING_RATE`,
  `CL_EWC_LAMBDA`, `CL_REPLAY_CAPACITY`, `CL_REPLAY_PERSIST`,
  `CL_REWARD_SUCCESS_BONUS`, `CL_REWARD_CONTRADICTION_PENALTY`,
  `PII_REDACTION_ENABLED`.

## [0.2.0] — Production hardening & glass console

### Added
- **API-key authentication** for the FastAPI service (`apps/api/security.py`).
  Constant-time `X-API-Key` comparison, enabled by setting `STATEFUL_AI_API_KEY`;
  disabled by default so the zero-infra dev path is unchanged.
- **Per-client rate limiting.** In-process token bucket keyed by API key or
  client IP, configurable via `RATE_LIMIT_ENABLED`, `RATE_LIMIT_PER_MINUTE`,
  and `RATE_LIMIT_BURST`. Returns `429` with `Retry-After`.
- **Request body size limits** (`MAX_REQUEST_BYTES`, default 1 MiB → `413`).
- **Security headers** on every response (`X-Content-Type-Options`,
  `X-Frame-Options`, `Referrer-Policy`, COOP, `Permissions-Policy`) plus
  `Cache-Control: no-store` on API routes; applied to both FastAPI and Flask.
- **Consistent error envelope** `{"error": {code, message, request_id}}` for
  HTTP errors, validation errors, and unhandled exceptions ("detail" kept for
  backward compatibility).
- **`GET /api/v1/stats`** on both services: counts by type/status, namespaces,
  average importance, access totals, unresolved contradictions, top tags.
- **`GET /health/ready`** readiness probe that verifies the relational,
  vector, and graph stores actually respond (returns `503` when degraded).
- **Configurable CORS** via `CORS_ALLOW_ORIGINS`.
- **New glass-themed product UI.** `apps/landing_page.py` rebuilt as a
  glassmorphism landing page (`/`) and a live operations console (`/demo`)
  with KPI cards, ingest form, semantic recall with score bars, memory table
  with version history and soft-delete, and one-click JSON export.
- **Security test suite** (`tests/api/test_security.py`): auth, rate limiting,
  size limits, headers, and stats aggregation.

## [0.1.0] — Hybrid retrieval, agent-native MCP, zero-infra service

This release closes the gap between what the project documented and what it
actually ran, and adds the capabilities that make stateful.ai a credible,
state-of-the-art agent memory system.

### Added
- **Hybrid retrieval.** Dependency-free BM25 sparse search and Reciprocal Rank
  Fusion (`domain/memory/lexical.py`), wired into the retrieval pipeline
  (`services/retrieve_service.py`) and folded into the composite score
  (`domain/memory/scoring.py`). Recovers rare tokens, names, and identifiers
  that dense-only search misses.
- **Working cross-encoder reranker.** `domain/memory/reranker.py` replaces the
  previous stub with a lazily-loaded `ms-marco-MiniLM` cross-encoder that
  degrades gracefully to the heuristic reranker. Selectable via `RERANKER_TYPE`.
- **Zero-infrastructure mode.** `adapters/relational_store/memory_store.py` is
  an async, API-compatible in-memory relational store (with optional JSON
  persistence). The full FastAPI service now boots end-to-end with no database,
  queue, or vector service; production backends are selected purely by config.
- **MCP server.** `integrations/mcp_server.py` exposes memory to any Model
  Context Protocol client via `remember`, `recall`, `forget`, `list_memories`.
- **Observability.** Prometheus `/metrics` exporter (`core/observability/`),
  Prometheus scrape config, and Grafana wired into docker-compose under the
  `observability` profile.
- **Tests.** `tests/unit/test_lexical.py` (BM25 + RRF) and
  `tests/integration/test_hybrid_retrieval.py` (hybrid rescue + in-memory store).
- **CI.** A second job runs the FastAPI stack zero-infra, not just the Flask demo.
- **`.gitignore`** (previously absent) and a `CHANGELOG.md`.

### Fixed
- **Negative-similarity candidate drop.** Cosine similarity can be negative, but
  `RetrievalCandidate.semantic_score` required `>= 0.0`, so those candidates
  were silently discarded during enrichment. Scores are now clamped to `[0, 1]`.
- **`mock` embedding backend** rejected by settings validation despite being
  used throughout; added to the allowed values.
- **Structured logging** no longer hard-requires `python-json-logger`; falls
  back to plain formatting so the zero-infra path has no hidden dependency.
- **docker-compose** now sets `RELATIONAL_STORE=postgres` so the full stack
  actually uses Postgres instead of silently defaulting to in-memory.

### Changed
- **Dependencies reconciled.** `pyproject.toml` now declares a lean core that
  matches what the service imports, with honest optional extras (`postgres`,
  `qdrant`, `neo4j`, `queue`, `embeddings`, `llm`, `observability`, `agent`,
  `mcp`, `flask-demo`, `cli`). `requirements.txt` is the FastAPI core;
  `requirements-flask-demo.txt` holds the Flask demo set; the Dockerfile uses
  the latter.
- **Docs.** README and the architecture paper rewritten to describe the
  two-entry-point design (FastAPI product vs Flask demo), the hybrid pipeline,
  the MCP server, and observability — matching the running code.

### Security / hygiene
- `.env` should be untracked (`git rm --cached .env`); it is now gitignored.
