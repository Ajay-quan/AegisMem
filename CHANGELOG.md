# Changelog

## [Unreleased] — Hybrid retrieval, agent-native MCP, zero-infra service

This release closes the gap between what the project documented and what it
actually ran, and adds the capabilities that make AegisMem a credible,
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
