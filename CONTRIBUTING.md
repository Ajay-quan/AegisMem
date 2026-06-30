# Contributing to stateful.ai / stateful.ai

Thanks for your interest in contributing. This guide gets you productive fast.

## Development setup

Requires Python 3.11+ (the package targets 3.11) and Node 18+ for the frontend.

```bash
git clone https://github.com/Ajay-quan/stateful.ai.git
cd stateful.ai
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-asyncio httpx        # test deps
```

The service runs with **zero infrastructure** by default (in-memory stores +
deterministic mock embeddings):

```bash
uvicorn apps.api.main:app --reload     # http://127.0.0.1:8000/docs
```

## Running tests

```bash
# Full Python suite (no external infra needed):
pytest -q

# A focused area:
pytest tests/unit/test_online_scorer.py -q

# Frontend type-check:
npx tsc --noEmit
```

All PRs must keep the suite green. Add tests for new behavior — unit tests for
domain logic, API tests (httpx + ASGITransport) for endpoints.

## Project layout

```
core/        config, schemas, logging, observability, security, version
domain/      pure logic: memory scoring, lexical/BM25, reranker, learning, privacy, evaluations
services/    orchestration: ingest, retrieve, update, contradiction, reflect, feedback
adapters/    swappable backends: relational, vector, graph, embeddings, llm
apps/        FastAPI app + Flask demo + worker + CLI
integrations/ MCP server
sdk/         zero-dependency Python client
src/         Next.js frontend (stateful.ai marketing site)
```

Architecture: `API → services → domain → adapters`, every external backend
behind an adapter, defaults in-memory. See `docs/stateful_ai_architecture.md`,
`docs/continual_learning_design.md`, and `docs/ELITE_UPGRADE.md`.

## Conventions

- Keep the **zero-infra default** working: new external backends must be opt-in
  via settings and degrade gracefully (dev/staging) or fail loudly only in
  production (`APP_ENV=production`).
- Prefer narrow exceptions; log on graceful-degradation paths.
- No secrets in commits (`.env` is gitignored; use `.env.example`).
- Update `CHANGELOG.md` and relevant docs with user-facing changes.

## Pull requests

1. Branch from `main`.
2. Keep changes focused; one concern per PR.
3. `pytest -q` and `npx tsc --noEmit` must pass.
4. Fill in the PR template.

## Reporting bugs / requesting features

Use the issue templates under `.github/ISSUE_TEMPLATE`.
