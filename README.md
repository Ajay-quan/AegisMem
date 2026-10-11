# AegisMem

**The trust layer for AI agent memory.**

Agents are getting long-term memory faster than companies can trust it. AegisMem makes sure that what an
agent remembers is **true, current, safe and provably erasable**, and it gets better with every human decision.

| Problem in typical agent memory | AegisMem |
|---|---|
| A prompt injection in a web page or tool result becomes a "fact" | Provenance + trust on every write; low-trust instruction-like content is **quarantined** |
| Updated or revoked facts keep being served | A **read-time guard** re-checks every result against the canonical record and swaps in the current version |
| "Delete" leaves copies, summaries and vectors behind | **Cascading erasure** across versions, derivatives, duplicates and every store, with a **signed receipt** |
| Screening rules never improve | **Continual learning** from reviewer decisions and corrections: calibrated per-origin trust, a self-improving risk model, learned ranking |
| Context is lost between agents and sessions | **Handoff packets**: token-budgeted, cited briefings with open questions and recent changes |
| Vector search misses names, ids and rare terms | **Hybrid retrieval**: dense + BM25 fused with Reciprocal Rank Fusion, then the integrity guard |
| Agents persist whatever they see, including PII | **PII redaction at ingest** (emails, phones, cards, SSNs, IPs, secrets) before embedding or storage |

Run the free [integrity audit](audit/) against any memory system to see where it stands. A typical naive
vector memory scores 1/7; AegisMem scores 7/7.

## Quickstart (no Docker)

```bash
pip install -e ".[dev]"
uvicorn apps.api.main:app        # zero infrastructure: in-memory stores and mock embeddings by default
```

```python
from aegismem_client import AegisMem          # pip install ./sdk/python

async with AegisMem("http://localhost:8000") as mem:
    await mem.ingest("User prefers aisle seats", user_id="u1")
    w = await mem.ingest(page_text, user_id="u1", source_type="tool_output", origin="domain:example.com")
    if w.quarantined: ...                     # held for review, never served as trusted memory
    r = await mem.retrieve("seat preference", user_id="u1")
    await mem.feedback(r.retrieval_id, user_id="u1", used=[r.results[0].memory_id])
    brief = await mem.handoff(user_id="u1", task="book a flight", token_budget=800)
    receipt = await mem.forget("u1", memory_ids=[...], reason="GDPR Art. 17 request")
```

Also available: a [TypeScript SDK](sdk/typescript), an [MCP server](integrations/mcp) for any MCP client,
[LangChain/LangGraph and OpenAI Agents tools](integrations/frameworks), a CLI (`python -m apps.cli`),
and production deployment with [Docker Compose or Kubernetes](docs/DEPLOYMENT.md).

## How it works

```
write ──► provenance + learned trust ──► rule + risk-model screening ──► store │ quarantine
                                                                                   │
read  ──► hybrid retrieval ──► integrity guard (current version? expired? quarantined?
                               erased? trusted enough? right tenant?) ──► learned ranking ──► agent
                                                                                   │
human outcomes (approve / reject / correct / confirm / used) ──► event log ──► trust, risk & ranking models
```

- **Event-sourced learning.** Every learned update is an append-only event; models are deterministic replays,
  consistent across replicas and rebuildable for audit. Risk-model events store keyed feature hashes, never text.
- **Multi-tenant by construction.** Scoped stores in the app, Postgres row-level security in the database,
  scoped API keys, per-tenant rate limits.
- **Operable.** Idempotent writes, a durable index-repair outbox, `/ready` probes, Prometheus metrics,
  OpenTelemetry traces (no content in telemetry), Alembic migrations.

## Measured, not claimed

`python -m bench.run` runs the gated benchmark suite (it also runs in CI):

| Suite | Current result |
|---|---|
| Poisoning (held-out, cold start) | recall 0.58 vs 0.125 rules-only; false-positive rate 0.11; 0 false positives on users' own words |
| Learning from reviewers (8 splits) | held-out recall 0.57 → 0.78 with FPR +0.02 |
| Revocation (7 scenarios) | 0 revoked memories served; 100% replaced with the current version |
| Erasure (20 subjects) | 0 residue, 100% cascade recall, 0 collateral, receipts tamper-evident |
| **Recall, LoCoMo** (real embeddings, 1,536 questions) | recall@5 **0.565** vs 0.514 raw vector search (+5.1 pts); 0 / 5,882 benign turns quarantined |
| External injections (deepset, EN+DE) | recall 0.12 cold → 0.83 after reviewer learning, FPR 0.09 |
| Overhead | screening p95 ≈ 0.05 ms, guard p95 ≈ 0.05 ms (in-memory backends) |

Details and reproduction commands: [bench/RESULTS.md](bench/RESULTS.md).

## Repository map

| Path | What |
|---|---|
| `apps/api` | FastAPI app, auth, idempotency, rate limits |
| `services/` | ingest, retrieve, guard, erasure, outcomes, handoff, feedback, learning runtime, vector sync |
| `domain/` | pure logic: provenance, receipts, scoring, trust calibrator, risk model, weight learner |
| `adapters/` | Postgres, pgvector, Qdrant, Neo4j, in-memory stores, tenancy wrappers |
| `bench/`, `audit/` | benchmark suite with gates; open-source integrity audit |
| `sdk/`, `integrations/mcp` | Python and TypeScript clients; MCP server |
| `infra/` | migrations, Dockerfile, compose, Kubernetes manifests |
| `ROADMAP.md`, `PRODUCT_PLAN_12M.md`, `project_log.md` | strategy, plan, engineering log |

## License

Apache-2.0
