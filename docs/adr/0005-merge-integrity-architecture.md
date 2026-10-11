# ADR 0005: One product, integrity-first architecture (merge of `main` v0.3.0 and `q1/production-core`)

> Status: Accepted (2026-10-11). Supersedes ADRs 0001–0004.

## Context
Two lines of development diverged from `89fd9c1`. `main` shipped v0.2.0–v0.3.0 ("stateful.ai"): a Flask
single-node demo on FAISS/JSON, hybrid BM25 retrieval, single-key auth, PII redaction, an EWC learning loop,
and a landing site. `q1/production-core` built the multi-tenant integrity layer: scoped keys, Postgres RLS,
event-sourced learning, guard/quarantine/erasure, pgvector, outbox, SDKs, MCP, Helm, and benchmarks verified
on real infrastructure.

## Decision
- Product name **AegisMem**, version **0.4.0**. One API (FastAPI); the Flask demo stack is retired.
- **Kept from main and integrated:** BM25 + Reciprocal Rank Fusion as a candidate stage *before* the integrity
  guard (tuned on LoCoMo: blend 0.15, corpus pool 2,000; main's 200-memory pool cost 10 pts recall@5),
  cross-encoder reranker option, PII redaction at ingest (before screening and embedding), security headers,
  request-size limits, error envelope, `/stats`, retrieval/feedback telemetry, zero-infra development defaults,
  EWC policy + replay + BWT/FWT modules (evaluation harness), CLI (rewired to the SDK), landing site.
- **Superseded by Q1 equivalents:** single-key registry → scoped durable API keys; in-process audit ring
  buffer → durable tenant-scoped `audit_events`; `query_id` feedback → `retrieval_id` feedback with atomic
  ledger; stdlib SDK → `aegismem-client`; main's MCP server → `aegismem-mcp`; FAISS (sync, pickle metadata)
  → pgvector; JSON store → Postgres.
- Production refuses in-memory stores and mock embeddings; every deployment declares backends explicitly.

## Consequences
- Recall improved: LoCoMo recall@5 0.565 vs 0.514 raw dense (+5.1 pts), 0/5,882 benign turns quarantined.
- One code path to secure, test and operate. Retired code remains in history at `e20ba0b`.
- Open (Q2): head-to-head of main's EWC policy vs. the durable ranking learner; Postgres full-text index
  for very large users instead of per-query BM25.
