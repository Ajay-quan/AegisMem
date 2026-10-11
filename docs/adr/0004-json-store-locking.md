# ADR 0004: JSON Store With Advisory File Locking

> **Status: Superseded** by [ADR 0005](0005-merge-integrity-architecture.md) (2026-10-11). Kept for history.

## Status

Accepted

## Context

The Free Tier demo stores canonical records locally to avoid managed database cost. Multi-worker Gunicorn can otherwise cause concurrent JSON writes.

## Decision

Keep the JSON store for zero-cost portability, but guard mutations with an advisory Unix file lock and persist through atomic temp-file replacement. The deployment can also run a single Gunicorn worker if maximum simplicity is preferred.

## Consequences

- The demo is safer under concurrent requests than plain file writes.
- This is not a replacement for PostgreSQL in a high-write production workload.
- The architecture remains easy to inspect and tear down.
