# ADR 0003: Optional API Key Authentication

## Status

Accepted

## Context

A public EC2 demo endpoint should not allow unrestricted writes if left online. Full OAuth or identity-provider integration would be excessive for a portfolio demo.

## Decision

Support optional static API-key auth with `STATEFUL_AI_API_KEY`. When configured, API routes require `X-API-Key`; `/` and `/health` remain public for quick checks.

## Consequences

- The demo gains a realistic security control at zero cost.
- Key rotation is manual.
- This is appropriate for a demo, not a complete production auth system.
