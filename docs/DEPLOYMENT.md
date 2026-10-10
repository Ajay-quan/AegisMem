# Deploying AegisMem

## Topologies

| Mode | Stores | Use |
|---|---|---|
| **Single-store** (recommended) | Postgres 16 + pgvector, Redis | Design partners, most production installs |
| Multi-store | Postgres, Qdrant, Neo4j, Redis | Very large vector corpora or graph features |
| Dev (no Docker) | in-process | `MEMORY_BACKEND=memory VECTOR_BACKEND=memory GRAPH_BACKEND=none uvicorn apps.api.main:app` |

## Security model (read this first)

1. **Two database roles.** `aegis_owner` owns the schema and runs migrations. `aegis_app` is what the
   API connects as and must be `NOSUPERUSER NOBYPASSRLS`. Postgres row-level security (migration 0005)
   is ignored for superusers and BYPASSRLS roles, so the API **refuses to boot** in staging/production if
   its role could bypass RLS (`REQUIRE_RLS=true`).
2. **Secrets.** `APP_SECRET_KEY` (receipt signing, risk-feature hashing) and `API_KEY_PEPPER` (API-key
   hashing) must be distinct, random and at least 48 characters. Rotating the pepper invalidates all API keys;
   rotating the secret key changes receipt signatures for *new* receipts only.
3. **Auth is mandatory** outside development. Every route has an explicit scope; unmapped routes are denied.
4. **Fail closed.** In staging/production the API will not start without its durable stores, and never
   falls back to in-memory storage.

## Docker Compose (evaluation / single node)

```bash
cp infra/compose/.env.example infra/compose/.env      # fill every secret
docker compose -f infra/compose/docker-compose.yml --env-file infra/compose/.env up -d
docker compose -f infra/compose/docker-compose.yml exec api \
  python scripts/aegisctl.py create-key --tenant acme --name bootstrap --scopes admin
```

The `migrate` service runs `alembic upgrade head` as `aegis_owner` before the API starts.
Add `--profile observability` for Prometheus (`:9090`) and Grafana (`:3000`).

## Kubernetes

```bash
kubectl create namespace aegismem
# Create the Secret from your secret manager (see infra/k8s/base/secret.example.yaml)
kubectl apply -k infra/k8s/overlays/production          # config, job, deployment, hpa, pdb, netpol
kubectl -n aegismem wait --for=condition=complete job/aegismem-migrate --timeout=300s
```

In CD, run the migration Job to completion **before** rolling the Deployment, and pin the image by digest.
Pods run as non-root (uid 10001) with a read-only root filesystem, dropped capabilities, `/ready`
readiness (checks Postgres and the control plane), `/health` liveness, and a PDB plus zone spreading.

## Operations

| Task | Command / endpoint |
|---|---|
| Create / list / revoke API keys | `scripts/aegisctl.py create-key|list-keys|revoke-key`, or `/api/v1/admin/keys` |
| Vector index repair queue | `aegisctl outbox-status`, `aegisctl drain-outbox` (the API also drains it every 5s) |
| Rebuild a learned model from its event log | `aegisctl rebuild --tenant T --component trust|risk` |
| Metrics | `GET /metrics` (expose only to the monitoring namespace; see NetworkPolicy) |
| Tracing | `OTEL_ENABLED=true` + standard `OTEL_EXPORTER_OTLP_*` env; requires the `otel` extra |
| Integrity regression check | `python -m bench.run`; `aegis-audit --target aegismem --url ...` |

### Key alerts
- `aegis_outbox_ops_total{result="dead"}` increasing: index repairs failing permanently.
- `aegis_erasures_total{verified="false"}` > 0: an erasure could not be verified in every store.
- `aegis_auth_failures_total{reason="invalid_or_revoked"}` spike: leaked or brute-forced keys.
- `aegis_rate_limited_total` sustained: tenant needs a higher limit or is misbehaving.
- `/ready` returning 503: a durable dependency is down (traffic is drained automatically).

### Upgrading existing data
Deployments that predate tenancy must backfill `tenant_id` into Qdrant payloads (pgvector rows are
written with it). Qdrant point ids changed from salted `hash()` to `uuid5`, so re-index Qdrant
collections created before migration 0002.
