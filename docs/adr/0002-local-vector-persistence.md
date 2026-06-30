# ADR 0002: Local FAISS First, Optional Chroma

## Status

Accepted

## Context

The project needs semantic retrieval with a vector database while staying easy to run locally and on EC2 Free Tier. Managed vector search services would add cost and operational complexity.

## Decision

Use FAISS as the default local vector index and persist index metadata under the mounted data directory. Provide an optional local ChromaDB adapter through `STATEFUL_AI_VECTOR_STORE=chroma` for users who want Chroma persistence without provisioning a managed database.

## Consequences

- FAISS works well for a single-node demo.
- Chroma strengthens the FAISS/ChromaDB resume claim without adding paid infrastructure.
- The production path still needs stronger concurrency and backup design before real multi-user use.
