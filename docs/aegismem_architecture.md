# AegisMem Architecture Paper

## Abstract

AegisMem is a persistent memory layer for long-running LLM agents. It separates memory ingestion, lifecycle management, semantic retrieval, exact lookup, and graph traversal into service and adapter boundaries. For cost control, the demonstration deployment runs as one Dockerized Flask service on AWS Free Tier EC2 while keeping the internal structure compatible with a future multi-service deployment.

## Problem

Long-running LLM agents need more than chat history. They need to store observations, retrieve relevant prior context, update stale facts, delete incorrect memories, and connect related events. A basic transcript search path is weak because it cannot combine semantic relevance, exact identifiers, recency, importance, and graph relationships.

## Design Goals

- Provide a REST API for full memory lifecycle operations.
- Support semantic retrieval through embeddings and a vector index.
- Support exact key retrieval through a hash-indexed path.
- Support related-memory traversal through a graph algorithm.
- Keep the deployment free by using local persistence on a single EC2 Free Tier instance.
- Keep code boundaries clear enough to split into services later.

## System Architecture

The Flask API is the public boundary. It delegates to a memory service that coordinates four local components:

- `JsonMemoryStore`: canonical records, lifecycle state, and SHA-256 key index.
- `FaissVectorStore`: vector database for semantic similarity search.
- `LocalMemoryGraph`: persistent graph with weighted BFS traversal.
- `HotMemoryIndex`: latency-oriented cache index using a priority queue, hash map, and sorted recency tree.

The production-style container runs Gunicorn and mounts `/data/aegismem` to an EBS-backed host directory. The FAISS index, graph file, and JSON store survive container restarts.

## Retrieval Pipeline

1. Ingestion receives `content`, `user_id`, optional `key`, metadata, tags, and related memory IDs.
2. The embedding adapter converts content to a vector using the LangChain `Embeddings` interface.
3. The memory record is stored in JSON and indexed by SHA-256 hash key.
4. The vector is upserted into FAISS with metadata filters.
5. Related IDs are connected in the local graph.
6. Retrieval embeds the query, searches FAISS, filters by user/status, and hydrates full records from the JSON store.

## Primary Service (FastAPI) and Hybrid Retrieval

The repository has two entry points. The **Flask demo** (described above) is a single-file, zero-cost deployment for AWS Free Tier using local FAISS + JSON. The **FastAPI service** (`apps/api/`) is the primary product: a layered system (API → services → domain → adapters) with pluggable backends for the relational store (in-memory or PostgreSQL), vector store (in-memory or Qdrant), and graph store (in-memory or Neo4j). It boots end-to-end with no external infrastructure by defaulting to in-memory stores and deterministic mock embeddings; production backends are selected purely through configuration (`RELATIONAL_STORE`, `EMBEDDING_BACKEND`, etc.).

Retrieval in the FastAPI service is a **hybrid** pipeline rather than dense-only:

1. Dense semantic search over-retrieves a broad candidate pool from the vector store.
2. Sparse **BM25** search (`domain/memory/lexical.py`, pure Python) ranks the user's memory corpus on lexical overlap, recovering rare tokens, names, and identifiers that dense embeddings under-weight.
3. The two ranked lists are merged with **Reciprocal Rank Fusion** (`k=60`), which is robust to the fact that BM25 scores and cosine similarities are not directly comparable.
4. Each fused candidate is scored on a weighted composite of semantic, lexical, recency (exponential decay), importance, and access-frequency signals. Similarity scores are clamped to `[0, 1]` so negative cosine values never silently drop candidates.
5. A second-stage reranker applies diversity filtering and returns the top-k. The default is a heuristic reranker; a cross-encoder reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) is available via configuration and loads lazily with graceful fallback.

The service is also exposed over the **Model Context Protocol** (`integrations/mcp_server.py`) so any MCP-capable agent can use AegisMem's memory directly through `remember`, `recall`, `forget`, and `list_memories` tools.

## Algorithms and Data Structures

Hash-indexed exact lookup stores `sha256(user_id:key) -> memory_id`, which provides constant-time lookup for named memories.

Graph retrieval uses weighted breadth-first traversal. Each edge carries a relation and weight. Traversal returns reachable memories ordered by distance and path score.

The hot-memory index uses:

- `heapq` priority queue for low-priority eviction.
- Python dict hash map for direct memory ID access.
- Sorted recency tree implemented with `bisect.insort` for recent-memory scans.

## AWS Free Tier Deployment

The AWS deployment intentionally avoids managed services. One EC2 Free Tier instance runs Docker. The container exposes Gunicorn on port 8000, mapped to host port 80. An 8 GB gp3 EBS root volume stores `/opt/aegismem/data`, mounted into the container at `/data/aegismem`.

This validates cloud deployability without adding RDS, OpenSearch, EFS, S3, API Gateway, NAT Gateway, load balancers, Route 53, or other cost-bearing resources.

## Evaluation

The benchmark in `scripts/evaluate_memory_retrieval.py` seeds synthetic memories and measures Precision@1, Precision@3, Recall@5, MRR, average latency, and p95 latency. It writes machine-readable metrics and SVG charts under `docs/benchmarks` and `docs/assets`.

## Limitations

The Free Tier deployment is a single-node demonstration, not a horizontally distributed production cluster. The local embedding fallback is deterministic and free, but a real deployment can switch to sentence-transformers for stronger semantic quality. The current benchmark is synthetic; a stronger paper would add real agent traces and human-labeled relevance judgments.

## Implemented since first draft

- Hybrid retrieval: BM25 sparse search fused with dense search via Reciprocal Rank Fusion.
- Working cross-encoder reranker (lazy-loaded, with heuristic fallback).
- Zero-infra in-memory relational store so the full FastAPI service runs with no database.
- MCP server exposing memory as agent tools.
- Prometheus `/metrics` exporter and structured JSON logging.
- Optional Chroma persistent adapter; API-key auth; import/export snapshots.

## Future Work

- Cognitive layer depth: forgetting/salience model, episodic→semantic consolidation at scale, and a temporal knowledge graph for relationship reasoning.
- Real-corpus evaluation on standard memory benchmarks (LoCoMo, LongMemEval) with dense-vs-hybrid-vs-reranker ablations, replacing the synthetic benchmark.
- Per-tenant authn/z (scoped keys / JWT), rate limiting, and PII redaction at ingest.
- Multi-process consistency tests for concurrent writes; rate limiting; larger noisy benchmarks.

## Security and Portability Additions

The Flask demo supports optional API-key authentication through `AEGISMEM_API_KEY`. When configured, all API routes require `X-API-Key`; `/`, `/demo`, and `/health` stay public for status and local UI access.

Memory updates append prior record states to version history instead of silently overwriting content. Import/export endpoints allow a complete memory snapshot to be saved, restored, or moved between local and EC2 demo environments without using S3 or a managed database.

## Optional ChromaDB Mode

FAISS is the default vector store because it is lightweight and file-based. A local ChromaDB adapter is also available with `AEGISMEM_VECTOR_STORE=chroma`, using Chroma's persistent local client in the same EBS-mounted data directory. This keeps the FAISS/ChromaDB architecture claim defensible without provisioning a separate vector database service.
