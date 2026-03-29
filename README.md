# AegisMem

**Persistent Memory Architecture for Long-Running LLM Agents**

AegisMem is a production-grade, open-source memory system for AI agents that solves long-term retention, context management, and memory quality at scale.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-green.svg)](https://fastapi.tiangolo.com)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

## Why AegisMem?

Most agent systems today treat memory as an afterthought:

| Problem | AegisMem Solution |
|---------|-------------------|
| Memory is just raw chat history | Structured memory types: observations, facts, episodes, reflections |
| Retrieval is only embedding similarity | Hybrid retrieval: semantic + recency + importance + symbolic filters |
| Updates are append-only and ungoverned | Versioned, LLM-governed updates with full audit trail |
| Contradictions are never resolved | Automatic contradiction detection and resolution workflow |
| No higher-level synthesis | Reflection generation: patterns, preferences, behavioral insights |
| Memory is never benchmarked | Built-in evaluation harness with retrieval and contradiction metrics |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI REST API                        │
│          /ingest  /retrieve  /reflect  /contradictions       │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    Service Layer                              │
│  IngestionService  RetrievalService  ReflectionService       │
│  UpdateService     ContradictionService  EvaluationRunner    │
└──────┬────────────────┬────────────────────┬────────────────┘
       │                │                    │
┌──────▼──────┐ ┌───────▼───────┐ ┌─────────▼──────┐
│  PostgreSQL  │ │    Qdrant     │ │     Neo4j      │
│  (canonical  │ │  (semantic    │ │    (entity     │
│   records,   │ │   retrieval,  │ │   graph,       │
│   versions,  │ │   ANN search) │ │   relations,   │
│   audit log) │ └───────────────┘ │   contradicts) │
└──────────────┘                   └────────────────┘
       │                                    │
┌──────▼────────────────────────────────────────────┐
│  Redis (cache, session state, Celery broker)       │
└────────────────────────────────────────────────────┘
```

### Multi-Store Design

AegisMem uses four complementary stores — no single database represents the entire memory system:

- **PostgreSQL** — canonical records, versioned history, audit logs, facts, evaluation results
- **Qdrant** — vector index for semantic nearest-neighbor retrieval
- **Neo4j** — entity-relationship graph for multi-hop reasoning, contradiction edges
- **Redis** — session working memory, retrieval cache, Celery task queue

---

## Memory Types

| Type | Description | Example |
|------|-------------|---------|
| `observation` | Raw input from conversation or tool | "User asked about internship opportunities" |
| `fact` | Discrete, verifiable structured fact | `user -> lives_in -> San Francisco` |
| `episode` | Bounded event with temporal context | "Meeting on March 3rd resulted in..." |
| `procedure` | Learned workflow or process | "To deploy: run tests, tag release, push" |
| `reflection` | Higher-level synthesized insight | "User consistently prioritizes AI career paths" |
| `summary` | Compressed summary of memory cluster | "Project history through Q2 2025" |
| `working` | Transient in-session context | Current plan, recent tool outputs |

---

## Memory Lifecycle

```
observe → score → store → embed → index → retrieve → rank
                              ↓
                         contradiction scan
                              ↓
                         versioned update (create|update|merge|supersede|skip)
                              ↓
                         reflection generation
                              ↓
                         summarize / archive / expire
```

---

## Hybrid Retrieval

Retrieval score combines multiple signals — not just embeddings:

```
composite_score = 0.40 × semantic_similarity
                + 0.25 × recency_score        (exponential decay)
                + 0.25 × importance_score      (heuristic + LLM)
                + 0.10 × access_frequency
                - contradiction_penalty        (if flagged)
```

---

## Quickstart

### Local (no external services)

```bash
git clone https://github.com/your-org/aegismem
cd aegismem
pip install -e ".[dev]"

# Run the demo (mock backends - no API keys needed)
python examples/simple_chat_memory/demo.py

# Run the benchmark
python scripts/run_local_eval.py
```

### With Docker (full stack)

```bash
cp .env.example .env
# Add OPENAI_API_KEY or ANTHROPIC_API_KEY to .env

docker compose -f infra/compose/docker-compose.yml up
# API available at http://localhost:8000
# Docs at http://localhost:8000/docs
```

### With observability (Prometheus + Grafana)

```bash
docker compose -f infra/compose/docker-compose.yml --profile observability up
# Grafana at http://localhost:3000 (admin/admin)
# Prometheus at http://localhost:9090
```

---

## API

```bash
# Ingest a memory
curl -X POST http://localhost:8000/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "User prefers async Python APIs", "user_id": "alice"}'

# Retrieve relevant memories
curl -X POST http://localhost:8000/api/v1/retrieve \
  -H "Content-Type: application/json" \
  -d '{"query": "What does the user prefer?", "user_id": "alice", "top_k": 5}'

# Smart update (versioned, LLM-governed)
curl -X POST http://localhost:8000/api/v1/update \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice", "new_content": "User moved from SF to Austin"}'

# Generate reflections
curl -X POST http://localhost:8000/api/v1/reflect \
  -H "Content-Type: application/json" \
  -d '{"user_id": "alice"}'

# Run evaluation suite
curl -X POST http://localhost:8000/api/v1/eval/run
```

Full API docs: `http://localhost:8000/docs`

---

## Evaluation

AegisMem ships with a built-in evaluation harness:

```bash
# Run full benchmark
python scripts/run_local_eval.py
```

**Benchmark results (mock LLM baseline):**

| Metric | Score |
|--------|-------|
| Precision@5 | 0.60 |
| MRR | 0.33 |
| Contradiction F1 | 0.86 |
| Avg Retrieval Latency | <1ms (mock) |

---

## Integration with Local LLMs (Ollama)

AegisMem is built with out-of-the-box support for [Ollama](https://ollama.com/), allowing you to run powerful LLMs completely locally, securing your agent memories without any cloud API keys.

### 1. Install and Pull the Model
Ensure you have Ollama installed on your system. Pull the model you intend to use (AegisMem defaults to `llama3.2`):
```bash
ollama run llama3.2
```
Once the model is spun up locally, you can exit the CLI prompt (`/bye`) and the daemon will remain active.

### 2. Configure AegisMem
By default, AegisMem's Docker stack is rigged to securely bridge internal networking to your host machine's Ollama instance natively via `host.docker.internal` port configurations.

If running manually (without Docker), update your `.env` to match your local port:
```env
DEFAULT_LLM_PROVIDER=local
DEFAULT_LLM_MODEL=llama3.2
OLLAMA_BASE_URL=http://localhost:11434
```

---

## LLM Providers

AegisMem is fully provider-agnostic. The internal factory can flexibly spool the provider you need at runtime:

```python
# Ollama (Local)
client = OllamaClient(base_url="http://localhost:11434", model="llama3.2")

# OpenAI
client = OpenAIClient(api_key="sk-...", model="gpt-4o-mini")

# Anthropic
client = AnthropicClient(api_key="sk-ant-...", model="claude-3-haiku")

# Mock (testing, no API key)
client = MockLLMClient()
```

---

## Repository Structure

```
aegismem/
├── apps/api/           # FastAPI application + routers
├── core/               # Config, schemas, logging, exceptions
├── domain/             # Memory scoring, evaluation
├── adapters/           # LLM, embedding, vector, graph, relational adapters
├── services/           # Ingest, retrieve, update, reflect, contradiction
├── pipelines/          # Async pipeline definitions
├── datasets/           # Synthetic & benchmark datasets
├── tests/              # Unit, integration, API tests
├── scripts/            # CLI tools
├── examples/           # Runnable demos
├── infra/              # Docker, Alembic migrations
└── docs/               # Documentation
```

---

## Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific suite
pytest tests/unit/
pytest tests/integration/
pytest tests/api/
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.11+ |
| API | FastAPI + Uvicorn |
| Validation | Pydantic v2 |
| ORM | SQLAlchemy 2.x + Alembic |
| Relational Store | PostgreSQL |
| Vector Store | Qdrant (+ in-memory fallback) |
| Graph Store | Neo4j (+ mock fallback) |
| Cache / Queue | Redis + Celery |
| Embeddings | sentence-transformers / OpenAI |
| LLM Providers | OpenAI, Anthropic, mock |
| Agent Orchestration | LangGraph |
| Observability | OpenTelemetry, Prometheus, Grafana |
| Testing | pytest + pytest-asyncio |
| Deployment | Docker + Compose |

---

## License

Apache 2.0 — see [LICENSE](LICENSE)
