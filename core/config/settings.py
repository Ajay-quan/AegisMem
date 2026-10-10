"""AegisMem core configuration using pydantic-settings."""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_env: Literal["development", "staging", "production"] = "development"
    app_secret_key: str = "dev-secret-key"
    log_level: str = "INFO"
    debug: bool = False

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "aegismem"
    postgres_user: str = "aegismem"
    postgres_password: str = "aegismem_password"

    @property
    def postgres_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def postgres_url_sync(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # Control plane (learning event log, receipts, ledger, API keys).
    # Empty -> Postgres in staging/production, a local SQLite file in development.
    control_database_url: str = ""
    dev_control_db_path: str = ".aegis/control.db"
    default_tenant: str = "default"
    # Refuse to boot in staging/production if Postgres RLS cannot be enforced for the app role.
    require_rls: bool = True

    # Observability
    metrics_enabled: bool = True
    otel_enabled: bool = False
    otel_exporter: Literal["otlp", "console"] = "otlp"
    otel_service_name: str = "aegismem-api"

    # Per-tenant rate limits (token bucket). "redis" is required for multi-replica correctness.
    rate_limit_backend: Literal["off", "memory", "redis"] = "memory"
    rate_limit_read_per_minute: int = 1200
    rate_limit_read_burst: int = 200
    rate_limit_write_per_minute: int = 600
    rate_limit_write_burst: int = 100

    # Vector-sync outbox worker (repairs index writes that failed inline)
    outbox_worker_enabled: bool = True
    outbox_interval_seconds: float = 5.0
    outbox_max_attempts: int = 8
    # Retention for operational records (pruned by the background maintenance loop).
    ledger_retention_hours: int = 168
    idempotency_retention_hours: int = 24

    # Auth. Disabled only in development; staging/production refuse to boot without it.
    auth_enabled: bool = False
    api_key_pepper: str = ""

    @property
    def resolved_control_database_url(self) -> str:
        if self.control_database_url:
            return self.control_database_url
        if self.app_env == "development":
            return f"sqlite+aiosqlite:///{self.dev_control_db_path}"
        return self.postgres_url

    @property
    def is_production_like(self) -> bool:
        return self.app_env in ("staging", "production")

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    # Storage topology. "pgvector" + graph "none" = single-store mode (one Postgres).
    vector_backend: Literal["qdrant", "pgvector", "memory"] = "qdrant"
    # "memory" = Docker-free local development (nothing persists). Rejected outside development.
    memory_backend: Literal["postgres", "memory"] = "postgres"
    graph_backend: Literal["neo4j", "none"] = "neo4j"

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "aegismem_memories"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "aegismem_password"

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embedding
    # "openai" = any OpenAI-compatible /embeddings API: OpenAI, Azure gateways, Voyage, vLLM,
    # LM Studio, Ollama (http://host:11434/v1). "mock" is development-only (meaningless vectors).
    embedding_backend: Literal["openai", "sentence_transformers", "mock"] = "mock"
    embedding_model: str = "text-embedding-3-small"
    embedding_api_base: str = "https://api.openai.com/v1"
    embedding_api_key: str = ""  # falls back to OPENAI_API_KEY
    # Single source of truth for vector size (index, migrations, embedding probe).
    embedding_dimension: int = 384

    @property
    def vector_dimension(self) -> int:
        return self.embedding_dimension

    # Default LLM
    default_llm_provider: Literal["openai", "anthropic", "local", "mock"] = "local"
    default_llm_model: str = "llama3.2"
    ollama_base_url: str = "http://localhost:11434"

    # Memory settings
    memory_importance_threshold: float = 0.3
    memory_max_working_tokens: int = 4096
    reflection_trigger_count: int = 10

    # Retrieval pipeline
    retrieval_top_n_candidates: int = 30
    retrieval_top_k: int = 5

    # Ranking weights (must sum to ~1.0 for interpretability)
    # Relevance dominates; the rest are tie-breakers (tuned on LoCoMo, see bench/locomo.py).
    weight_semantic: float = 0.85
    weight_recency: float = 0.05
    weight_importance: float = 0.05
    weight_utility: float = 0.05

    # Penalties
    contradiction_penalty_weight: float = 0.40

    # Integrity guard: memories whose (learned) trust falls below this are not served.
    guard_min_trust: float = 0.2
    contradiction_confidence_threshold: float = 0.7

    # Recency decay
    recency_decay_hours: float = 168.0  # 7 days

    # Diversity filtering
    diversity_threshold: float = 0.92  # suppress if cosine sim > this

    # Consolidation pipeline
    consolidation_access_threshold: int = 3        # promote after N accesses
    consolidation_importance_threshold: float = 0.6 # promote if importance >= this
    consolidation_similarity_threshold: float = 0.90 # merge if cosine sim > this

    # Memory-type scoring boosts (applied to composite score)
    type_boost_semantic: float = 0.05   # semantic memories get a small boost
    type_boost_preference: float = 0.03 # preference-type queries boost preference memories


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
