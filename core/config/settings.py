"""stateful.ai core configuration using pydantic-settings."""
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

    # --- API security -------------------------------------------------
    # When ``api_key`` is empty, authentication is disabled (zero-config dev).
    # Set STATEFUL_AI_API_KEY in production to require the ``X-API-Key`` header
    # on all /api/* routes.
    api_key: str = Field(default="", validation_alias="STATEFUL_AI_API_KEY")

    # Scoped multi-tenant keys (revocable, named, optional per-tenant). Format:
    # comma- or newline-separated "name:secret" or "name:secret:tenant". Works
    # alongside the single STATEFUL_AI_API_KEY above (which maps to name "default").
    api_keys: str = Field(default="", validation_alias="STATEFUL_AI_API_KEYS")

    # Comma-separated list of allowed CORS origins ("*" for any).
    cors_allow_origins: str = "*"

    # Token-bucket rate limiting per client (API key if present, else IP).
    rate_limit_enabled: bool = False
    rate_limit_per_minute: int = 120
    rate_limit_burst: int = 30

    # Reject request bodies larger than this many bytes (default 1 MiB).
    max_request_bytes: int = 1_048_576

    # Storage backend selection. "memory" is the zero-infra default so the
    # full FastAPI service boots with no external database; the production
    # values opt into external services. When a non-memory backend is selected
    # but unreachable, behavior depends on app_env: development/staging fall
    # back to in-memory (with a warning); production raises so a misconfigured
    # deployment fails loudly instead of silently losing durability.
    relational_store: Literal["memory", "postgres"] = "memory"
    vector_store: Literal["memory", "qdrant"] = "memory"
    graph_store: Literal["memory", "neo4j"] = "memory"
    data_dir: str = "./data/stateful_ai"

    @property
    def strict_stores(self) -> bool:
        """In production, configured external stores must be reachable."""
        return self.app_env == "production"

    # PostgreSQL
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "stateful_ai"
    postgres_user: str = "stateful_ai"
    postgres_password: str = "stateful_ai_password"

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

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "stateful_ai_memories"

    # Neo4j
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "stateful_ai_password"

    # LLM Providers
    openai_api_key: str = ""
    anthropic_api_key: str = ""

    # Embedding
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    # "mock" is a deterministic, dependency-free backend used by the zero-infra
    # default and the test suite.
    embedding_backend: Literal["mock", "sentence_transformers", "openai"] = (
        "mock"
    )
    embedding_dimension: int = 1024
    # OpenAI embedding model used when embedding_backend="openai".
    openai_embedding_model: str = "text-embedding-3-small"

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

    # Hybrid retrieval (dense semantic + sparse lexical fused with RRF)
    hybrid_retrieval_enabled: bool = True
    lexical_candidate_pool: int = 200   # max memories pulled into the BM25 corpus
    rrf_k: int = 60                     # Reciprocal Rank Fusion constant
    bm25_k1: float = 1.5                # BM25 term-frequency saturation
    bm25_b: float = 0.75                # BM25 length normalization

    # Reranking
    reranker_type: Literal["heuristic", "cross_encoder"] = "heuristic"
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # Ranking weights (kept ~1.0 in aggregate for interpretability)
    weight_semantic: float = 0.35
    weight_lexical: float = 0.15
    weight_recency: float = 0.20
    weight_importance: float = 0.20
    weight_access: float = 0.10

    # Penalties
    contradiction_penalty_weight: float = 0.40
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

    # --- Continual learning (Stateful-CL) --------------------------------
    # Master switch. When False (default) retrieval uses the static
    # ``weight_*`` values above and nothing is logged/learned, so the
    # zero-infra behavior and existing tests are unchanged. When True, the
    # per-namespace online ranking policy supplies learned weights, retrievals
    # are logged to the replay buffer, and /feedback drives online updates.
    continual_learning_enabled: bool = False
    cl_learning_rate: float = 0.05       # online SGD step size for the policy
    cl_ewc_lambda: float = 0.1           # EWC anchor strength (anti-forgetting)
    cl_replay_capacity: int = 5000       # max logged interactions / labeled examples
    cl_replay_persist: bool = False      # persist replay buffer to data_dir as JSON
    cl_reward_success_bonus: float = 0.2 # outcome=success/failure nudge on reward
    cl_reward_contradiction_penalty: float = 0.3  # penalty for stale/contradicted hits

    # --- Privacy --------------------------------------------------------
    # Redact PII (emails, phones, cards, SSNs, IPs, provider secrets) at the
    # ingest boundary so sensitive tokens are never embedded, stored, or
    # retrieved. Off by default; enable for deployments handling real user data.
    pii_redaction_enabled: bool = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
