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

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Qdrant
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
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_backend: Literal["sentence_transformers", "openai", "voyage"] = (
        "sentence_transformers"
    )
    embedding_dimension: int = 1024

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
    weight_semantic: float = 0.40
    weight_recency: float = 0.25
    weight_importance: float = 0.25
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


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
