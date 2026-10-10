"""AegisMem custom exceptions."""


class AegisMemError(Exception):
    """Base exception for AegisMem."""


class MemoryNotFoundError(AegisMemError):
    def __init__(self, memory_id: str) -> None:
        super().__init__(f"Memory not found: {memory_id}")
        self.memory_id = memory_id


class MemoryStorageError(AegisMemError):
    """Error in storage backend."""


class MemoryRetrievalError(AegisMemError):
    """Error during retrieval."""


class EmbeddingError(AegisMemError):
    """Error generating embeddings."""


class LLMError(AegisMemError):
    """Error from LLM provider."""


class ContradictionError(AegisMemError):
    """Error during contradiction detection."""


class ConfigurationError(AegisMemError):
    """Invalid configuration."""


class NamespaceError(AegisMemError):
    """Invalid or inaccessible namespace."""


class ValidationError(AegisMemError):
    """Schema validation failure."""
