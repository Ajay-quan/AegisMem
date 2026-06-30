"""stateful.ai MCP server — expose agent memory as Model Context Protocol tools.

This turns stateful.ai from a REST service into something any MCP-capable agent
(Claude Desktop, Cursor, LangGraph, custom runtimes) can plug into directly,
giving the agent durable, cross-session memory through four tools:

    remember(content, ...)   -> store a memory
    recall(query, ...)       -> hybrid (dense + BM25) retrieval
    forget(memory_id)        -> soft-delete a memory
    list_memories(user_id)   -> browse stored memories

It reuses the exact same services as the HTTP API and defaults to the
zero-infra in-memory store, so ``python -m integrations.mcp_server`` runs with
no external database, queue, or vector service.

Run (stdio transport, the default for desktop agents):

    pip install "mcp[cli]"
    python -m integrations.mcp_server

Register with Claude Desktop by adding to ``claude_desktop_config.json``:

    {
      "mcpServers": {
        "stateful_ai": {"command": "python", "args": ["-m", "integrations.mcp_server"]}
      }
    }
"""
from __future__ import annotations

import asyncio
import logging

from core.logging.logger import setup_logging
from core.config.settings import settings
from core.schemas.memory import RetrievalQuery, MemoryType

logger = logging.getLogger(__name__)

try:
    from mcp.server.fastmcp import FastMCP
except Exception as exc:  # pragma: no cover - dependency guard
    raise SystemExit(
        "The MCP SDK is required to run the stateful.ai MCP server.\n"
        "Install it with:  pip install 'mcp[cli]'\n"
        f"(import error: {exc})"
    )


mcp = FastMCP("stateful.ai")

# Valid memory types — mirrors core.schemas.memory.MemoryType exactly.
VALID_MEMORY_TYPES = [t.value for t in MemoryType]

# Lazily-initialized singletons so tool calls share one store/index instance.
_services: dict[str, object] = {}
_init_lock = asyncio.Lock()


async def _ensure_services() -> None:
    if _services:
        return
    async with _init_lock:
        if _services:
            return
        from apps.api.dependencies import (
            get_db_store, get_vector_store, get_graph_store,
            get_ingest_service, get_retrieve_service,
        )
        db = await get_db_store()
        vs = await get_vector_store()
        graph = await get_graph_store()
        _services["db"] = db
        _services["ingest"] = await get_ingest_service(db, vs, graph)
        _services["retrieve"] = await get_retrieve_service(db, vs)
        logger.info("stateful.ai MCP services initialized (store=%s)", type(db).__name__)


@mcp.tool()
async def remember(
    content: str,
    user_id: str = "default",
    memory_type: str = "observation",
    importance: float | None = None,
) -> dict:
    """Store a new memory for an agent/user.

    Args:
        content: The text to remember (a fact, observation, episode, or note).
        user_id: Owner of the memory; isolates memories per user/agent.
        memory_type: One of observation, fact, episode, procedure, reflection,
            working, summary. Defaults to observation.
        importance: Optional 0..1 importance override; auto-scored if omitted.

    Returns the stored memory id and content, or an error if memory_type is
    invalid (so the caller can correct it rather than silently mis-typing).
    """
    await _ensure_services()
    try:
        mtype = MemoryType(memory_type)
    except ValueError:
        return {
            "error": f"invalid memory_type '{memory_type}'",
            "valid_memory_types": VALID_MEMORY_TYPES,
        }
    memory = await _services["ingest"].ingest_text(  # type: ignore[attr-defined]
        text=content, user_id=user_id, memory_type=mtype, importance_override=importance,
    )
    return {"memory_id": memory.memory_id, "content": memory.content,
            "importance": memory.importance_score}


@mcp.tool()
async def recall(query: str, user_id: str = "default", top_k: int = 5) -> dict:
    """Retrieve the most relevant memories for a query.

    Uses stateful.ai's hybrid pipeline: dense semantic search + BM25 lexical
    search fused with Reciprocal Rank Fusion, then multi-signal reranking
    (recency, importance, access frequency).

    Returns a ranked list of memories with their scores.
    """
    await _ensure_services()
    result = await _services["retrieve"].retrieve(  # type: ignore[attr-defined]
        RetrievalQuery(query_text=query, user_id=user_id, top_k=top_k)
    )
    return {
        "query": query,
        "results": [
            {
                "rank": c.rank,
                "memory_id": c.memory.memory_id,
                "content": c.memory.content,
                "score": round(c.composite_score, 4),
                "semantic": round(c.semantic_score, 4),
                "lexical": round(c.lexical_score, 4),
            }
            for c in result.candidates
        ],
        "total_found": result.total_found,
        "latency_ms": round(result.latency_ms, 2),
    }


@mcp.tool()
async def forget(memory_id: str, user_id: str = "default") -> dict:
    """Soft-delete a memory by id so it is no longer retrieved."""
    await _ensure_services()
    await _services["db"].delete_memory(memory_id, user_id)  # type: ignore[attr-defined]
    return {"deleted": True, "memory_id": memory_id}


@mcp.tool()
async def list_memories(user_id: str = "default", limit: int = 20) -> dict:
    """List stored memories for a user, most recent first."""
    await _ensure_services()
    items = await _services["db"].list_memories(user_id=user_id, limit=limit)  # type: ignore[attr-defined]
    return {
        "user_id": user_id,
        "count": len(items),
        "memories": [
            {"memory_id": m.memory_id, "content": m.content,
             "importance": m.importance_score} for m in items
        ],
    }


def main() -> None:
    setup_logging(settings.log_level)
    logger.info("Starting stateful.ai MCP server (stdio transport)")
    mcp.run()


if __name__ == "__main__":
    main()
