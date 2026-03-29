"""AegisMem FastAPI application."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging.logger import setup_logging, new_request_id, set_context
from apps.api.routers import router
from apps.api.eval_router import eval_router
from apps.api.schemas import HealthResponse

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"AegisMem starting up (env={settings.app_env})")
    try:
        from apps.api.dependencies import get_db_store, get_vector_store, get_graph_store
        await get_db_store()
        await get_vector_store()
        await get_graph_store()
        logger.info("All stores initialized successfully")
    except Exception as e:
        logger.warning(f"Store initialization warning: {e}")
    yield
    logger.info("AegisMem shutting down")
    try:
        from apps.api.dependencies import _db_store, _graph_store
        if _db_store:
            await _db_store.close()
        if _graph_store:
            await _graph_store.close()
    except Exception:
        pass


def create_app() -> FastAPI:
    app = FastAPI(
        title="AegisMem - Persistent Memory for LLM Agents",
        description=(
            "Production-grade persistent memory system for long-running LLM agents. "
            "Supports multi-store hybrid retrieval, versioned updates, contradiction detection, "
            "and higher-level reflection generation."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request tracing middleware
    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        request_id = new_request_id()
        start = time.time()
        response = await call_next(request)
        duration_ms = (time.time() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({duration_ms:.1f}ms) rid={request_id}"
        )
        return response

    # Exception handler
    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "type": type(exc).__name__},
        )

    # Health check
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        return HealthResponse(
            status="ok",
            version="0.1.0",
            components={
                "api": "ok",
                "config": settings.app_env,
            },
        )

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": "AegisMem",
            "version": "0.1.0",
            "description": "Persistent Memory Architecture for LLM Agents",
            "docs": "/docs",
            "health": "/health",
        }

    # Mount memory API routes
    app.include_router(router, prefix="/api/v1", tags=["memory"])
    app.include_router(eval_router, prefix="/api/v1/eval", tags=["evaluation"])

    return app


app = create_app()
