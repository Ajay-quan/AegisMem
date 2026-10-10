"""AegisMem FastAPI application."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.logging.logger import setup_logging, new_request_id, set_context
from apps.api.routers import router
from apps.api.eval_router import eval_router
from apps.api.admin_router import admin_router
from apps.api.auth import authorize, require_admin
from adapters.tenancy import TenantViolation
from core.exceptions import MemoryNotFoundError
from apps.api.schemas import HealthResponse

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"AegisMem starting up (env={settings.app_env})")
    validate_runtime_config()
    try:
        from apps.api.control import get_control_store
        from apps.api.dependencies import get_raw_db_store, get_raw_vector_store, get_graph_store
        await get_control_store()
        await get_raw_db_store()
        await get_raw_vector_store()
        await get_graph_store()
        logger.info("All stores initialized successfully")
    except Exception as e:
        if settings.is_production_like:
            raise  # fail closed: never serve without durable stores
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


def validate_runtime_config() -> None:
    """Refuse to boot staging/production with unsafe settings."""
    if not settings.is_production_like:
        if not settings.auth_enabled:
            logger.warning("AUTH DISABLED (development only): all requests act as tenant "
                           f"'{settings.default_tenant}' with every scope")
        return
    problems = []
    if not settings.auth_enabled:
        problems.append("AUTH_ENABLED must be true")
    if settings.app_secret_key in ("", "dev-secret-key", "change-me-in-production", "your-secret-key-change-in-production"):
        problems.append("APP_SECRET_KEY must be set to a strong secret")
    if not settings.api_key_pepper:
        problems.append("API_KEY_PEPPER must be set (separate from APP_SECRET_KEY)")
    if problems:
        raise RuntimeError("Unsafe configuration: " + "; ".join(problems))


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

    # Typed errors first. Cross-tenant access is indistinguishable from "not found".
    @app.exception_handler(MemoryNotFoundError)
    async def not_found_handler(request: Request, exc: MemoryNotFoundError):
        return JSONResponse(status_code=404, content={"detail": "memory not found"})

    @app.exception_handler(TenantViolation)
    async def tenant_handler(request: Request, exc: TenantViolation):
        logger.warning(f"Tenant violation on {request.url.path}: {exc}")
        return JSONResponse(status_code=404, content={"detail": "not found"})

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
    app.include_router(router, prefix="/api/v1", tags=["memory"],
                       dependencies=[Depends(authorize)])
    app.include_router(admin_router, prefix="/api/v1/admin", tags=["admin"])
    app.include_router(eval_router, prefix="/api/v1/eval", tags=["evaluation"],
                       dependencies=[Depends(require_admin)])

    return app


app = create_app()
