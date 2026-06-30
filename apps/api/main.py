"""stateful.ai FastAPI application."""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from core.config import settings
from core.version import __version__, PRODUCT_NAME, ENGINE_NAME
from core.logging.logger import setup_logging, new_request_id, set_context
from core.observability import metrics
from apps.api.routers import router
from apps.api.eval_router import eval_router
from apps.api.schemas import HealthResponse
from apps.api.security import install_security

setup_logging(settings.log_level)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info(f"stateful.ai starting up (env={settings.app_env})")
    try:
        from apps.api.dependencies import get_db_store, get_vector_store, get_graph_store
        await get_db_store()
        await get_vector_store()
        await get_graph_store()
        logger.info("All stores initialized successfully")
    except Exception as e:
        logger.warning(f"Store initialization warning: {e}")
    yield
    logger.info("stateful.ai shutting down")
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
        title=f"{PRODUCT_NAME} — Self-Improving Memory for LLM Agents",
        description=(
            f"{PRODUCT_NAME} (engine: {ENGINE_NAME}) — production-grade persistent memory "
            "for long-running LLM agents: multi-store hybrid retrieval, versioned updates, "
            "contradiction detection, reflection, and continual learning from feedback."
        ),
        version=__version__,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # CORS (configurable; "*" by default for local development)
    origins = [o.strip() for o in settings.cors_allow_origins.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API-key auth, rate limiting, body-size limits, security headers
    install_security(app)

    # Request tracing middleware
    @app.middleware("http")
    async def trace_middleware(request: Request, call_next):
        request_id = new_request_id()
        start = time.time()
        response = await call_next(request)
        duration_s = time.time() - start
        duration_ms = duration_s * 1000
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration_ms:.1f}ms"
        # Use the route template (not the raw path) to keep metric cardinality low.
        route = request.scope.get("route")
        path_label = getattr(route, "path", request.url.path)
        metrics.observe_request(
            request.method, path_label, response.status_code, duration_s,
        )
        logger.info(
            f"{request.method} {request.url.path} -> {response.status_code} "
            f"({duration_ms:.1f}ms) rid={request_id}"
        )
        return response

    # Exception handler - never leak internals; include the request id so
    # operators can correlate a client report with server logs.
    @app.exception_handler(Exception)
    async def generic_handler(request: Request, exc: Exception):
        request_id = request.headers.get("X-Request-ID", "")
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "code": "internal_error",
                    "message": "Internal server error",
                    "request_id": request_id,
                },
                # Kept for backward compatibility with pre-0.2 clients.
                "detail": "Internal server error",
                "type": type(exc).__name__,
            },
        )

    # Health check (liveness)
    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health():
        return HealthResponse(
            status="ok",
            version=__version__,
            components={
                "api": "ok",
                "product": PRODUCT_NAME,
                "config": settings.app_env,
                "auth": "enabled" if settings.api_key else "disabled",
                "rate_limit": "enabled" if settings.rate_limit_enabled else "disabled",
            },
        )

    # Readiness: verifies the stores actually respond, not just that the
    # process is alive. Wire this to your orchestrator's readiness probe.
    @app.get("/health/ready", tags=["system"])
    async def readiness():
        components: dict[str, str] = {}
        healthy = True
        try:
            from apps.api.dependencies import get_db_store, get_vector_store, get_graph_store
            db = await get_db_store()
            components["relational_store"] = type(db).__name__
            vs = await get_vector_store()
            components["vector_store"] = type(vs).__name__
            graph = await get_graph_store()
            components["graph_store"] = type(graph).__name__
        except Exception as e:
            healthy = False
            components["error"] = str(e)
        return JSONResponse(
            status_code=200 if healthy else 503,
            content={"status": "ready" if healthy else "degraded", "components": components},
        )

    # Prometheus metrics exporter
    @app.get("/metrics", tags=["system"])
    async def prometheus_metrics():
        return Response(content=metrics.render(), media_type=metrics.CONTENT_TYPE_LATEST)

    # Consistent error envelope for HTTP errors (keeps "detail" for
    # backward compatibility with pre-0.2 clients).
    from fastapi import HTTPException
    from fastapi.exceptions import RequestValidationError

    @app.exception_handler(HTTPException)
    async def http_error_handler(request: Request, exc: HTTPException):
        codes = {401: "unauthorized", 403: "forbidden", 404: "not_found",
                 409: "conflict", 422: "validation_error", 429: "rate_limited"}
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": codes.get(exc.status_code, "error"),
                    "message": str(exc.detail),
                },
                "detail": exc.detail,
            },
            headers=getattr(exc, "headers", None) or {},
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        from fastapi.encoders import jsonable_encoder
        fields = jsonable_encoder(exc.errors(), custom_encoder={Exception: str})
        return JSONResponse(
            status_code=422,
            content={
                "error": {
                    "code": "validation_error",
                    "message": "request validation failed",
                    "fields": fields,
                },
                "detail": fields,
            },
        )

    @app.get("/", tags=["system"])
    async def root():
        return {
            "name": PRODUCT_NAME,
            "engine": ENGINE_NAME,
            "version": __version__,
            "description": "Self-improving persistent memory for LLM agents",
            "docs": "/docs",
            "health": "/health",
        }

    # Mount memory API routes
    app.include_router(router, prefix="/api/v1", tags=["memory"])
    app.include_router(eval_router, prefix="/api/v1/eval", tags=["evaluation"])

    return app


app = create_app()
