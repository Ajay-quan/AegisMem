"""Production security middleware for the stateful.ai FastAPI service.

Provides, in middleware order (outermost first):

1. **Security headers** - standard hardening headers on every response.
2. **Request size limiting** - rejects oversized bodies before they hit
   the JSON parser (HTTP 413).
3. **API-key authentication** - constant-time comparison of the
   ``X-API-Key`` header against ``settings.api_key``. Disabled when no key
   is configured, so the zero-infra dev experience stays untouched.
4. **Token-bucket rate limiting** - per-client (API key if present,
   otherwise client IP) with a configurable steady rate and burst.

All rejections use a consistent error envelope::

    {"error": {"code": "...", "message": "...", "request_id": "..."}}
"""
from __future__ import annotations

import hmac
import threading
import time

from fastapi import Request
from fastapi.responses import JSONResponse

from core.config import settings
from core.security.keys import get_key_registry
from core.security.audit import get_audit_log

#: Methods that mutate state and are therefore audited.
_MUTATING = frozenset({"POST", "PUT", "PATCH", "DELETE"})

#: Paths that never require authentication or rate limiting.
PUBLIC_PATHS = frozenset(
    {"/", "/health", "/health/ready", "/metrics", "/docs", "/redoc", "/openapi.json"}
)

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "X-Frame-Options": "DENY",
    "Referrer-Policy": "no-referrer",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
}


def error_json(status_code: int, code: str, message: str, request_id: str = "", **headers: str) -> JSONResponse:
    """Build the standard error envelope used by every middleware rejection."""
    response = JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "request_id": request_id}},
    )
    for key, value in headers.items():
        response.headers[key.replace("_", "-")] = value
    return response


def is_public(path: str) -> bool:
    return path in PUBLIC_PATHS or path.startswith("/docs")


def api_key_valid(provided: str | None) -> bool:
    """Constant-time API key check. True when auth is disabled."""
    if not settings.api_key:
        return True
    if not provided:
        return False
    return hmac.compare_digest(provided.encode(), settings.api_key.encode())


def client_identity(request: Request) -> str:
    """Identify a client for rate limiting: API key first, then IP."""
    key = request.headers.get("X-API-Key")
    if key:
        return f"key:{key[:16]}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    return f"ip:{request.client.host if request.client else 'unknown'}"


class TokenBucketLimiter:
    """Thread-safe in-process token bucket limiter.

    Suitable for a single-node deployment; swap for a Redis-backed limiter
    when running multiple replicas.
    """

    def __init__(self, rate_per_minute: int, burst: int) -> None:
        self.rate = rate_per_minute / 60.0  # tokens per second
        self.burst = float(max(burst, 1))
        self._buckets: dict[str, tuple[float, float]] = {}  # id -> (tokens, last_ts)
        self._lock = threading.Lock()
        self._last_sweep = time.monotonic()

    def allow(self, identity: str) -> tuple[bool, float]:
        """Return (allowed, retry_after_seconds)."""
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(identity, (self.burst, now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)
            if tokens >= 1.0:
                self._buckets[identity] = (tokens - 1.0, now)
                self._maybe_sweep(now)
                return True, 0.0
            self._buckets[identity] = (tokens, now)
            return False, (1.0 - tokens) / self.rate

    def _maybe_sweep(self, now: float) -> None:
        """Periodically drop idle buckets so memory stays bounded."""
        if now - self._last_sweep < 300:
            return
        self._last_sweep = now
        idle = [k for k, (_, last) in self._buckets.items() if now - last > 600]
        for k in idle:
            del self._buckets[k]


limiter = TokenBucketLimiter(settings.rate_limit_per_minute, settings.rate_limit_burst)


def install_security(app) -> None:
    """Attach security middleware to the app (registered inner-to-outer)."""

    @app.middleware("http")
    async def auth_and_rate_limit(request: Request, call_next):
        path = request.url.path
        principal = None
        if not is_public(path):
            principal = get_key_registry().resolve(request.headers.get("X-API-Key"))
            if principal is None:
                return error_json(401, "unauthorized", "invalid or missing API key")
            request.state.principal = principal
            if settings.rate_limit_enabled:
                allowed, retry_after = limiter.allow(client_identity(request))
                if not allowed:
                    return error_json(
                        429, "rate_limited",
                        "rate limit exceeded; retry later",
                        Retry_After=f"{retry_after:.0f}" or "1",
                    )

        response = await call_next(request)

        # Audit mutating API operations with the resolved principal.
        if (
            request.method in _MUTATING
            and path.startswith("/api/")
            and principal is not None
        ):
            get_audit_log().record(
                principal=principal.name,
                tenant=principal.tenant,
                method=request.method,
                path=path,
                status=response.status_code,
            )
        return response

    @app.middleware("http")
    async def limit_body_size(request: Request, call_next):
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > settings.max_request_bytes:
            return error_json(
                413, "payload_too_large",
                f"request body exceeds {settings.max_request_bytes} bytes",
            )
        return await call_next(request)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        for header, value in SECURITY_HEADERS.items():
            response.headers.setdefault(header, value)
        if request.url.path.startswith("/api/"):
            response.headers.setdefault("Cache-Control", "no-store")
        return response
