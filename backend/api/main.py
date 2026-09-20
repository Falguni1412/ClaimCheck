"""
ClaimCheck API: Claim-Level LLM Verification Engine.
FastAPI application with NLI verification, hybrid retrieval, caching, and
streaming support.
"""
from __future__ import annotations

import logging
import sys
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

from .core.config import settings
from .core.metrics import render_metrics
from .middleware.audit import audit_middleware
from .middleware.rate_limit import setup_rate_limit
from .routes.admin import router as admin_router
from .routes.batch import router as batch_router
from .routes.docs import router as docs_router
from .routes.history import router as history_router
from .routes.verify import init_verifier as init_verify
from .routes.verify import models_ready
from .routes.verify import router as verify_router
from .routes.webhooks import router as webhook_router
from .services.cache import get_cache
from .services.retriever import retrieval_backends
from .utils.exceptions import ClaimCheckException

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("claimcheck")

STARTED_AT = time.time()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _error_body(
    code: str,
    message: str,
    request: Request,
    extra: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """The single error shape every failure response uses."""
    return {
        "error": {"code": code, "message": message, "extra": extra or {}},
        # Mirrored for clients (including this repo's frontend) that read `detail`.
        "detail": message,
        "path": str(request.url.path),
        "request_id": getattr(request.state, "request_id", None),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init on startup, cleanup on shutdown."""
    logger.info(
        "Starting ClaimCheck v%s in %s mode...", settings.app_version, settings.environment
    )
    for warning in settings.startup_warnings():
        logger.warning("CONFIG: %s", warning)

    # The database is optional: verification works without it, history does not.
    try:
        from .models.database import init_db

        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init skipped (verification still works): %s", e)

    try:
        await init_verify()
    except Exception:
        # A model that fails to load must not stop the process from booting —
        # otherwise the platform restart-loops and you never see the traceback.
        # /health stays up, /ready reports not-ready, /verify returns 503.
        logger.exception("Model initialization failed; API will report not-ready")

    yield

    logger.info("Shutting down ClaimCheck...")
    try:
        await get_cache().close()
    except Exception:
        pass
    try:
        from .services.verifier import reset_verifier

        reset_verifier()
    except Exception:
        pass


TAGS_METADATA = [
    {"name": "verification", "description": "Verify LLM answers against source documents."},
    {"name": "history", "description": "Past verification runs and their details."},
    {"name": "system", "description": "Health, readiness, metrics, and API metadata."},
    {"name": "admin", "description": "Accounts and API keys."},
    {"name": "webhooks", "description": "Delivery of verification events to your endpoints."},
]

app = FastAPI(
    title=settings.app_name,
    description=(
        "Claim-Level LLM Verification Engine — decompose any LLM answer into atomic "
        "claims, retrieve supporting evidence from your own documents, and classify "
        "each claim as SUPPORTED, CONTRADICTED, or UNVERIFIABLE.\n\n"
        "**Errors** always return `{ \"error\": { \"code\", \"message\" }, \"detail\", \"path\", \"request_id\" }`.\n\n"
        "**Probes**: `/health` is liveness (cheap), `/ready` is readiness (reports whether "
        "the model is actually loaded). During a cold start `/health` returns 200 while "
        "`/ready` returns 503."
    ),
    version=settings.app_version,
    lifespan=lifespan,
    openapi_tags=TAGS_METADATA,
    contact={"name": "ClaimCheck", "url": "https://github.com/Falguni1412/ClaimCheck"},
    license_info={"name": "MIT"},
)

# Compress large verification payloads (evidence text repeats a lot).
app.add_middleware(GZipMiddleware, minimum_size=1024)

# Browsers reject `Access-Control-Allow-Origin: *` together with credentials,
# so the two settings are derived from one another rather than set blindly.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=not settings.cors_allow_all,
    allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["X-Request-ID", "X-Response-Time"],
    max_age=600,
)

app.middleware("http")(audit_middleware)

if settings.rate_limit_enabled:
    setup_rate_limit(app)


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    """Attach a request id + timing to every response (outermost middleware)."""
    request_id = request.headers.get("X-Request-ID") or str(uuid.uuid4())
    request.state.request_id = request_id
    start = time.perf_counter()
    request.state.request_time = start

    response = await call_next(request)

    duration_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Response-Time"] = f"{duration_ms:.0f}ms"
    if duration_ms > 5000:
        logger.warning(
            "Slow request %s %s took %.0fms (request_id=%s)",
            request.method, request.url.path, duration_ms, request_id,
        )
    return response


# ---------------------------------------------------------------- probes


@app.get("/", include_in_schema=False)
async def root():
    """Root — liveness plus a pointer at the docs."""
    return {
        "status": "healthy",
        "model_loaded": models_ready(),
        "version": settings.app_version,
        "docs": "/docs",
    }


@app.get("/health", tags=["system"], summary="Liveness probe")
async def health():
    """
    Cheap liveness check. Returns 200 as soon as the process is serving, even
    while models are still loading — so platform health checks do not kill a
    cold start.
    """
    return {"status": "healthy", "version": settings.app_version, "timestamp": _now()}


@app.get("/api/health", tags=["system"], summary="Liveness probe (legacy path)")
async def api_health():
    """Alias of `/health`, kept for existing clients."""
    return await health()


@app.get("/ready", tags=["system"], summary="Readiness probe")
async def ready():
    """
    Readiness check: reports whether this instance can actually serve a
    verification. Returns 503 until the NLI model is loaded.
    """
    cache_ok = False
    try:
        cache_ok = bool(await get_cache().health_check())
    except Exception:
        cache_ok = False

    db_ok = False
    try:
        from sqlalchemy import text

        from .models.database import get_engine

        async with get_engine().connect() as conn:
            await conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        db_ok = False

    model_ok = models_ready()
    payload = {
        "status": "ready" if model_ok else "loading",
        "ready": model_ok,
        "model_loaded": model_ok,
        "version": settings.app_version,
        "timestamp": _now(),
        "uptime_seconds": round(time.time() - STARTED_AT, 1),
        "database": db_ok,
        "cache": cache_ok,
        "retrieval": retrieval_backends(),
    }
    return JSONResponse(status_code=200 if model_ok else 503, content=payload)


@app.get("/metrics", include_in_schema=False)
async def metrics():
    if not settings.enable_metrics:
        return JSONResponse(status_code=404, content={"detail": "Metrics disabled"})
    data, content_type = render_metrics()
    return Response(content=data, media_type=content_type)


# ---------------------------------------------------------------- routers

app.include_router(verify_router)
app.include_router(batch_router)
app.include_router(history_router)
app.include_router(admin_router)
app.include_router(webhook_router)
app.include_router(docs_router)


# ---------------------------------------------------------------- errors


@app.exception_handler(ClaimCheckException)
async def claim_check_exception_handler(request: Request, exc: ClaimCheckException):
    """All ClaimCheck errors (including 503/429/404 subclasses) share one shape."""
    headers = None
    if exc.status_code == 429:
        headers = {"Retry-After": str(exc.extra.get("retry_after_seconds", 60))}
    if exc.status_code >= 500:
        logger.error(
            "%s on %s: %s", exc.error_code, request.url.path, exc.detail,
        )
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(exc.error_code or "ERROR", str(exc.detail), request, exc.extra),
        headers=headers,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """Plain HTTPExceptions (404s, manual raises) get the same envelope."""
    code = {400: "BAD_REQUEST", 401: "UNAUTHORIZED", 403: "FORBIDDEN",
            404: "NOT_FOUND", 405: "METHOD_NOT_ALLOWED"}.get(exc.status_code, "HTTP_ERROR")
    return JSONResponse(
        status_code=exc.status_code,
        content=_error_body(code, str(exc.detail), request),
        headers=getattr(exc, "headers", None),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Pydantic validation failures, in the standard envelope with field details."""
    fields = [
        {
            "field": ".".join(str(p) for p in err.get("loc", []) if p != "body") or "body",
            "message": err.get("msg", "invalid value"),
            "type": err.get("type", "value_error"),
        }
        for err in exc.errors()
    ]
    message = fields[0]["message"] if len(fields) == 1 else f"{len(fields)} fields failed validation"
    return JSONResponse(
        status_code=422,
        content=_error_body("VALIDATION_ERROR", message, request, {"fields": fields}),
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Last resort. The traceback goes to logs; the client gets a request id."""
    logger.exception(
        "Unhandled error on %s (request_id=%s)",
        request.url.path, getattr(request.state, "request_id", None),
    )
    return JSONResponse(
        status_code=500,
        content=_error_body(
            "INTERNAL_ERROR",
            "An unexpected error occurred. Quote the request_id when reporting this.",
            request,
        ),
    )
