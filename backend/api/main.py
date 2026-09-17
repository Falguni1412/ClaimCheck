"""
ClaimCheck API: Claim-Level LLM Verification Engine.
FastAPI application with multi-model NLI, caching, and streaming support.
"""
from __future__ import annotations

import asyncio
import logging
import sys
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from .core.config import settings
from .core.metrics import render_metrics
from .services.cache import CacheService
from .utils.exceptions import ClaimCheckException, ModelNotLoadedException, RateLimitException
from .middleware.rate_limit import setup_rate_limit
from .middleware.audit import audit_middleware
from .routes.verify import router as verify_router, init_verifier as init_verify
from .routes.batch import router as batch_router
from .routes.history import router as history_router
from .routes.admin import router as admin_router
from .routes.webhooks import router as webhook_router
from .routes.docs import router as docs_router

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("claimcheck")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: init on startup, cleanup on shutdown."""
    logger.info(
        "Starting ClaimCheck v%s in %s mode...",
        settings.app_version,
        settings.environment,
    )

    await init_verify()

    cache = CacheService()
    await cache.connect()
    logger.info("Cache service ready")

    try:
        from .models.database import init_db

        await init_db()
        logger.info("Database initialized")
    except Exception as e:
        logger.warning("Database init skipped (verification still works): %s", e)

    if settings.rate_limit_enabled:
        setup_rate_limit(app)
        logger.info("Rate limiting configured")

    yield
    logger.info("Shutting down ClaimCheck...")


app = FastAPI(
    title=settings.app_name,
    description="Claim-Level LLM Verification Engine — verify any LLM answer against source documents",
    version=settings.app_version,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.middleware("http")(audit_middleware)

if settings.rate_limit_enabled:
    setup_rate_limit(app)


@app.middleware("http")
async def add_request_time(request: Request, call_next):
    """Track request time and add to state for audit."""
    start = asyncio.get_event_loop().time()
    request.state.request_time = start
    request.state.request_id = str(uuid.uuid4())
    response = await call_next(request)
    duration = (asyncio.get_event_loop().time() - start) * 1000
    response.headers["X-Response-Time"] = f"{duration:.0f}ms"
    return response


@app.get("/", response_model=dict, include_in_schema=False)
async def root():
    """Root endpoint — health check."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": settings.app_version,
    }


@app.get("/api/health", response_model=dict)
async def health():
    """Explicit health endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": settings.app_version,
    }


app.include_router(verify_router)
app.include_router(batch_router)
app.include_router(history_router)
app.include_router(admin_router)
app.include_router(webhook_router)
app.include_router(docs_router)


@app.exception_handler(ClaimCheckException)
async def claim_check_exception_handler(request: Request, exc: ClaimCheckException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {
                "code": exc.error_code,
                "message": exc.detail,
                "extra": exc.extra,
            },
            "path": str(request.url.path),
        },
    )


@app.exception_handler(ModelNotLoadedException)
async def model_not_loaded_handler(request: Request, exc: ModelNotLoadedException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": {"code": exc.error_code, "message": exc.detail},
            "path": str(request.url.path),
        },
    )


@app.exception_handler(RateLimitException)
async def rate_limit_exception_handler(request: Request, exc: RateLimitException):
    headers = {"Retry-After": str(exc.extra.get("retry_after_seconds", 60))}
    return JSONResponse(
        status_code=429,
        content={
            "error": {"code": exc.error_code, "message": exc.detail},
            "path": str(request.url.path),
        },
        headers=headers,
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled error on %s", request.url.path)
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An unexpected error occurred. See logs for details.",
            },
            "path": str(request.url.path),
        },
    )


@app.get("/metrics", include_in_schema=False)
async def metrics():
    data, content_type = render_metrics()
    return Response(content=data, media_type=content_type)
