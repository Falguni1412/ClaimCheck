"""
API metadata endpoint.

Health (`/health`, `/api/health`), readiness (`/ready`) and metrics (`/metrics`)
live on the app itself — defining them here as well meant two handlers claimed
the same path and the shadowed one silently never ran.
"""
import logging
from typing import Any, Dict

from fastapi import APIRouter

from ..core.config import settings
from ..services.retriever import retrieval_backends

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["system"])


@router.get("/docs", response_model=Dict[str, Any], summary="Machine-readable API summary")
async def get_docs():
    """
    Endpoint catalogue and enabled features.

    For the full interactive schema use `/docs` (Swagger UI), `/redoc`, or
    `/openapi.json`.
    """
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "interactive_docs": {"swagger": "/docs", "redoc": "/redoc", "openapi": "/openapi.json"},
        "endpoints": {
            "POST /verify": "Verify an LLM answer against sources",
            "POST /verify/batch": "Batch verification (up to 50 items)",
            "POST /verify/stream": "Streaming verification (SSE)",
            "POST /batch/validate": "Pre-flight check for a batch payload",
            "GET /history/": "List verification history (paginated)",
            "GET /history/{request_id}": "Get verification details",
            "POST /admin/login": "Authenticate and get JWT tokens",
            "POST /admin/register": "Create a user account",
            "POST /admin/api-keys": "Create API key",
            "GET /admin/stats": "Aggregate verification statistics",
            "POST /webhooks/": "Register webhook",
            "GET /webhooks/": "List webhooks",
            "DELETE /webhooks/{id}": "Delete a webhook",
            "GET /api/docs": "This document",
            "GET /metrics": "Prometheus metrics",
            "GET /health": "Liveness probe",
            "GET /ready": "Readiness probe (model loaded?)",
        },
        "error_format": {
            "error": {"code": "STRING_CODE", "message": "human readable", "extra": {}},
            "detail": "mirror of error.message",
            "path": "/verify",
            "request_id": "uuid, also returned as the X-Request-ID header",
        },
        "features": {
            "multi_model_ensemble": settings.use_ensemble,
            "numerical_verification": True,
            "streaming": settings.enable_streaming,
            "caching": True,
            "rate_limiting": settings.rate_limit_enabled,
            "auth": True,
            "retrieval": retrieval_backends(),
        },
        "limits": {
            "max_answer_chars": 50_000,
            "max_sources": settings.max_sources,
            "max_source_chars": settings.max_source_length,
            "max_claims_per_request": settings.max_claims_per_request,
            "max_batch_items": 50,
        },
    }
