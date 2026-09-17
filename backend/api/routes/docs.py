"""
System documentation and health endpoints.
"""
import uuid
import logging
from datetime import datetime

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Any, Dict

from ..models.database import Verification, get_db_session
from ..core.config import settings
from ..core.metrics import APP_INFO

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["docs"])


@router.get("/docs", response_model=Dict[str, Any])
async def get_docs():
    """Return API documentation info."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
        "endpoints": {
            "POST /verify": "Verify an LLM answer against sources",
            "POST /verify/batch": "Batch verification (up to 50 items)",
            "POST /verify/stream": "Streaming verification (SSE)",
            "GET /history": "Get verification history",
            "GET /history/{request_id}": "Get verification details",
            "POST /admin/login": "Authenticate and get JWT tokens",
            "POST /admin/api-keys": "Create API key",
            "POST /webhooks": "Register webhook",
            "GET /api/docs": "This documentation",
            "GET /metrics": "Prometheus metrics",
            "GET /health": "Health check",
        },
        "features": {
            "multi_model_ensemble": settings.use_ensemble,
            "numerical_verification": True,
            "streaming": settings.enable_streaming,
            "caching": settings.redis_url != "redis://localhost:6379/0",
            "rate_limiting": settings.rate_limit_enabled,
            "auth": True,
        },
    }


@router.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": settings.app_version,
        "timestamp": datetime.now().isoformat(),
    }


@router.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    from ..core.metrics import render_metrics
    data, content_type = render_metrics()
    from fastapi.responses import Response
    return Response(content=data, media_type=content_type)
