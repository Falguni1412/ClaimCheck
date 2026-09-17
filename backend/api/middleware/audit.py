"""
Audit logging middleware.
Records all API requests to the audit_logs table for compliance.
"""
import time
import logging
from typing import Callable

from fastapi import Request, Response

from ..core.metrics import REQUEST_COUNT, REQUEST_LATENCY, ACTIVE_REQUESTS
from ..models.database import AuditLog, db_session_context
from ..utils.exceptions import ClaimCheckException

logger = logging.getLogger(__name__)


async def log_request(
    request: Request,
    response: Response,
    next: Callable,
):
    """
    ASGI middleware that logs request details to the audit log.
    """
    start_time = time.time()
    request_id = getattr(request.state, "request_id", None)

    # Active requests gauge
    ACTIVE_REQUESTS.labels(endpoint=request.url.path).inc()

    # Count request
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status_code=response.status_code,
    ).inc()

    # Track latency
    REQUEST_LATENCY.labels(method=request.method, endpoint=request.url.path).observe(
        time.time() - start_time
    )

    try:
        response = await next(request)
        status_code = response.status_code
    except Exception:
        status_code = 500
        raise
    finally:
        ACTIVE_REQUESTS.labels(endpoint=request.url.path).dec()

        # Write audit log asynchronously (best effort)
        try:
            await _write_audit_log(
                request=request,
                status_code=status_code,
                duration_ms=(time.time() - start_time) * 1000,
            )
        except Exception:
            logger.warning("Failed to write audit log", exc_info=True)

    return response


async def _write_audit_log(
    request: Request,
    status_code: int,
    duration_ms: float,
):
    """Write an audit log entry to the database."""
    # This is called from middleware; skip if DB not configured
    try:
        async with db_session_context() as session:
            audit_entry = AuditLog(
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                request_size=int(request.headers.get("content-length", 0)),
                duration_ms=int(duration_ms),
                error=str(request.url.path),
            )
            session.add(audit_entry)
    except Exception:
        pass  # Best-effort logging


async def audit_middleware(
    request: Request,
    call_next: Callable,
):
    """Main audit middleware factory."""
    start_time = time.time()
    user_id = getattr(request.state, "user_id", None)

    try:
        response = await call_next(request)
        duration_ms = (time.time() - start_time) * 1000

        # Write audit log in background
        if user_id and user_id != "anonymous":
            await _record_audit_entry(
                request=request,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id,
            )

        return response
    except Exception as e:
        duration_ms = (time.time() - start_time) * 1000
        await _record_error_audit(
            request=request,
            error=str(e),
            duration_ms=duration_ms,
        )
        raise


async def _record_audit_entry(
    request: Request,
    status_code: int,
    duration_ms: float,
    user_id: int,
):
    """Record a successful request to audit log."""
    try:
        async with db_session_context() as session:
            audit_entry = AuditLog(
                user_id=user_id,
                method=request.method,
                path=str(request.url.path),
                status_code=status_code,
                ip_address=request.client.host if request.client else None,
                user_agent=request.headers.get("user-agent"),
                request_size=int(request.headers.get("content-length", 0)),
                duration_ms=int(duration_ms),
            )
            session.add(audit_entry)
    except Exception:
        pass


async def _record_error_audit(
    request: Request,
    error: str,
    duration_ms: float,
):
    """Record a failed request to audit log."""
    try:
        async with db_session_context() as session:
            audit_entry = AuditLog(
                method=request.method,
                path=str(request.url.path),
                status_code=500,
                ip_address=request.client.host if request.client else None,
                duration_ms=int(duration_ms),
                error=error[:500],
            )
            session.add(audit_entry)
    except Exception:
        pass
