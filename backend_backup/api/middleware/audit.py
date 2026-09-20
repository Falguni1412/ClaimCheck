"""
Audit + request metrics middleware.

Records Prometheus request metrics for every call and (best effort) writes an
audit row. The previous version defined an unused `log_request` helper that
read `response.status_code` before the response existed, and recorded metrics
nowhere that ran — so `claimcheck_requests_total` was always empty.
"""
import logging
import time
from typing import Callable, Optional

from fastapi import Request

from ..core.config import settings
from ..core.metrics import ACTIVE_REQUESTS, REQUEST_COUNT, REQUEST_LATENCY
from ..models.database import AuditLog, db_session_context

logger = logging.getLogger(__name__)

# Unbounded label values blow up Prometheus cardinality, so path params are
# collapsed to the matched route template where possible.
_SKIP_PATHS = {"/metrics", "/health", "/api/health", "/ready", "/favicon.ico"}


def _endpoint_label(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


async def audit_middleware(request: Request, call_next: Callable):
    """Time every request, count it, and record an audit row for failures."""
    path = request.url.path
    start = time.perf_counter()
    status_code = 500
    ACTIVE_REQUESTS.labels(endpoint=path).inc()

    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    except Exception as e:
        duration_ms = (time.perf_counter() - start) * 1000
        await _record_audit_entry(
            request, status_code=500, duration_ms=duration_ms, error=str(e)
        )
        raise
    finally:
        duration = time.perf_counter() - start
        ACTIVE_REQUESTS.labels(endpoint=path).dec()
        if path not in _SKIP_PATHS:
            endpoint = _endpoint_label(request)
            REQUEST_COUNT.labels(
                method=request.method, endpoint=endpoint, status_code=status_code
            ).inc()
            REQUEST_LATENCY.labels(method=request.method, endpoint=endpoint).observe(duration)


async def _record_audit_entry(
    request: Request,
    status_code: int,
    duration_ms: float,
    error: Optional[str] = None,
) -> None:
    """Write one audit row. Never raises — auditing must not break the request."""
    if not settings.persist_verifications:
        return
    try:
        async with db_session_context() as session:
            session.add(
                AuditLog(
                    user_id=getattr(request.state, "user_id", None),
                    method=request.method,
                    path=str(request.url.path)[:500],
                    status_code=status_code,
                    ip_address=request.client.host if request.client else None,
                    user_agent=(request.headers.get("user-agent") or "")[:500] or None,
                    request_size=int(request.headers.get("content-length") or 0),
                    duration_ms=int(duration_ms),
                    error=error[:500] if error else None,
                )
            )
    except Exception:
        logger.debug("Audit log write skipped", exc_info=True)
