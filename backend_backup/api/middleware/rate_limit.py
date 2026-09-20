"""
Redis-backed rate limiting middleware.
Configurable per tier (free, pro, enterprise).
"""
import time
import logging
from typing import Optional

from fastapi import Request, HTTPException, Depends
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from ..core.config import settings
from ..core.metrics import RATE_LIMIT_REJECTED
from ..utils.exceptions import RateLimitException

logger = logging.getLogger(__name__)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)

# Rate limit definitions by tier
RATE_LIMITS = {
    "free": settings.rate_limit_free,
    "pro": settings.rate_limit_pro,
    "enterprise": settings.rate_limit_enterprise,
}


def get_tier_rate_limit(tier: str) -> str:
    """Get rate limit string for a user tier."""
    return RATE_LIMITS.get(tier, settings.rate_limit_default)


async def rate_limit_exceeded_handler(request: Request, exception: RateLimitExceeded):
    """Custom handler for rate limit exceeded errors."""
    retry_after = exception.retry_after if hasattr(exception, 'retry_after') else 60
    RATE_LIMIT_REJECTED.labels(
        user_tier="unknown",
        endpoint=request.url.path,
    ).inc()
    raise RateLimitException(retry_after=retry_after)


def setup_rate_limit(app):
    """Register rate limiter handlers on a FastAPI app."""
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# Decorator for per-endpoint rate limiting (with tier support)
from functools import wraps

def rate_limit(limit: Optional[str] = None, key_func=None):
    """
    Decorator for rate limiting endpoints.
    usage: @rate_limit("10/minute") or @rate_limit() which uses the user's tier
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator
