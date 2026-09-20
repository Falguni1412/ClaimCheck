"""Core package for configuration, security, and metrics."""
from .config import settings
from .security import (
    create_access_token,
    create_refresh_token,
    verify_token,
    hash_password,
    verify_password,
    generate_api_key,
)
from .metrics import (
    REQUEST_COUNT,
    REQUEST_LATENCY,
    VERIFY_REQUESTS,
    CACHE_HITS,
    CACHE_MISSES,
    ACTIVE_REQUESTS,
)

__all__ = [
    "settings",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "hash_password",
    "verify_password",
    "generate_api_key",
    "REQUEST_COUNT",
    "REQUEST_LATENCY",
    "VERIFY_REQUESTS",
    "CACHE_HITS",
    "CACHE_MISSES",
    "ACTIVE_REQUESTS",
]
