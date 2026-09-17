"""
Redis cache layer with in-memory fallback.
Caches embeddings, verification results, and token blocklists.
"""
import json
import hashlib
import logging
from typing import Any, Optional, Dict
from datetime import datetime, timedelta

try:
    import redis.asyncio as redis_async
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

from ..core.config import settings
from ..core.metrics import CACHE_HITS, CACHE_MISSES, CACHE_SIZE

logger = logging.getLogger(__name__)


class InMemoryCache:
    """Simple in-memory cache with TTL."""

    def __init__(self, max_size: int = 10000):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._max_size = max_size

    def get(self, key: str) -> Optional[Any]:
        if key not in self._cache:
            CACHE_MISSES.labels(cache_type="in_memory").inc()
            return None

        value, expires_at = self._cache[key]
        if datetime.utcnow() >= expires_at:
            del self._cache[key]
            CACHE_MISSES.labels(cache_type="in_memory").inc()
            return None

        CACHE_HITS.labels(cache_type="in_memory").inc()
        return value

    def set(self, key: str, value: Any, ttl_seconds: int = 3600) -> None:
        # Evict oldest if over capacity
        if len(self._cache) >= self._max_size:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]

        expires_at = datetime.utcnow() + timedelta(seconds=ttl_seconds)
        self._cache[key] = (value, expires_at)
        CACHE_SIZE.labels(cache_type="in_memory").set(len(self._cache))

    def delete(self, key: str) -> None:
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        self._cache.clear()

    def size(self) -> int:
        return len(self._cache)


class CacheService:
    """
    Cache abstraction with Redis primary and in-memory fallback.
    Handles serialization, TTL, and key namespacing.
    """

    def __init__(self):
        self._redis: Optional[Any] = None
        self._memory = InMemoryCache(max_size=settings.cache_max_size)
        self._use_redis = False

    async def connect(self) -> None:
        """Connect to Redis if available; otherwise use in-memory."""
        if not REDIS_AVAILABLE:
            logger.warning("redis package not installed, using in-memory cache only")
            return

        try:
            self._redis = redis_async.from_url(
                settings.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=2.0,
            )
            # Test connection
            await self._redis.ping()
            self._use_redis = True
            logger.info(f"Connected to Redis at {settings.redis_url}")
        except Exception as e:
            logger.warning(f"Could not connect to Redis ({e}); using in-memory cache")
            self._redis = None
            self._use_redis = False

    async def close(self) -> None:
        if self._redis:
            await self._redis.close()

    @staticmethod
    def make_key(namespace: str, *parts: Any) -> str:
        """Build a namespaced cache key."""
        key_parts = [str(namespace)] + [str(p) for p in parts]
        return ":".join(key_parts)

    @staticmethod
    def hash_key(data: Any) -> str:
        """Generate a stable hash for arbitrary data."""
        if isinstance(data, str):
            payload = data
        else:
            payload = json.dumps(data, sort_keys=True, default=str)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        if self._use_redis and self._redis:
            try:
                raw = await self._redis.get(key)
                if raw is None:
                    CACHE_MISSES.labels(cache_type="redis").inc()
                    return None
                CACHE_HITS.labels(cache_type="redis").inc()
                return json.loads(raw)
            except Exception as e:
                logger.warning(f"Redis get failed: {e}; falling back to memory")

        return self._memory.get(key)

    async def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None) -> None:
        """Set a value with TTL."""
        if ttl_seconds is None:
            ttl_seconds = settings.cache_ttl

        serialized = json.dumps(value, default=str)

        if self._use_redis and self._redis:
            try:
                await self._redis.set(key, serialized, ex=ttl_seconds)
                return
            except Exception as e:
                logger.warning(f"Redis set failed: {e}; falling back to memory")

        self._memory.set(key, json.loads(serialized), ttl_seconds)

    async def delete(self, key: str) -> None:
        """Delete a key from cache."""
        if self._use_redis and self._redis:
            try:
                await self._redis.delete(key)
                return
            except Exception:
                pass
        self._memory.delete(key)

    async def clear_namespace(self, namespace: str) -> None:
        """Clear all keys in a namespace (Redis only)."""
        if self._use_redis and self._redis:
            try:
                cursor = 0
                pattern = f"{namespace}:*"
                while True:
                    cursor, keys = await self._redis.scan(cursor=cursor, match=pattern, count=100)
                    if keys:
                        await self._redis.delete(*keys)
                    if cursor == 0:
                        break
            except Exception as e:
                logger.warning(f"Redis namespace clear failed: {e}")

    async def health_check(self) -> bool:
        """Check whether the cache is reachable."""
        if self._use_redis and self._redis:
            try:
                return await self._redis.ping()
            except Exception:
                return False
        return True  # In-memory cache is always "healthy"


# Singleton
_cache: Optional[CacheService] = None


def get_cache() -> CacheService:
    global _cache
    if _cache is None:
        _cache = CacheService()
    return _cache
