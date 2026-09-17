"""
Unit tests for cache service.
"""
import pytest
import asyncio
from backend.api.services.cache import InMemoryCache, CacheService, get_cache


class TestInMemoryCache:
    """Tests for InMemoryCache."""

    def test_set_and_get(self):
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl_seconds=60)
        assert cache.get("key1") == "value1"

    def test_get_missing_key(self):
        cache = InMemoryCache()
        assert cache.get("nonexistent") is None

    def test_ttl_expiration(self):
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl_seconds=0)  # 0-second TTL
        import time
        time.sleep(0.1)
        assert cache.get("key1") is None

    def test_delete(self):
        cache = InMemoryCache()
        cache.set("key1", "value1", ttl_seconds=60)
        cache.delete("key1")
        assert cache.get("key1") is None

    def test_max_size_eviction(self):
        cache = InMemoryCache(max_size=2)
        cache.set("a", 1, ttl_seconds=60)
        cache.set("b", 2, ttl_seconds=60)
        cache.set("c", 3, ttl_seconds=60)  # Should evict "a"
        # One of a, b should be gone
        assert cache.size() == 2


class TestCacheService:
    """Tests for CacheService."""

    @pytest.mark.asyncio
    async def test_connect_without_redis(self):
        """Without Redis, falls back to in-memory."""
        cache = CacheService()
        # Don't call connect() — simulates Redis being unavailable
        cache._use_redis = False
        await cache.set("test", {"data": "value"})
        result = await cache.get("test")
        assert result == {"data": "value"}

    @pytest.mark.asyncio
    async def test_health_check(self):
        cache = CacheService()
        # In-memory always healthy
        is_healthy = await cache.health_check()
        assert is_healthy is True

    def test_make_key(self):
        key = CacheService.make_key("namespace", "a", "b", "c")
        assert key == "namespace:a:b:c"

    def test_hash_key_consistent(self):
        h1 = CacheService.hash_key("test data")
        h2 = CacheService.hash_key("test data")
        assert h1 == h2

    def test_hash_key_unique(self):
        h1 = CacheService.hash_key("data1")
        h2 = CacheService.hash_key("data2")
        assert h1 != h2
