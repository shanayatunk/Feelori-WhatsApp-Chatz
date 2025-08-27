# /app/services/cache_service.py

import json
import logging
from typing import Optional
import redis.asyncio as redis

from app.config.settings import settings
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import cache_operations

# This service manages all interactions with the Redis cache, providing a
# centralized point for caching logic with built-in error handling and metrics.

logger = logging.getLogger(__name__)

class CacheService:
    def __init__(self, redis_url: str):
        try:
            self.redis_pool = redis.ConnectionPool.from_url(redis_url, max_connections=20)
            self.redis = redis.Redis(connection_pool=self.redis_pool)
            self.circuit_breaker = CircuitBreaker()
        except Exception as e:
            logger.critical(f"Failed to connect to Redis at {redis_url}: {e}")
            self.redis = None # Ensure redis is None if connection fails

    async def get(self, key: str) -> Optional[str]:
        if not self.redis: return None
        try:
            result = await self.circuit_breaker.call(self.redis.get, key)
            cache_operations.labels(operation="get", status="hit" if result else "miss").inc()
            return result.decode('utf-8') if result else None
        except Exception as e:
            cache_operations.labels(operation="get", status="error").inc()
            logger.warning(f"Cache get failed for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, ttl: int = 300):
        if not self.redis: return
        try:
            await self.circuit_breaker.call(self.redis.setex, key, ttl, value)
            cache_operations.labels(operation="set", status="success").inc()
        except Exception as e:
            cache_operations.labels(operation="set", status="error").inc()
            logger.warning(f"Cache set failed for key {key}: {e}")

    async def get_or_set(self, key: str, fetch_func, ttl: int = 300):
        cached_value = await self.get(key)
        if cached_value is not None:
            try: return json.loads(cached_value)
            except json.JSONDecodeError: return cached_value
        
        try:
            fetched_value = await fetch_func()
            await self.set(key, json.dumps(fetched_value, default=str), ttl)
            return fetched_value
        except Exception as e:
            logger.error(f"Cache get_or_set error for key {key}: {e}")
            return None

# Globally accessible instance
cache_service = CacheService(settings.redis_url)