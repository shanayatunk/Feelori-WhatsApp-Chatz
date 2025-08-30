# /app/utils/circuit_breaker.py

import asyncio
import time
import logging
from enum import Enum
from typing import Any, Callable

logger = logging.getLogger(__name__)

class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    def __init__(self, failure_threshold: int = 5, timeout: int = 60, success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if self.last_failure_time and (time.time() - self.last_failure_time > self.timeout):
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info(f"Circuit breaker is now HALF_OPEN for {func.__name__}")
                else:
                    logger.warning(f"Circuit breaker is OPEN. Call to {func.__name__} is blocked.")
                    raise Exception(f"Circuit breaker is OPEN for {func.__name__}")
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _on_success(self):
        async with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
                    logger.info("Circuit breaker has been reset to CLOSED.")
            else:
                self.failure_count = 0

    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker has OPENED due to {self.failure_count} failures.")

class RedisCircuitBreaker:
    def __init__(self, redis_client: Any, service_name: str, failure_threshold: int = 5, timeout: int = 60, success_threshold: int = 3):
        self.redis = redis_client
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_key = f"cb_failures:{service_name}"
        self.success_key = f"cb_success:{service_name}"
        self.state_key = f"cb_state:{service_name}"
        self.last_failure_key = f"cb_last_failure:{service_name}"

    async def is_open(self) -> bool:
        """Checks if the circuit is currently OPEN."""
        try:
            if not self.redis:
                return False  # If no Redis, assume closed
            state = await self._get_state()
            if state == "OPEN":
                last_failure_raw = await self.redis.get(self.last_failure_key)
                if last_failure_raw and time.time() - float(last_failure_raw) > self.timeout:
                    await self._set_state("HALF_OPEN")
                    return False
                return True
            return False
        except Exception as e:
            logger.error(f"Could not check circuit breaker state for {self.service_name}: {e}")
            return True  # Fail-safe: assume OPEN if we can't check

    async def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        if not self.redis:
            logger.warning(f"Redis client not available for circuit breaker '{self.service_name}'. Calling function directly.")
            return await func(*args, **kwargs)
        
        if await self.is_open():
             raise Exception(f"Circuit breaker is OPEN for {self.service_name}")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e

    async def _get_state(self) -> str:
        try:
            state = await self.redis.get(self.state_key)
            return state.decode() if state else "CLOSED"
        except Exception as e:
            logger.error(f"Failed to get circuit breaker state for {self.service_name}: {e}")
            return "CLOSED"

    async def _set_state(self, state: str):
        try:
            await self.redis.set(self.state_key, state, ex=self.timeout * 2)
        except Exception as e:
            logger.error(f"Failed to set circuit breaker state for {self.service_name}: {e}")

    async def _on_success(self):
        try:
            state = await self._get_state()
            if state == "HALF_OPEN":
                success_count = await self.redis.incr(self.success_key)
                if success_count >= self.success_threshold:
                    await self._set_state("CLOSED")
                    await self.redis.delete(self.failure_key)
                    await self.redis.delete(self.success_key)
                    logger.info(f"Circuit breaker has been reset to CLOSED for service: {self.service_name}")
            else:
                await self.redis.delete(self.failure_key)
        except Exception as e:
            logger.error(f"Error in circuit breaker success handler for {self.service_name}: {e}")

    async def _on_failure(self):
        try:
            failure_count = await self.redis.incr(self.failure_key)
            await self.redis.set(self.last_failure_key, str(time.time()), ex=self.timeout * 2)
            
            if failure_count >= self.failure_threshold:
                await self._set_state("OPEN")
                logger.error(f"Circuit breaker has OPENED for service '{self.service_name}' due to {failure_count} failures.")
        except Exception as e:
            logger.error(f"Error in circuit breaker failure handler for {self.service_name}: {e}")