# /app/services/security_service.py

import hmac
import hashlib
import bcrypt
import re
from fastapi import Request, HTTPException

from app.services.cache_service import cache_service
from app.utils.request_utils import get_remote_address

# This service provides core security functionalities, such as password hashing,
# webhook signature verification, input sanitization, and tracking login attempts.

class SecurityService:
    @staticmethod
    def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
        if not signature or not signature.startswith('sha256='): 
            return False
        expected_signature = hmac.new(secret.encode('utf-8'), payload, hashlib.sha256).hexdigest()
        return hmac.compare_digest(expected_signature, signature[7:])
    
    @staticmethod
    def hash_password(password: str) -> str:
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """
        Verifies a password against a hash.
        This is now wrapped in a try-except block to prevent server crashes
        if the provided 'hashed' value is invalid or None.
        """
        # --- THIS IS THE FIX ---
        try:
            # The 'hashed' password must be encoded to bytes for bcrypt
            return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
        except (ValueError, TypeError):
            # If hashed is None, empty, or not a valid hash, bcrypt will raise
            # an error. We catch it and safely return False.
            return False
        # --- END OF FIX ---

class EnhancedSecurityService(SecurityService):
    @staticmethod
    def sanitize_phone_number(phone: str) -> str:
        """
        Sanitizes a phone number.
        - Returns a normalized E.164-style string (e.g., +919876543210) if valid.
        - Returns an empty string for invalid or empty inputs (instead of raising).
        """
        if not phone or not isinstance(phone, str):
            return ""

        # Remove all characters except digits and leading +
        clean_phone = re.sub(r"[^\d+]", "", phone.strip())

        # Ensure it starts with +
        if not clean_phone.startswith("+"):
            clean_phone = "+" + clean_phone.lstrip("+")

        # Require 10–15 digits after +
        if not re.match(r"^\+\d{10,15}$", clean_phone):
            return ""

        return clean_phone

    @staticmethod
    def validate_message_content(message: str) -> str:
        if len(message) > 4096: 
            raise ValueError("Message too long")
        return message.strip()

    @staticmethod
    def validate_admin_session(request: Request, payload: dict):
        token_ip = payload.get("ip")
        current_ip = get_remote_address(request)
        if token_ip and token_ip != current_ip:
            raise HTTPException(status_code=401, detail="Token IP mismatch - please login again")

# --- Rate Limiting & Login Tracking ---
class RedisLoginAttemptTracker:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.lockout_duration = 900
        self.max_attempts = 5

    async def is_locked_out(self, ip: str) -> bool:
        if not self.redis: 
            return False
        key = f"login_attempts:{ip}"
        attempts = await self.redis.get(key)
        return attempts and int(attempts) >= self.max_attempts

    async def record_attempt(self, ip: str):
        if not self.redis: 
            return
        key = f"login_attempts:{ip}"
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, self.lockout_duration)

class AdvancedRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client

    async def check_phone_rate_limit(self, phone_number: str, limit: int = 10, window: int = 60) -> bool:
        if not self.redis: 
            return True
        key = f"rate_limit:phone:{phone_number}"
        current_count = await self.redis.incr(key)
        if current_count == 1: 
            await self.redis.expire(key, window)
        return current_count <= limit

    async def check_ip_rate_limit(self, ip_address: str, limit: int = 50, window: int = 60) -> bool:
        if not self.redis: 
            return True
        key = f"rate_limit:ip:{ip_address}"
        current_count = await self.redis.incr(key)
        if current_count == 1: 
            await self.redis.expire(key, window)
        return current_count <= limit

# Globally accessible instances
login_tracker = RedisLoginAttemptTracker(cache_service.redis)
rate_limiter = AdvancedRateLimiter(cache_service.redis)