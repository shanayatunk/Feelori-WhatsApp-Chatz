# /app/utils/rate_limiter.py

from slowapi import Limiter
from app.utils.request_utils import get_remote_address
from app.config.settings import settings

# This file centralizes the rate limiter instance to prevent circular imports.
# Both the main app and the route files will import the 'limiter' from here.

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"]
)