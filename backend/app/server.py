# ✅ TODO / Migration Notes
# - This is the final v1 code with preparations for v2 migration.
# - Order lookup and webhook logic preserved as v1 (using Shopify webhooks and minimal Mongo storage).
# - For v2: Migrate to full Mongo schema for orders/customers (add indexes for new fields like fulfillments).
# - WhatsApp Catalog Enhancements: Marked with # WHATSAPP CATALOG ENHANCEMENT comments.
# - Product Search Toggle: Controlled via PRODUCT_SEARCH_SOURCE in .env ("storefront" for Storefront API, "admin" for Admin API).
# - No functionality lost: All original endpoints, services, and handlers preserved.
# - Startup Logs: Added for product search mode and API endpoint.
# - Add to .env: PRODUCT_SEARCH_SOURCE=storefront/admin, WHATSAPP_BUSINESS_ACCOUNT_ID=your_waba_id
# - Ensure Shopify Storefront Access Token is set for storefront mode.
# - Webhook routes for v2 ready but inactive (commented; uncomment when switching to v2 full Mongo).
# - Cleaned-up Admin GraphQL search: Inline query added.
# - Storefront search: Includes SKU + image as per updates.

import os
import sys
import logging
import json
import hashlib
import uuid
import re
import asyncio
import secrets
import time
import hmac
import base64
import difflib
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any, Annotated, Tuple, Set, Union
from enum import Enum
from dataclasses import dataclass, field
from fastapi.staticfiles import StaticFiles
import html
import httpx
import redis.asyncio as redis
import bcrypt
import structlog
import tenacity
import redis as redis_package
from collections import defaultdict
from jose import JWTError, jwt
from fastapi import FastAPI, HTTPException, Request, Depends, Security, status, APIRouter, Query
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.sessions import SessionMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, OAuth2PasswordBearer
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
from openai import AsyncOpenAI
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from rapidfuzz import fuzz, process
import google.generativeai as genai
import psutil
import signal


# Load environment variables first
from dotenv import load_dotenv

load_dotenv()

# ==================== CONFIGURATION ====================
class Settings(BaseSettings):
    mongo_atlas_uri: str

    whatsapp_access_token: str
    whatsapp_phone_id: str
    whatsapp_verify_token: str
    whatsapp_webhook_secret: str
    whatsapp_catalog_id: Optional[str] = None
    whatsapp_business_account_id: Optional[str] = None

    shopify_store_url: str = "feelori.myshopify.com"
    shopify_access_token: str
    shopify_webhook_secret: Optional[str] = None
    shopify_storefront_access_token: Optional[str] = None

    gemini_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None

    packing_dept_whatsapp_number: Optional[str] = None
    packing_executive_names: str = "Default User"

    dashboard_url: str = "https://d38224fb4c5e.ngrok-free.app/static/dashboard.html"

    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_access_token_expire_hours: int = 24
    admin_password: str
    session_secret_key: str
    api_key: Optional[str] = None

    https_only: bool = True
    ssl_cert_path: Optional[str] = None
    ssl_key_path: Optional[str] = None
    
    redis_url: str = "redis://localhost:6379"
    redis_ssl: bool = False
    
    cors_allowed_origins: str = Field(default="https://feelori.com", env="CORS_ALLOWED_ORIGINS")
    allowed_hosts: str = Field(default="feelori.com,*.feelori.com", env="ALLOWED_HOSTS")
    
    max_pool_size: int = 10
    min_pool_size: int = 1
    mongo_ssl: bool = True
    
    workers: int = 4
    worker_class: str = "uvicorn.workers.UvicornWorker"
    
    sentry_dsn: Optional[str] = None
    sentry_environment: str = "production"
    
    alerting_webhook_url: Optional[str] = None

    jaeger_agent_host: str = "localhost"
    jaeger_agent_port: int = 6831
    
    environment: str = Field(default="production", env="ENVIRONMENT")
    
    api_version: str = "v1"
    
    rate_limit_per_minute: int = 100
    auth_rate_limit_per_minute: int = 5
    
    class Config:
        env_file = '.env.test' if os.path.exists('.env.test') else '.env'
        env_file_encoding = 'utf-8'

def validate_environment():
    try:
        settings = Settings()
        if len(settings.jwt_secret_key) < 32:
            raise ValueError("JWT_SECRET_KEY must be at least 32 characters long")
        if len(settings.session_secret_key) < 32:
            raise ValueError("SESSION_SECRET_KEY must be at least 32 characters long")
        if len(settings.whatsapp_webhook_secret) < 16:
            raise ValueError("WHATSAPP_WEBHOOK_SECRET must be at least 16 characters long")
        if not settings.whatsapp_verify_token:
            raise ValueError("WHATSAPP_VERIFY_TOKEN is required")
        if not settings.gemini_api_key and not settings.openai_api_key:
            raise ValueError("At least one AI API key (GEMINI_API_KEY or OPENAI_API_KEY) must be provided")
        if not re.match(r'^\d+$', settings.whatsapp_phone_id):
            raise ValueError("WHATSAPP_PHONE_ID must contain only digits")
        if len(settings.admin_password) < 12:
            raise ValueError("ADMIN_PASSWORD must be at least 12 characters long")
        required_vars = [
            'MONGO_ATLAS_URI', 'WHATSAPP_ACCESS_TOKEN', 'SHOPIFY_ACCESS_TOKEN'
        ]
        for var in required_vars:
            if not getattr(settings, var.lower()):
                raise ValueError(f"{var} is required for production")
        return settings
    except Exception as e:
        print(f"--- [ERROR] Environment validation failed: {str(e)}")
        sys.exit(1)

settings = validate_environment()

# ==================== JWT SERVICE ====================
class JWTService:
    def __init__(self, secret_key: str, algorithm: str = "HS256"):
        self.secret_key = secret_key
        self.algorithm = algorithm
    
    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=settings.jwt_access_token_expire_hours)
        
        to_encode.update({
            "exp": expire,
            "iat": datetime.utcnow(),
            "jti": str(uuid.uuid4())  # JWT ID for token tracking
        })
        
        return jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
    
    def verify_token(self, token: str) -> dict:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )

jwt_service = JWTService(settings.jwt_secret_key, settings.jwt_algorithm)

# ==================== SECURITY SERVICES ====================
class SecurityService:
    @staticmethod
    def verify_webhook_signature(payload: bytes, signature: str, secret: str) -> bool:
        """Verify WhatsApp webhook signature"""
        if not signature or not signature.startswith('sha256='):
            return False
        
        expected_signature = hmac.new(
            secret.encode('utf-8'),
            payload,
            hashlib.sha256
        ).hexdigest()
        
        received_signature = signature[7:]  # Remove 'sha256=' prefix
        return hmac.compare_digest(expected_signature, received_signature)
    
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash password with bcrypt"""
        return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
    
    @staticmethod
    def verify_password(password: str, hashed: str) -> bool:
        """Verify password against hash"""
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))

# Enhanced Security Service
class EnhancedSecurityService(SecurityService):
    @staticmethod
    def sanitize_phone_number(phone: str) -> str:
        """Enhanced phone number sanitization"""
        if not phone or len(phone) > 20:
            raise ValueError("Invalid phone number length")
        
        # Remove all non-digit characters except +
        clean_phone = re.sub(r'[^\d+]', '', phone.strip())
        
        # Ensure it starts with +
        if not clean_phone.startswith('+'):
            clean_phone = '+' + clean_phone.lstrip('+')
        
        # Validate format
        if not re.match(r'^\+\d{10,15}$', clean_phone):
            raise ValueError("Invalid phone number format")
        
        return clean_phone
    
    @staticmethod
    def validate_message_content(message: str) -> str:
        """Validate and sanitize message content"""
        if not isinstance(message, str):
            raise ValueError("Message must be a string")
        
        # Length validation
        if len(message) > 4096:  # WhatsApp limit
            raise ValueError("Message too long")
        
        if len(message.strip()) == 0:
            raise ValueError("Message cannot be empty")
        
        # Basic content filtering (expand as needed)
        suspicious_patterns = [
            r'<script.*?>.*?</script>',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
        ]
        
        message_lower = message.lower()
        for pattern in suspicious_patterns:
            if re.search(pattern, message_lower, re.IGNORECASE):
                logger.warning("suspicious_message_content", 
                             pattern=pattern, 
                             message_preview=message[:100])
                raise ValueError("Suspicious message content detected")
        
        return message.strip()
    
    @staticmethod
    def validate_admin_session(request: Request, payload: dict):
        """Validate admin session security"""
        # IP binding validation
        token_ip = payload.get("ip")
        current_ip = get_remote_address(request)
        
        if token_ip and token_ip != current_ip:
            logger.warning("ip_mismatch_detected", 
                         token_ip=token_ip, 
                         current_ip=current_ip)
            raise HTTPException(
                status_code=401,
                detail="Token IP mismatch - please login again"
            )
        
        # Token age validation (additional security)
        issued_at = payload.get("iat")
        if issued_at:
            token_age = datetime.utcnow().timestamp() - issued_at
            max_age = settings.jwt_access_token_expire_hours * 3600
            
            if token_age > max_age:
                raise HTTPException(
                    status_code=401,
                    detail="Token expired"
                )

# Hash admin password for comparison
ADMIN_PASSWORD_HASH = None # Will be set on app startup

# ==================== LOGGING ====================
def setup_logging():
    """Setup structured logging"""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer() if os.getenv("ENVIRONMENT") == "development" 
            else structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

setup_logging()
logger = structlog.get_logger(__name__)

# ==================== SENTRY INITIALIZATION ====================
def initialize_sentry():
    """Initialize Sentry for error tracking"""
    if settings.sentry_dsn:
        try:
            import sentry_sdk
            from sentry_sdk.integrations.fastapi import FastApiIntegration
            from sentry_sdk.integrations.httpx import HttpxIntegration
            from sentry_sdk.integrations.redis import RedisIntegration
            from sentry_sdk.integrations.pymongo import PyMongoIntegration
            
            sentry_sdk.init(
                dsn=settings.sentry_dsn,
                environment=settings.sentry_environment,
                traces_sample_rate=0.1,  # Adjust based on traffic
                profiles_sample_rate=0.1,  # Adjust based on traffic
                integrations=[
                    FastApiIntegration(auto_enabling_integrations=False),
                    HttpxIntegration(),
                    RedisIntegration(),
                    PyMongoIntegration(),
                ],
                before_send=lambda event, hint: event if event.get("level") != "info" else None,
                attach_stacktrace=True,
                send_default_pii=False,  # Important for privacy
            )
            logger.info("sentry_initialized", environment=settings.sentry_environment)
            return True
        except Exception as e:
            logger.error("sentry_initialization_failed", error=str(e))
            return False
    return False

# ==================== TRACING SETUP ====================
def setup_tracing():
    """Setup OpenTelemetry tracing with configurable Jaeger"""
    if settings.jaeger_agent_host:
        try:
            from opentelemetry import trace
            from opentelemetry.exporter.jaeger.thrift import JaegerExporter
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor
            
            trace.set_tracer_provider(TracerProvider())
            
            jaeger_exporter = JaegerExporter(
                agent_host_name=settings.jaeger_agent_host,
                agent_port=settings.jaeger_agent_port,
            )
            
            span_processor = BatchSpanProcessor(jaeger_exporter)
            trace.get_tracer_provider().add_span_processor(span_processor)
            
            logger.info("tracing_initialized", 
                       jaeger_host=settings.jaeger_agent_host,
                       jaeger_port=settings.jaeger_agent_port)
            return True
        except Exception as e:
            logger.error("tracing_initialization_failed", error=str(e))
            return False
    return False

# ==================== METRICS ====================
# Business Logic Metrics
message_counter = Counter('whatsapp_messages_total', 'Total messages processed', ['status', 'message_type'])
response_time_histogram = Histogram('response_time_seconds', 'Response time in seconds', ['endpoint'])
active_customers_gauge = Gauge('active_customers', 'Number of active customers')
ai_requests_counter = Counter('ai_requests_total', 'Total AI requests', ['model', 'status'])
database_operations_counter = Counter('database_operations_total', 'Database operations', ['operation', 'status'])

# Security Metrics
auth_attempts_counter = Counter('auth_attempts_total', 'Authentication attempts', ['status', 'method'])
webhook_signature_counter = Counter('webhook_signature_verifications_total', 'Webhook signature verifications', ['status'])

# Performance Metrics
cache_operations = Counter('cache_operations_total', 'Cache operations', ['operation', 'status'])
product_searches_total = Counter('product_searches_total', 'Total product searches performed')
product_searches_with_results = Counter('product_searches_with_results_total', 'Product searches that returned results')
product_searches_no_results = Counter('product_searches_no_results_total', 'Product searches that returned no results')

# ==================== CIRCUIT BREAKER ====================
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
        self.last_failure_time = None
        self.state = CircuitState.CLOSED
        self._lock = asyncio.Lock()
    
    async def call(self, func, *args, **kwargs):
        async with self._lock:
            if self.state == CircuitState.OPEN:
                if time.time() - self.last_failure_time > self.timeout:
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                    logger.info("circuit_breaker_half_open", func=func.__name__)
                else:
                    logger.warning("circuit_breaker_blocked", func=func.__name__)
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
                    logger.info("circuit_breaker_closed")
            else:
                self.failure_count = 0
    
    async def _on_failure(self):
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN
                logger.error("circuit_breaker_opened", failure_count=self.failure_count)

# ==================== REDIS CIRCUIT BREAKER ====================
class RedisCircuitBreaker:
    """Redis-backed circuit breaker for multi-worker environments"""
    
    def __init__(self, redis_client, service_name: str, failure_threshold: int = 5, 
                 timeout: int = 60, success_threshold: int = 3):
        self.redis = redis_client
        self.service_name = service_name
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.success_threshold = success_threshold
        self.failure_key = f"cb_failures:{service_name}"
        self.success_key = f"cb_success:{service_name}"
        self.state_key = f"cb_state:{service_name}"
        self.last_failure_key = f"cb_last_failure:{service_name}"
    
    async def call(self, func, *args, **kwargs):
        state = await self._get_state()
        
        if state == "OPEN":
            last_failure = await self.redis.get(self.last_failure_key)
            if last_failure and time.time() - float(last_failure) > self.timeout:
                await self._set_state("HALF_OPEN")
                await self.redis.delete(self.success_key)
                logger.info("circuit_breaker_half_open", service=self.service_name)
            else:
                logger.warning("circuit_breaker_blocked", service=self.service_name)
                raise Exception(f"Circuit breaker is OPEN for {self.service_name}")
        
        try:
            result = await func(*args, **kwargs)
            await self._on_success()
            return result
        except Exception as e:
            await self._on_failure()
            raise e
    
    async def _get_state(self) -> str:
        state = await self.redis.get(self.state_key)
        return state.decode() if state else "CLOSED"
    
    async def _set_state(self, state: str):
        await self.redis.set(self.state_key, state, ex=self.timeout * 2)
    
    async def _on_success(self):
        state = await self._get_state()
        
        if state == "HALF_OPEN":
            success_count = await self.redis.incr(self.success_key)
            if success_count >= self.success_threshold:
                await self._set_state("CLOSED")
                await self.redis.delete(self.failure_key)
                await self.redis.delete(self.success_key)
                logger.info("circuit_breaker_closed", service=self.service_name)
        else:
            await self.redis.delete(self.failure_key)
    
    async def _on_failure(self):
        failure_count = await self.redis.incr(self.failure_key)
        await self.redis.set(self.last_failure_key, str(time.time()), ex=self.timeout * 2)
        
        if failure_count >= self.failure_threshold:
            await self._set_state("OPEN")
            logger.error("circuit_breaker_opened", 
                        service=self.service_name, 
                        failure_count=failure_count)

# ==================== REDIS LOGIN ATTEMPT TRACKER ====================
class RedisLoginAttemptTracker:
    """Redis-backed login attempt tracker for multi-worker environments"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.lockout_duration = 900  # 15 minutes
        self.max_attempts = 5
    
    async def is_locked_out(self, ip: str) -> bool:
        key = f"login_attempts:{ip}"
        attempts = await self.redis.lrange(key, 0, -1)
        
        # Clean old attempts and count recent ones
        now = time.time()
        recent_attempts = []
        
        for attempt in attempts:
            attempt_time = float(attempt.decode())
            if now - attempt_time < self.lockout_duration:
                recent_attempts.append(attempt_time)
        
        # Update Redis with only recent attempts
        if len(recent_attempts) != len(attempts):
            await self.redis.delete(key)
            if recent_attempts:
                await self.redis.lpush(key, *recent_attempts)
                await self.redis.expire(key, self.lockout_duration)
        
        return len(recent_attempts) >= self.max_attempts
    
    async def record_attempt(self, ip: str):
        key = f"login_attempts:{ip}"
        await self.redis.lpush(key, str(time.time()))
        await self.redis.expire(key, self.lockout_duration)

# ==================== RATE LIMITING ====================
class AdvancedRateLimiter:
    def __init__(self, redis_client):
        self.redis = redis_client
    
    async def check_phone_rate_limit(self, phone_number: str, 
                                   limit: int = 10, window: int = 60) -> bool:
        """Rate limit per phone number"""
        key = f"rate_limit:phone:{phone_number}"
        
        try:
            # Sliding window rate limiting
            now = time.time()
            pipeline = self.redis.pipeline()
            
            # Remove old entries
            pipeline.zremrangebyscore(key, 0, now - window)
            
            # Count current requests in window
            pipeline.zcard(key)
            
            # Add current request
            pipeline.zadd(key, {str(uuid.uuid4()): now})
            
            # Set expiry
            pipeline.expire(key, window)
            
            results = await pipeline.execute()
            current_count = results[1]
            
            if current_count >= limit:
                logger.warning("phone_rate_limit_exceeded", 
                             phone=phone_number, 
                             count=current_count, 
                             limit=limit)
                return False
            
            return True
            
        except Exception as e:
            logger.error("rate_limit_check_error", error=str(e))
            return True  # Allow on error (fail open)
    
    async def check_ip_rate_limit(self, ip_address: str, 
                                limit: int = 50, window: int = 60) -> bool:
        """Rate limit per IP address"""
        key = f"rate_limit:ip:{ip_address}"
        
        try:
            current_count = await self.redis.incr(key)
            if current_count == 1:
                await self.redis.expire(key, window)
            
            if current_count > limit:
                logger.warning("ip_rate_limit_exceeded", 
                             ip=ip_address, 
                             count=current_count, 
                             limit=limit)
                return False
            
            return True
            
        except Exception as e:
            logger.error("ip_rate_limit_check_error", error=str(e))
            return True  # Allow on error

# ==================== ALERTING SERVICE ====================
class AlertingService:
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url
        self.client = httpx.AsyncClient(timeout=5.0) if webhook_url else None
    
    async def send_critical_alert(self, error: str, context: Dict[str, Any]):
        """Send critical alerts to external systems"""
        if not self.webhook_url or not self.client:
            logger.warning("alert_webhook_not_configured", error=error)
            return
        
        try:
            alert_data = {
                "severity": "critical",
                "service": "feelori-whatsapp-assistant",
                "error": error,
                "context": context,
                "timestamp": datetime.utcnow().isoformat(),
                "environment": os.getenv("ENVIRONMENT", "production")
            }
            
            await self.client.post(
                self.webhook_url,
                json=alert_data,
                timeout=5.0
            )
            
            logger.info("critical_alert_sent", error=error)
        except Exception as e:
            logger.error("failed_to_send_alert", error=str(e))
    
    async def cleanup(self):
        if self.client:
            await self.client.aclose()

alerting = AlertingService(settings.alerting_webhook_url)

# ==================== MODELS ====================
def validate_phone_number(phone: str) -> str:
    """Enhanced phone number validation"""
    clean_phone = re.sub(r'[^\d+]', '', phone)
    if not clean_phone.startswith('+'):
        clean_phone = '+' + clean_phone
    if not re.match(r'^\+\d{10,15}$', clean_phone):
        raise ValueError("Invalid phone number format")
    return clean_phone

class LoginRequest(BaseModel):
    password: str = Field(..., min_length=12, max_length=255)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class BroadcastRequest(BaseModel):
    message: str
    target_type: str = "all"
    target_phones: Optional[List[str]] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str = Field(default=settings.api_version)

class HoldOrderRequest(BaseModel):
    reason: str
    # CHANGE THIS LINE
    problem_item_skus: Optional[List[str]] = None # Changed from singular to plural to accept a list
    notes: Optional[str] = None

class FulfillOrderRequest(BaseModel):
    packer_name: str
    tracking_number: str
    carrier: str

class Product(BaseModel):
    id: str
    title: str
    description: str
    price: float
    variant_id: str
    sku: Optional[str] = None
    currency: str = "INR"
    image_url: Optional[str] = None
    availability: str = "in_stock"
    tags: List[str] = []
    handle: str = ""  # Added for URL generation

# ==================== DATABASE SERVICE ====================
class DatabaseService:
    def __init__(self, mongo_uri: str):
        self.client = AsyncIOMotorClient(
            mongo_uri,
            maxPoolSize=settings.max_pool_size,
            minPoolSize=settings.min_pool_size,
            maxIdleTimeMS=30000,
            waitQueueTimeoutMS=5000,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
            socketTimeoutMS=10000,
            retryWrites=True,
            readPreference='secondaryPreferred'
        )
        self.db = self.client.get_default_database()
        self.circuit_breaker = CircuitBreaker()
    
    async def create_indexes(self):
        """Create all necessary database indexes on startup."""
        try:
            # Orders indexes for faster lookups (V2 Readiness)
            await self.db.orders.create_index([("id", 1)], unique=True)
            await self.db.orders.create_index([("order_number", 1)])
            await self.db.orders.create_index([("phone_numbers", 1)])
            await self.db.orders.create_index([("created_at", -1)])

            # Customer indexes
            await self.db.customers.create_index("phone_number", unique=True)
            await self.db.customers.create_index("last_interaction")
            await self.db.customers.create_index("created_at")
            await self.db.customers.create_index("conversation_history.timestamp")

            # Security events indexes
            await self.db.security_events.create_index("event_type")
            await self.db.security_events.create_index([("timestamp", -1)]) # Use timestamp from event
            
            logger.info("database_indexes_created")
        except Exception as e:
            logger.error("failed_to_create_indexes", error=str(e))
    
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type(Exception),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    async def get_customer(self, phone_number: str):
        """Get customer with retry logic."""
        return await self.circuit_breaker.call(
            self.db.customers.find_one, {"phone_number": phone_number}
        )

    async def get_customer(self, phone_number: str):
        """Get customer with retry logic"""
        return await self.circuit_breaker.call(
            self.db.customers.find_one, {"phone_number": phone_number}
        )
    
    async def create_customer(self, customer_data: dict):
        """Create new customer"""
        try:
            result = await self.circuit_breaker.call(
                self.db.customers.insert_one, customer_data
            )
            database_operations_counter.labels(operation="create_customer", status="success").inc()
            return result
        except Exception as e:
            database_operations_counter.labels(operation="create_customer", status="error").inc()
            logger.error("create_customer_error", error=str(e))
            raise
    
    async def update_conversation_history(self, phone_number: str, message: str, response: str, wamid: Optional[str] = None):
        """Update customer conversation history, including the WhatsApp message ID."""
        try:
            conversation_entry = {
                "timestamp": datetime.utcnow(),
                "message": message,
                "response": response,
                "wamid": wamid,      # CHANGE: Store the ID
                "status": "sent" if wamid else None  # CHANGE: Set initial status
            }
            
            await self.circuit_breaker.call(
                self.db.customers.update_one,
                {"phone_number": phone_number},
                {
                    "$push": {"conversation_history": conversation_entry},
                    "$set": {"last_interaction": datetime.utcnow()}
                }
            )
            database_operations_counter.labels(operation="update_conversation", status="success").inc()
        except Exception as e:
            database_operations_counter.labels(operation="update_conversation", status="error").inc()
            logger.error("update_conversation_error", error=str(e))
    
    async def log_security_event(self, event_type: str, ip_address: str, details: dict):
        """Log security events"""
        try:
            event_data = {
                "event_type": event_type,
                "ip_address": ip_address,
                "timestamp": datetime.utcnow(),
                "details": details
            }
            
            await self.circuit_breaker.call(
                self.db.security_events.insert_one, event_data
            )
            database_operations_counter.labels(operation="log_security", status="success").inc()
        except Exception as e:
            database_operations_counter.labels(operation="log_security", status="error").inc()
            logger.error("log_security_event_error", error=str(e))

# ==================== MIGRATION SERVICE ====================
class MigrationService:
    def __init__(self, db_service):
        self.db = db_service.db

    async def get_schema_version(self):
        version_doc = await self.db.schema_version.find_one({"_id": "schema"})
        return version_doc.get("version", 0) if version_doc else 0

    def get_pending_migrations(self, current_version):
        # Define migrations as list of dicts with version and execute func
        migrations = [
            {"version": 1, "execute": self.migration_1},
            # Add more migrations as needed
        ]
        return [m for m in migrations if m["version"] > current_version]

    async def update_schema_version(self, version: int):
        await self.db.schema_version.update_one(
            {"_id": "schema"},
            {"$set": {"version": version}},
            upsert=True
        )

    async def migration_1(self):
        # Example migration: add a new index
        await self.db.customers.create_index("new_field")

    async def run_migrations(self):
        current_version = await self.get_schema_version()
        migrations = self.get_pending_migrations(current_version)
        for migration in migrations:
            await migration["execute"]()
            await self.update_schema_version(migration["version"])
        logger.info("migrations_completed", applied=len(migrations))

# ==================== DATABASE HEALTH MONITOR ====================
class DatabaseHealthMonitor:
    def __init__(self, db_service, circuit_breaker):
        self.db = db_service.db
        self.circuit_breaker = circuit_breaker

    async def monitor_connection(self):
        while True:
            try:
                await self.db.command("ping")
                await self.circuit_breaker._on_success()
            except Exception as e:
                logger.error("db_health_check_failed", error=str(e))
                await self.circuit_breaker._on_failure()
            await asyncio.sleep(30)

# ==================== RESOURCE MONITOR ====================
class ResourceMonitor:
    def __init__(self):
        self.memory_threshold = 80  # 80% memory usage threshold
    
    async def check_memory_usage(self):
        while True:
            memory_percent = psutil.virtual_memory().percent
            if memory_percent > self.memory_threshold:
                logger.warning("high_memory_usage", percent=memory_percent)
                await self.trigger_memory_cleanup()
            await asyncio.sleep(60)  # Check every minute
    
    async def trigger_memory_cleanup(self):
        # Implement cleanup logic, e.g., clear caches
        pass

# ==================== CACHE SERVICE ====================
class CacheService:
    def __init__(self, redis_url: str):
        self.redis_pool = redis.ConnectionPool.from_url(
            redis_url, 
            max_connections=20
        )
        self.redis = redis.Redis(connection_pool=self.redis_pool)
        self.circuit_breaker = CircuitBreaker()
    
    async def get(self, key: str) -> Optional[str]:
        try:
            result = await self.circuit_breaker.call(self.redis.get, key)
            cache_operations.labels(operation="get", status="hit" if result else "miss").inc()
            return result.decode('utf-8') if result else None
        except Exception as e:
            cache_operations.labels(operation="get", status="error").inc()
            logger.warning("cache_get_failed", key=key, error=str(e))
            return None
    
    async def set(self, key: str, value: str, ttl: int = 300):
        try:
            await self.circuit_breaker.call(self.redis.setex, key, ttl, value)
            cache_operations.labels(operation="set", status="success").inc()
        except Exception as e:
            cache_operations.labels(operation="set", status="error").inc()
            logger.warning("cache_set_failed", key=key, error=str(e))
    
    async def get_or_set(self, key: str, fetch_func, ttl: int = 300):
        """Get from cache or fetch and set"""
        cached_value = await self.get(key)
        if cached_value is not None:
            try:
                return json.loads(cached_value)
            except json.JSONDecodeError:
                return cached_value
        
        # Fetch new value
        try:
            fetched_value = await fetch_func()
            await self.set(key, json.dumps(fetched_value, default=str), ttl)
            return fetched_value
        except Exception as e:
            logger.error("cache_get_or_set_error", key=key, error=str(e))
            return None

# ==================== STATE MANAGEMENT SERVICE ====================
class StateManagementService:
    def __init__(self, redis_client):
        self.redis = redis_client
        self.ttl = 900  # State will be remembered for 15 minutes

    async def set_last_single_product(self, phone_number: str, product: Product):
        """Saves the last SINGLE product viewed by a user (e.g., from 'View Details')."""
        key = f"state:last_single_product:{phone_number}"
        await self.redis.set(key, product.json(), ex=self.ttl)
        # Clear any previous list to avoid ambiguity
        await self.redis.delete(f"state:last_product_list:{phone_number}")
        logger.info("state_set_single_product", phone=phone_number, product_id=product.id)

    async def get_last_single_product(self, phone_number: str) -> Optional[Product]:
        """Retrieves the last SINGLE product viewed by a user."""
        key = f"state:last_single_product:{phone_number}"
        product_data = await self.redis.get(key)
        if product_data:
            return Product.parse_raw(product_data)
        return None

    async def set_last_product_list(self, phone_number: str, products: List[Product]):
        """Saves the last LIST of products shown to a user (from a search)."""
        key = f"state:last_product_list:{phone_number}"
        # Store as a JSON array of product dictionaries
        product_list_data = json.dumps([p.dict() for p in products])
        await self.redis.set(key, product_list_data, ex=self.ttl)
        # Clear any previous single product to avoid ambiguity
        await self.redis.delete(f"state:last_single_product:{phone_number}")
        logger.info("state_set_product_list", phone=phone_number, count=len(products))

    async def get_last_product_list(self, phone_number: str) -> Optional[List[Product]]:
        """Retrieves the last LIST of products shown to a user."""
        key = f"state:last_product_list:{phone_number}"
        product_list_data = await self.redis.get(key)
        if product_list_data:
            products_raw = json.loads(product_list_data)
            return [Product(**p) for p in products_raw]
        return None

    async def set_last_search(self, phone_number: str, query: str, page: int):
        """Saves the last product search query and page number."""
        key = f"state:last_search:{phone_number}"
        search_state = {"query": query, "page": page}
        await self.redis.set(key, json.dumps(search_state), ex=self.ttl)

    async def get_last_search(self, phone_number: str) -> Optional[Dict]:
        """Retrieves the last product search state."""
        key = f"state:last_search:{phone_number}"
        search_state = await self.redis.get(key)
        if search_state:
            return json.loads(search_state)
        return None
    async def set_last_bot_question(self, phone_number: str, question_type: str):
        """Saves the context of the last direct question the bot asked."""
        key = f"state:last_bot_question:{phone_number}"
        await self.redis.set(key, question_type, ex=self.ttl) # Use the same 15-min TTL
        logger.info("state_set_last_bot_question", phone=phone_number, question=question_type)

    async def get_last_bot_question(self, phone_number: str) -> Optional[str]:
        """Retrieves the context of the last direct question the bot asked."""
        key = f"state:last_bot_question:{phone_number}"
        question_type = await self.redis.get(key)
        return question_type.decode() if question_type else None

    async def clear_last_bot_question(self, phone_number: str):
        """Clears the last bot question state after it has been handled."""
        key = f"state:last_bot_question:{phone_number}"
        await self.redis.delete(key)

    async def set_pending_question(self, phone_number: str, question_type: str, context: Dict = None):
        """Saves a user's question that is pending more information."""
        key = f"state:pending_question:{phone_number}"
        state = {"question_type": question_type, "context": context or {}}
        await self.redis.set(key, json.dumps(state), ex=self.ttl)
        logger.info("state_set_pending_question", phone=phone_number, question=question_type)

    async def get_pending_question(self, phone_number: str) -> Optional[Dict]:
        """Retrieves and immediately clears a pending user question."""
        key = f"state:pending_question:{phone_number}"
        state_data = await self.redis.get(key)
        if state_data:
            await self.redis.delete(key) # The question is retrieved once and then cleared
            return json.loads(state_data)
        return None

    async def set_last_active_order(self, phone_number: str, order_id: int, order_number: str):
        """Sets the specific order the packing team is currently working on."""
        key = f"state:last_active_order:{phone_number}"
        value = {"order_id": order_id, "order_number": str(order_number)}
        await self.redis.set(key, json.dumps(value), ex=self.ttl)

    async def get_last_active_order(self, phone_number: str) -> Optional[Dict]:
        """Gets the specific order the packing team is currently working on."""
        key = f"state:last_active_order:{phone_number}"
        data = await self.redis.get(key)
        return json.loads(data) if data else None

    async def clear_last_active_order(self, phone_number: str):
        """Clears the active order state once the workflow is complete."""
        key = f"state:last_active_order:{phone_number}"
        await self.redis.delete(key)
# ==================== REDIS MESSAGE QUEUE ====================
# ==================== REDIS MESSAGE QUEUE ====================
class RedisMessageQueue:
    """Redis-backed message queue using Redis Streams"""

    def __init__(self, redis_client, stream_name: str = "webhook_messages", max_workers: int = 5):
        self.redis = redis_client
        self.stream_name = stream_name
        self.consumer_group = "webhook_processors"
        self.max_workers = max_workers
        self.workers = []
        self.running = False

    async def initialize(self):
        """Create consumer group if it doesn't exist"""
        try:
            await self.redis.xgroup_create(self.stream_name, self.consumer_group, id="0", mkstream=True)
        except redis_package.exceptions.ResponseError as e:
            if "BUSYGROUP Consumer Group name already exists" in str(e):
                logger.info("redis_consumer_group_exists", group=self.consumer_group)
                pass
            else:
                logger.error("redis_group_create_failed", error=str(e))
                raise

    async def start_workers(self):
        """Start message processing workers"""
        await self.initialize()
        self.running = True
        
        for i in range(self.max_workers):
            consumer_name = f"worker-{i}-{uuid.uuid4().hex[:8]}"
            worker = asyncio.create_task(self._worker(consumer_name))
            self.workers.append(worker)
        
        logger.info("redis_message_queue_workers_started", count=self.max_workers)

    async def stop_workers(self):
        """Stop message processing workers"""
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("redis_message_queue_workers_stopped")

    async def _worker(self, consumer_name: str):
        """Message processing worker using Redis Streams with self-healing for NOGROUP error."""
        while self.running:
            try:
                messages = await self.redis.xreadgroup(
                    self.consumer_group,
                    consumer_name,
                    {self.stream_name: ">"},
                    count=1,
                    block=1000
                )
                
                if messages:
                    stream_messages = messages[0][1]
                    for message_id, fields in stream_messages:
                        try:
                            message_data = {k.decode(): v.decode() for k, v in fields.items()}
                            message_data = json.loads(message_data.get('data', '{}'))
                            # ✅ FIX: This call now works because the method below exists.
                            await self._process_message(message_data)
                            await self.redis.xack(self.stream_name, self.consumer_group, message_id)
                        except Exception as e:
                            logger.error("redis_worker_message_error", worker=consumer_name, message_id=message_id.decode(), error=str(e))
            
            except redis_package.exceptions.ResponseError as e:
                if "NOGROUP" in str(e):
                    logger.warning("redis_worker_nogroup_error", detail="Stream/group not found, attempting to re-create...")
                    try:
                        await self.initialize()
                        await asyncio.sleep(1)
                    except Exception as init_e:
                        logger.error("redis_worker_init_failed", error=str(init_e))
                        await asyncio.sleep(5)
                else:
                    logger.error("redis_worker_response_error", worker=consumer_name, error=str(e))
                    await asyncio.sleep(5)

            except asyncio.CancelledError:
                break
            except Exception as e:
                if self.running:
                    logger.error("redis_worker_error", worker=consumer_name, error=str(e))
                    await asyncio.sleep(5)

    # ✅ FIX: ADD THIS METHOD BACK INTO THE CLASS
    # Inside the RedisMessageQueue class
    async def _process_message(self, message_data: Dict):
        """Process message, send response, and store the message ID."""
        try:
            from_number = message_data["from_number"]
            message_text = message_data["message_text"]
            message_type = message_data.get("message_type", "text")
            quoted_wamid = message_data.get("quoted_wamid")
            profile_name = message_data.get("profile_name") # Get the name

            # ✅ NEW: Update customer name if it's a new interaction
            customer = await get_or_create_customer(from_number)
            if profile_name and not customer.get("name"):
                await services.db_service.db.customers.update_one(
                    {"phone_number": from_number},
                    {"$set": {"name": profile_name}}
                )
                # Invalidate cache to reflect the name change
                await services.cache_service.redis.delete(f"customer:v2:{from_number}")

            response_text = await process_message(from_number, message_text, message_type, quoted_wamid)
            
            if response_text:
                wamid = await services.whatsapp_service.send_message(from_number, response_text)
                await update_conversation_history_safe(from_number, message_text, response_text, wamid)
                
                if wamid:
                    message_counter.labels(status="success", message_type=message_type).inc()
                else:
                    message_counter.labels(status="send_failed", message_type=message_type).inc()
        
        except Exception as e:
            logger.error("redis_message_processing_error", error=str(e))
            message_counter.labels(status="error", message_type="processing_error").inc()

    async def add_message(self, message_data: Dict):
        """Add message to Redis stream"""
        try:
            await self.redis.xadd(
                self.stream_name,
                {"data": json.dumps(message_data, default=str)},
                maxlen=10000
            )
        except Exception as e:
            logger.error("redis_add_message_error", error=str(e))

    async def is_duplicate_message(self, message_id: str, phone_number: str) -> bool:
        """Checks if a message ID has been processed recently to prevent duplicates."""
        key = f"processed_message:{phone_number}:{message_id}"
        if await self.redis.set(key, "1", ex=300, nx=True):
            return False
        else:
            return True

# ==================== WHATSAPP SERVICE ====================
class WhatsAppService:
    def __init__(self, access_token: str, phone_id: str, http_client: httpx.AsyncClient, business_account_id: Optional[str] = None):
        self.access_token = access_token
        self.phone_id = phone_id
        self.http_client = http_client
        self.base_url = "https://graph.facebook.com/v18.0"
        self.circuit_breaker = CircuitBreaker()
        self.whatsapp_business_account_id = business_account_id

    async def send_message(self, to_phone: str, message: str) -> Optional[str]:
        """Send text message via WhatsApp Business API and return the message ID (wamid)."""
        try:
            url = f"{self.base_url}/{self.phone_id}/messages"

            # Validate phone number format
            if not to_phone:
                logger.error("send_message_invalid_phone", phone=to_phone)
                return None

            # Clean phone number (remove any formatting)
            clean_phone = re.sub(r"[^\d+]", "", to_phone)
            if not clean_phone.startswith("+"):
                clean_phone = "+" + clean_phone.lstrip("+")

            # Prepare message payload
            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": clean_phone,
                "type": "text",
                "text": {
                    "body": message[:4096]  # WhatsApp message limit
                },
            }

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            # Send message through circuit breaker with retry
            response = await self.resilient_api_call(
                self.circuit_breaker.call,
                self.http_client.post, url, json=payload, headers=headers
            )

            # Check response status
            if response.status_code == 200:
                response_data = response.json()
                message_id = response_data.get("messages", [{}])[0].get("id")
                logger.info(
                    "whatsapp_message_sent",
                    to=clean_phone,
                    message_length=len(message),
                    wamid=message_id,
                )
                return message_id
            else:
                # Log detailed error information
                try:
                    error_data = response.json()
                    error_message = error_data.get("error", {}).get(
                        "message", "Unknown error"
                    )
                    error_code = error_data.get("error", {}).get(
                        "code", response.status_code
                    )
                except Exception:
                    error_message = response.text
                    error_code = response.status_code

                logger.error(
                    "whatsapp_send_failed",
                    to=clean_phone,
                    status_code=response.status_code,
                    error_code=error_code,
                    error_message=error_message,
                    response=response.text[:500],
                )

                # Handle specific error cases
                if response.status_code == 401:
                    logger.critical(
                        "whatsapp_auth_failed",
                        token_prefix=self.access_token[:10],
                    )
                    await alerting.send_critical_alert(
                        "WhatsApp authentication failed",
                        {"phone": clean_phone, "error": "Invalid access token"},
                    )
                elif response.status_code == 429:
                    logger.warning("whatsapp_rate_limited", to=clean_phone)
                elif response.status_code >= 500:
                    logger.error(
                        "whatsapp_server_error", status=response.status_code
                    )

                return None

        except asyncio.TimeoutError:
            logger.error("whatsapp_send_timeout", to=to_phone)
            return None
        except httpx.RequestError as e:
            logger.error("whatsapp_request_error", to=to_phone, error=str(e))
            return None
        except Exception as e:
            logger.error(
                "whatsapp_send_error",
                to=to_phone,
                error=str(e),
                error_type=type(e).__name__,
            )

            # Send critical alert for unexpected errors
            await alerting.send_critical_alert(
                "WhatsApp send message unexpected error",
                {
                    "phone": to_phone,
                    "error": str(e),
                    "error_type": type(e).__name__,
                },
            )
            return None

    async def get_catalog_id(self) -> Optional[str]:
        """
        Fetches the WhatsApp Business catalog ID from Meta Graph API.
        Caches the result for reuse to avoid repeated API calls.
        """
        if hasattr(self, "_catalog_id_cache") and self._catalog_id_cache:
            return self._catalog_id_cache

        if not hasattr(self, "whatsapp_business_account_id") or not self.whatsapp_business_account_id:
            logger.error("whatsapp_business_account_id_not_set")
            return None

        url = f"{self.base_url}/{self.whatsapp_business_account_id}/catalogs"
        params = {"access_token": self.access_token}

        try:
            resp = await self.http_client.get(url, params=params)
            data = resp.json()
            catalogs = data.get("data", [])
            if catalogs:
                self._catalog_id_cache = catalogs[0].get("id")
                logger.info("whatsapp_catalog_id_fetched", catalog_id=self._catalog_id_cache)
                return self._catalog_id_cache
            else:
                logger.warning("whatsapp_no_catalog_found")
                return None
        except Exception as e:
            logger.error("whatsapp_get_catalog_id_error", error=str(e))
            return None

    async def send_image_message(self, to_phone: str, image_url: str, caption: str):
        """Sends an image with a caption."""
        try:
            payload = {
                "messaging_product": "whatsapp",
                "to": to_phone,
                "type": "image",
                "image": {
                    "link": image_url,
                    "caption": caption
                }
            }
            # ✅ FIX: Added the missing authentication headers.
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/{self.phone_id}/messages"
            await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
            return True
        except Exception as e:
            logger.error("send_image_message_error", to=to_phone, error=str(e))
            return False


    async def send_multi_product_message(
        self,
        to: str,
        header_text: str,
        body_text: str,
        footer_text: str,
        catalog_id: str,
        section_title: str,
        product_items: list,
        fallback_products: list
    ):
        """
        Sends a WhatsApp Multi-Product Message.
        Falls back to image + button template if no catalog or missing SKUs.
        """
        # --- Main Logic: Try to send the rich multi-product message ---
        if catalog_id and product_items:
            try:
                payload = {
                    "messaging_product": "whatsapp", "to": to, "type": "interactive",
                    "interactive": {
                        "type": "product_list",
                        "header": {"type": "text", "text": header_text},
                        "body": {"text": body_text},
                        "footer": {"text": footer_text},
                        "action": {
                            "catalog_id": catalog_id,
                            "sections": [{"title": section_title, "product_items": product_items}]
                        }
                    }
                }
                url = f"{self.base_url}/{self.phone_id}/messages"
                headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
                
                response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    logger.info("whatsapp_multi_product_message_sent", to=to, count=len(product_items))
                    return # Success, we are done.
                else:
                    # If the main message fails, log it and proceed to fallback.
                    error_data = response.json().get("error", {})
                    logger.error(
                        "whatsapp_multi_product_send_failed",
                        status_code=response.status_code,
                        error=error_data.get("message", response.text)
                    )

            except Exception as e:
                logger.error("whatsapp_multi_product_send_error", error=str(e))

        # --- Fallback Logic: Send each product as an individual card ---
        logger.info("whatsapp_catalog_unavailable_using_fallback", to=to)
        for product in fallback_products:
            if not product.image_url:
                continue
            try:
                payload = {
                    "messaging_product": "whatsapp", "to": to, "type": "interactive",
                    "interactive": {
                        "type": "button",
                        "header": {"type": "image", "image": {"link": product.image_url}},
                        "body": {"text": f"{product.title}\nTap below to view details."},
                        "footer": {"text": footer_text},
                        "action": {
                            "buttons": [{"type": "reply", "reply": {"id": f"product_{product.id}", "title": "View Details"}}]
                        }
                    }
                }
                url = f"{self.base_url}/{self.phone_id}/messages"
                headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}

                # --- THIS IS THE CRITICAL FIX ---
                # We now check the response from the API call.
                response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    logger.info("whatsapp_fallback_message_sent", to=to, product_id=product.id)
                else:
                    # If the API returns an error, we log the detailed reason.
                    error_data = response.json().get("error", {})
                    logger.error(
                        "whatsapp_fallback_send_failed",
                        to=to,
                        status_code=response.status_code,
                        error_message=error_data.get("message", "Unknown error"),
                        details=error_data.get("error_data", {}),
                        response=response.text[:500]
                    )
                # --- END OF FIX ---
                
                await asyncio.sleep(0.5) # Small delay between messages
            except Exception as e:
                logger.error("whatsapp_button_template_error", error=str(e))

    async def send_interactive_list(
        self, to_phone: str, products: List["Product"], title: str = "Products"
    ) -> bool:
        """Send interactive list message."""
        try:
            if not products:
                return False

            url = f"{self.base_url}/{self.phone_id}/messages"

            # Build interactive list (max 10 items)
            sections = [
                {
                    "title": "Available Products",
                    "rows": [],
                }
            ]

            for product in products[:10]:  # WhatsApp limit
                sections[0]["rows"].append(
                    {
                        "id": f"product_{product.id}",
                        "title": product.title[:24],  # WhatsApp title limit
                        "description": f"₹{product.price:.2f}",
                    }
                )

            payload = {
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": to_phone,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "header": {"type": "text", "text": title},
                    "body": {"text": "Choose a product to learn more:"},
                    "footer": {"text": "Feelori - Your Fashion Assistant"},
                    "action": {
                        "button": "View Products",
                        "sections": sections,
                    },
                },
            }

            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }

            response = await self.resilient_api_call(
                self.circuit_breaker.call,
                self.http_client.post, url, json=payload, headers=headers
            )

            success = response.status_code == 200
            if success:
                logger.info(
                    "whatsapp_interactive_sent",
                    to=to_phone,
                    products_count=len(products),
                )
            else:
                logger.error(
                    "whatsapp_interactive_failed",
                    to=to_phone,
                    status_code=response.status_code,
                )

            return success

        except Exception as e:
            logger.error("whatsapp_interactive_error", to=to_phone, error=str(e))
            return False


    async def send_packer_selection_list(self, to_phone: str, order_number: str, order_id: str, executive_names: List[str]):
        """Sends an interactive list message for selecting a packer's name."""
        try:
            rows = []
            # The ID now includes the order_number for better context in the process_message function
            for name in executive_names[:10]: # WhatsApp lists are limited to 10 rows
                rows.append({
                    "id": f"packer_name_{name.strip()}_{order_id}_{order_number}",
                    "title": name.strip()
                })

            payload = {
                "messaging_product": "whatsapp",
                "to": to_phone,
                "type": "interactive",
                "interactive": {
                    "type": "list",
                    "header": {"type": "text", "text": f"Order #{order_number}"},
                    "body": {"text": "📦 The order is packed. Please select the name of the person who packed it."},
                    "footer": {"text": "Select one name"},
                    "action": {
                        "button": "Select Packer",
                        "sections": [{
                            "title": "Packing Team",
                            "rows": rows
                        }]
                    }
                }
            }

            # ✅ FIX: Added the missing authentication headers.
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/{self.phone_id}/messages"
            response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info("packer_selection_list_sent", to=to_phone, order_id=order_id)
                return True
            else:
                logger.error("packer_selection_list_failed", to=to_phone, status=response.status_code, response=response.text)
                return False

        except Exception as e:
            logger.error("send_packer_selection_list_error", to=to_phone, error=str(e))
            return False


    async def send_product_detail_with_buttons(self, to_phone: str, product: Product):
        """Sends a product detail message with interactive reply buttons."""
        try:
            availability_text = product.availability.replace('_', ' ').title()
            # FIXED: Create short description from the full description
            short_desc = (product.description[:120] + '...') if len(product.description) > 120 else product.description

            body_text = (
                f"*{product.title}*\n\n"
                f"💰 ₹{product.price:,.2f} | 📦 {availability_text}\n\n"
                f"✨ {short_desc}"
            )

            payload = {
                "messaging_product": "whatsapp",
                "to": to_phone,
                "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": body_text},
                    "action": {
                        "buttons": [
                            {
                                "type": "reply",
                                "reply": {
                                    "id": f"buy_{product.id}",
                                    "title": "🛒 Buy Now"
                                }
                            },
                            {
                                "type": "reply",
                                "reply": {
                                    "id": f"more_{product.id}",
                                    "title": "📖 More Info"
                                }
                            },
                            {
                                "type": "reply",
                                "reply": {
                                    "id": f"similar_{product.id}",
                                    "title": "🔍 Similar Items"
                                }
                            }
                        ]
                    }
                }
            }

            if product.image_url:
                payload["interactive"]["header"] = {
                    "type": "image",
                    "image": {"link": product.image_url}
                }

            url = f"{self.base_url}/{self.phone_id}/messages"
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json"
            }

            response = await self.resilient_api_call(
                self.circuit_breaker.call,
                self.http_client.post, url, json=payload, headers=headers
            )

            if response.status_code == 200:
                logger.info(
                    "whatsapp_interactive_product_sent",
                    to=to_phone,
                    product_id=product.id
                )
                return True
            else:
                logger.error(
                    "whatsapp_interactive_failed",
                    to=to_phone,
                    status_code=response.status_code,
                    response=response.text
                )
                return False

        except Exception as e:
            logger.error(
                "whatsapp_interactive_error",
                to=to_phone,
                error=str(e)
            )
            return False

    async def send_quick_replies(self, to_phone: str, message: str, options: Dict[str, str]):
        """Sends a message with up to 3 quick reply buttons."""
        try:
            buttons = []
            for title, option_id in options.items():
                buttons.append({
                    "type": "reply",
                    "reply": {
                        "id": option_id,
                        "title": title[:20]
                    }
                })

            payload = {
                "messaging_product": "whatsapp", "to": to_phone, "type": "interactive",
                "interactive": {
                    "type": "button",
                    "body": {"text": message},
                    "action": {"buttons": buttons}
                }
            }
            # ✅ FIX: Added the missing authentication headers.
            headers = {
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            }
            url = f"{self.base_url}/{self.phone_id}/messages"
            response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)

            if response.status_code == 200:
                logger.info("whatsapp_quick_replies_sent", to=to_phone, options_count=len(options))
                return True
            else:
                logger.error("whatsapp_quick_replies_failed", to=to_phone, status_code=response.status_code, response=response.text)
                return False

        except Exception as e:
            logger.error("send_quick_replies_error", to=to_phone, error=str(e))
            return False
    
    # Add exponential backoff for external APIs
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    async def resilient_api_call(self, func, *args, **kwargs):
        return await func(*args, **kwargs)


    async def get_media_url(self, media_id: str) -> Optional[str]:
        """Fetch a temporary URL for a media object from WhatsApp Graph API."""
        try:
            url = f"{self.base_url}/{media_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            resp = await self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            data = resp.json() if hasattr(resp, "json") else {}
            return data.get("url")
        except Exception as e:
            logger.error("whatsapp_get_media_url_failed", media_id=media_id, error=str(e))
            return None

    async def get_media_content(self, media_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Downloads media bytes and returns (bytes, mime_type)."""
        try:
            url = await self.get_media_url(media_id)
            if not url:
                return None, None
            headers = {"Authorization": f"Bearer {self.access_token}"}
            resp = await self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type") or resp.headers.get("Content-Type")
            return resp.content, content_type
        except Exception as e:
            logger.error("whatsapp_download_media_failed", media_id=media_id, error=str(e))
            return None, None
# ==================== SHOPIFY SERVICE ====================
class ShopifyService:
    def __init__(self, store_url: str, access_token: str, circuit_breaker, storefront_token: Optional[str] = None):
        self.store_url = store_url.replace('https://', '').replace('http://', '')
        self.access_token = access_token
        self.storefront_token = storefront_token
        self.circuit_breaker = circuit_breaker
        self.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(10.0, read=30.0, connect=5.0)
        )

    async def get_product_image_url(self, product_id: int) -> Optional[str]:
        """Gets a product's primary image URL using the REST API."""
        try:
            url = f"https://{self.store_url}/admin/api/2025-07/products/{product_id}/images.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            images = resp.json().get("images", [])
            return images[0].get("src") if images else None
        except Exception as e:
            logger.error("get_product_image_url_error", product_id=product_id, error=str(e))
            return None

    async def get_inventory_for_variant(self, variant_id: int) -> Optional[int]:
        """Gets the inventory quantity for a single variant ID."""
        try:
            # Using the REST API for simplicity
            url = f"https://{self.store_url}/admin/api/2025-07/variants/{variant_id}.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            variant_data = resp.json().get("variant", {})
            return variant_data.get("inventory_quantity")
        except Exception as e:
            logger.error("get_inventory_for_variant_error", variant_id=variant_id, error=str(e))
            return None
    
    # --- NEW STOREFRONT API METHODS FOR CART MANAGEMENT ---
    async def create_cart(self) -> Optional[str]:
        gql_mutation = "mutation { cartCreate { cart { id checkoutUrl } userErrors { field message } } }"
        data = await self._execute_storefront_gql_query(gql_mutation)
        return (data.get("cartCreate") or {}).get("cart", {}).get("id")

    async def add_item_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1) -> bool:
        gql_mutation = '''
        mutation($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart { id }
            userErrors { field message }
          }
        }'''
        variables = {"cartId": cart_id, "lines": [{"merchandiseId": variant_id, "quantity": quantity}]}
        data = await self._execute_storefront_gql_query(gql_mutation, variables)
        return not ((data.get("cartLinesAdd") or {}).get("userErrors"))

    async def get_checkout_url(self, cart_id: str) -> Optional[str]:
        gql_query = "query($cartId: ID!) { cart(id: $cartId) { checkoutUrl } }"
        data = await self._execute_storefront_gql_query(gql_query, {"cartId": cart_id})
        return (data.get("cart") or {}).get("checkoutUrl")

    async def get_product_by_handle(self, handle: str) -> Optional[Product]:
        """Gets a single product by its handle (URL slug)."""
        query = f'handle:"{handle}"'
        products = await self.get_products(query, limit=1)
        return products[0] if products else None
    
    async def get_location_id(self) -> Optional[int]:
        """Gets the first active Shopify location ID."""
        # This prevents having to hardcode the location ID.
        try:
            cache_key = "shopify:location_id"
            cached_id = await services.cache_service.get(cache_key)
            if cached_id:
                return int(cached_id)

            url = f"https://{self.store_url}/admin/api/2025-07/locations.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            locations = resp.json().get("locations", [])
            
            if locations:
                location_id = locations[0].get("id")
                await services.cache_service.set(cache_key, str(location_id), ttl=86400) # Cache for 24 hours
                return location_id
            return None
        except Exception as e:
            logger.error("get_shopify_location_id_error", error=str(e))
            return None

    # In server.py, inside the ShopifyService class:


    async def fulfill_order(self, order_id: int, tracking_number: str, packer_name: str, carrier: str = "India Post") -> Tuple[bool, Optional[int]]:
        """
        Fulfills an order using the Fulfillment Orders API and returns (success, fulfillment_id).
        """
        try:
            fulfillment_orders_url = f"https://{self.store_url}/admin/api/2025-07/orders/{order_id}/fulfillment_orders.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            
            fo_resp = await self.resilient_api_call(self.http_client.get, fulfillment_orders_url, headers=headers)
            fo_resp.raise_for_status()
            fulfillment_orders = fo_resp.json().get("fulfillment_orders", [])
            
            open_fo = next((fo for fo in fulfillment_orders if fo.get("status") == "open"), None)
            if not open_fo:
                logger.error("shopify_fulfill_no_open_fulfillment_orders", order_id=order_id)
                return False, None

            line_items = [{"id": item["id"], "quantity": item["fulfillable_quantity"]} for item in open_fo.get("line_items", []) if item.get("fulfillable_quantity", 0) > 0]

            fulfillment_payload = {
                "fulfillment": {
                    "line_items_by_fulfillment_order": [{"fulfillment_order_id": open_fo["id"], "fulfillment_order_line_items": line_items}],
                    "tracking_info": {"number": tracking_number, "company": carrier},
                    "notify_customer": True
                }
            }

            fulfillment_url = f"https://{self.store_url}/admin/api/2025-07/fulfillments.json"
            resp = await self.resilient_api_call(self.http_client.post, fulfillment_url, json=fulfillment_payload, headers=headers)
            
            if resp.status_code == 201:
                fulfillment_data = resp.json().get("fulfillment", {})
                fulfillment_id = fulfillment_data.get("id")
                logger.info("shopify_order_fulfilled_successfully", order_id=order_id, fulfillment_id=fulfillment_id)
                return True, fulfillment_id
            else:
                logger.error("shopify_fulfill_order_failed", order_id=order_id, status_code=resp.status_code, response_text=resp.text)
                return False, None
                
        except Exception as e:
            logger.error("shopify_fulfill_order_exception", order_id=order_id, error=str(e), exc_info=True)
            return False, None

    async def update_fulfillment_tracking(self, fulfillment_id: int, new_tracking_number: str, carrier: str) -> bool:
        """Updates the tracking number for an existing fulfillment."""
        try:
            update_url = f"https://{self.store_url}/admin/api/2025-07/fulfillments/{fulfillment_id}/update_tracking.json"
            payload = {
                "fulfillment": {
                    "tracking_info": {"number": new_tracking_number, "company": carrier},
                    "notify_customer": True
                }
            }
            headers = {"X-Shopify-Access-Token": self.access_token, "Content-Type": "application/json"}
            resp = await self.resilient_api_call(self.http_client.post, update_url, json=payload, headers=headers)
            
            if resp.status_code == 200:
                logger.info("shopify_tracking_updated_successfully", fulfillment_id=fulfillment_id)
                return True
            else:
                logger.error("shopify_tracking_update_failed", fulfillment_id=fulfillment_id, status=resp.status_code, response=resp.text)
                return False
        except Exception as e:
            logger.error("shopify_tracking_update_exception", fulfillment_id=fulfillment_id, error=str(e))
            return False


    # --- PRIVATE: STOREFRONT EXECUTOR ---
    async def _execute_storefront_gql_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        if not getattr(self, "storefront_token", None):
            return {}
        url = f"https://{self.store_url}/api/2025-07/graphql.json"
        headers = {
            "X-Shopify-Storefront-Access-Token": self.storefront_token,
            "Content-Type": "application/json"
        }
        resp = await self.resilient_api_call(
            self.circuit_breaker.call,
            self.http_client.post, url, json={"query": query, "variables": variables or {}}, headers=headers
        )
        resp.raise_for_status()
        try:
            body = resp.json()
        except Exception:
            return {}
        return body.get("data", {}) if isinstance(body, dict) else {}

# --- MAIN PUBLIC METHODS ---


    async def get_products(self, query: str, limit: int = 25, sort_key: str = "RELEVANCE", filters: Optional[Dict] = None) -> Tuple[List[Product], int]:
        """
        Executes a product search.
        Returns a tuple containing:
        - A list of products (filtered if applicable).
        - An integer representing the count of products *before* any local filtering was applied.
        """
        try:
            edges = await self._shopify_search(query, limit, sort_key, filters)
            if not edges:
                return [], 0

            products = []
            for edge in edges:
                node = edge.get("node", {})
                variants_edge = node.get("variants", {}).get("edges", [])
                if not variants_edge: continue
                variant_node = variants_edge[0].get("node", {})
                price_info = variant_node.get("priceV2", {})

                products.append(Product(
                    id=node.get("id"), title=node.get("title"),
                    description=node.get("description", "No description available."),
                    price=float(price_info.get("amount", 0.0)),
                    variant_id=variant_node.get("id"), sku=variant_node.get("sku"),
                    currency=price_info.get("currencyCode", "INR"),
                    image_url=node.get("featuredImage", {}).get("url"),
                    handle=node.get("handle", ""), tags=node.get("tags", [])
                ))
            
            unfiltered_count = len(products)

            if filters and "price" in filters and products:
                price_condition = filters["price"]
                locally_filtered_products = []
                
                if "lessThan" in price_condition:
                    max_price = price_condition["lessThan"]
                    locally_filtered_products = [p for p in products if p.price < max_price]
                    return locally_filtered_products, unfiltered_count

                if "greaterThan" in price_condition:
                    min_price = price_condition["greaterThan"]
                    locally_filtered_products = [p for p in products if p.price > min_price]
                    return locally_filtered_products, unfiltered_count

            return products, unfiltered_count
            
        except Exception as e:
            logger.error("shopify_get_products_error", error=str(e), error_type=type(e).__name__)
            return [], 0



    async def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Gets a single product by its GraphQL GID and parses it into a Product model."""
        gql_query = """
        query($id: ID!) {
          node(id: $id) { ... on Product { ...productFields } }
        }
        fragment productFields on Product { 
            id title handle bodyHtml productType tags 
            variants(first: 1){edges{node{id price inventoryQuantity}}} 
            images(first: 1){edges{node{originalSrc}}}
        }
        """
        variables = {"id": product_id}
        try:
            data = await self._execute_gql_query(gql_query, variables)
            if not data.get("node"): return None
            products = self._parse_products([{"node": data.get("node")}])
            return products[0] if products else None
        except Exception as e:
            logger.error("shopify_product_by_id_error", error=str(e), product_id=product_id)
            return None

    async def get_product_variants(self, product_id: str) -> List[Dict]:
        """Gets all variants (e.g., sizes, colors) for a given product ID."""
        gql_query = """
        query($id: ID!) {
          node(id: $id) {
            ... on Product {
              variants(first: 10) {
                edges { node { id title price inventoryQuantity } }
              }
            }
          }
        }
        """
        variables = {"id": product_id}
        try:
            data = await self._execute_gql_query(gql_query, variables)
            variant_edges = data.get("node", {}).get("variants", {}).get("edges", [])
            return [edge["node"] for edge in variant_edges]
        except Exception as e:
            logger.error("get_product_variants_error", error=str(e), product_id=product_id)
            return []

    # --- URL GENERATION HELPERS ---
    def get_add_to_cart_url(self, variant_gid: str) -> str:
        numeric_variant_id = variant_gid.split('/')[-1]
        return f"https://{self.store_url}/cart/{numeric_variant_id}:1"

    def get_product_page_url(self, handle: str) -> str:
        return f"https://{self.store_url}/products/{handle}"


    # --- PRIVATE HELPERS ---
    async def _shopify_search(self, query: str, limit: int = 25, sort_key: str = "RELEVANCE", 
                              filters: Optional[Dict] = None) -> List[Dict]:
        """
        Executes a GraphQL query against the Shopify Storefront API.
        This corrected version mirrors the working logic and relies on Python-based filtering.
        """
        logger.info("shopify_search_debug", 
               query=query, 
               limit=limit, 
               sort_key=sort_key, 
               filters=filters,
               store_url=self.store_url,
               has_storefront_token=bool(self.storefront_token))
        
        # ✅ FIX: Reverted to the simpler GraphQL query that does not include the 'filters' argument.
        graphql_query_payload = {
            "query": """
            query ($query: String!, $limit: Int!, $sortKey: ProductSortKeys!) {
              products(first: $limit, query: $query, sortKey: $sortKey) {
                edges {
                  node {
                    id title handle description tags
                    featuredImage { url }
                    variants(first: 1) {
                      edges { node { id sku priceV2 { amount currencyCode } } }
                    }
                  }
                }
              }
            }
            """,
            "variables": {
                "query": query, 
                "limit": limit, 
                "sortKey": sort_key
            }
        }
        
        url = f"https://{self.store_url}/api/2025-07/graphql.json"
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Storefront-Access-Token": self.storefront_token
        }

        try:
            # Use the corrected payload in the post request
            resp = await self.http_client.post(url, headers=headers, json=graphql_query_payload)
            resp.raise_for_status()
            
            data = resp.json()
            products = data.get("data", {}).get("products", {}).get("edges", [])
            logger.info("shopify_search_batch_fetched", count=len(products), sort_key=sort_key, filters=filters)
            return products
        except Exception as e:
            logger.error("shopify_search_error", error=str(e), query_vars=graphql_query_payload.get("variables"))
            return []


    async def _execute_gql_query(self, query: str, variables: Dict) -> Dict:
        url = f"https://{self.store_url}/admin/api/2025-07/graphql.json"
        headers = {"X-Shopify-Access-Token": self.access_token, "Content-Type": "application/json"}
        resp = await self.resilient_api_call(
            self.circuit_breaker.call,
            self.http_client.post, url, json={"query": query, "variables": variables}, headers=headers
        )
        resp.raise_for_status()
        return resp.json().get("data", {})



    def _parse_products(self, product_edges: List[Dict]) -> List[Product]:
        """Parses GraphQL product edges into Pydantic Product models with full descriptions."""
        products = []
        for edge in product_edges:
            node = edge.get("node", {})
            if not node: continue
            variant_edge = node.get("variants", {}).get("edges", [])
            if not variant_edge: continue
            
            image_edge = node.get("images", {}).get("edges", [])
            inventory = variant_edge[0]["node"].get("inventoryQuantity")
            
            # FIXED: Store the full, clean description
            clean_description = html.unescape(re.sub("<[^<]+?>", "", node.get("bodyHtml", "") or ""))

            products.append(Product(
                id=node.get("id"), title=node.get("title"), description=clean_description,
                price=float(variant_edge[0]["node"].get("price", 0.0)),
                handle=node.get("handle"), variant_id=variant_edge[0]["node"].get("id"),
                image_url=image_edge[0]["node"]["originalSrc"] if image_edge else None,
                availability="in_stock" if inventory > 0 else "out_of_stock",
                tags=node.get("tags", []),
            ))
        return products

    async def search_orders_by_phone(self, phone_number: str, max_fetch: int = 250) -> List[Dict]:
        """ Search for recent orders using a customer's phone number via the REST API. Fully async-safe using httpx.AsyncClient. """
        try:
            cache_key = f"shopify:orders_by_phone:{phone_number}"
            try:
                cached = await services.cache_service.get(cache_key)
                if cached:
                    return json.loads(cached)
            except Exception:
                logger.debug("shopify_cache_get_failed", phone=phone_number, error="cache miss or error")

            def digits_of(s: Optional[str]) -> str:
                if not s: return ""
                return re.sub(r'\D', '', s)

            user_digits = digits_of(phone_number)
            last10 = user_digits[-10:] if len(user_digits) >= 10 else user_digits

            rest_url = f"https://{self.store_url}/admin/api/2025-07/orders.json"
            params = {"status": "any", "limit": min(max_fetch, 250)}
            headers = {"X-Shopify-Access-Token": self.access_token}

            resp = await self.resilient_api_call(
                self.circuit_breaker.call,
                self.http_client.get, rest_url, params=params, headers=headers, timeout=10.0
            )
            resp.raise_for_status()

            all_orders = resp.json().get("orders", []) or []
            matching = []

            for order in all_orders:
                customer_phone = (order.get("customer") or {}).get("phone") if order.get("customer") else None
                shipping_phone = (order.get("shipping_address") or {}).get("phone") if order.get("shipping_address") else None

                matched = False
                for candidate in (customer_phone, shipping_phone):
                    if candidate:
                        try:
                            strict_user = EnhancedSecurityService.sanitize_phone_number(phone_number)
                            cand_strict = EnhancedSecurityService.sanitize_phone_number(candidate)
                            if cand_strict == strict_user:
                                matched = True
                                break
                        except Exception:
                            pass

                if not matched:
                    for candidate in (customer_phone, shipping_phone):
                        if candidate:
                            if last10 and digits_of(candidate).endswith(last10):
                                matched = True
                                break
                            if len(last10) < 10 and digits_of(candidate).endswith(last10):
                                matched = True
                                break

                if matched:
                    matching.append(order)

            try:
                await services.cache_service.set(cache_key, json.dumps(matching, default=str), ttl=120)
            except Exception:
                logger.debug("shopify_cache_set_failed", phone=phone_number)

            logger.info("shopify_orders_found", phone=phone_number, count=len(matching))
            return matching
        except Exception as e:
            logger.error("shopify_orders_error", error=str(e), phone=phone_number)
            return []

    # Add exponential backoff for external APIs
    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    async def resilient_api_call(self, func, *args, **kwargs):
        return await func(*args, **kwargs)


    # --- NEW STOREFRONT API METHODS FOR CART MANAGEMENT ---
    async def create_cart(self) -> Optional[str]:
        gql_mutation = "mutation { cartCreate { cart { id checkoutUrl } userErrors { field message } } }"
        data = await self._execute_storefront_gql_query(gql_mutation)
        return (data.get("cartCreate") or {}).get("cart", {}).get("id")

    async def add_item_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1) -> bool:
        gql_mutation = '''
        mutation($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) {
            cart { id }
            userErrors { field message }
          }
        }'''
        variables = {"cartId": cart_id, "lines": [{"merchandiseId": variant_id, "quantity": quantity}]}
        data = await self._execute_storefront_gql_query(gql_mutation, variables)
        return not ((data.get("cartLinesAdd") or {}).get("userErrors"))

    async def get_checkout_url(self, cart_id: str) -> Optional[str]:
        gql_query = "query($cartId: ID!) { cart(id: $cartId) { checkoutUrl } }"
        data = await self._execute_storefront_gql_query(gql_query, {"cartId": cart_id})
        return (data.get("cart") or {}).get("checkoutUrl")

    # --- PRIVATE: STOREFRONT EXECUTOR ---
    async def _execute_storefront_gql_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        if not self.storefront_token:
            return {}
        url = f"https://{self.store_url}/api/2025-07/graphql.json"
        headers = {
            "X-Shopify-Storefront-Access-Token": self.storefront_token,
            "Content-Type": "application/json"
        }
        resp = await self.resilient_api_call(
            self.circuit_breaker.call,
            self.http_client.post, url, json={"query": query, "variables": variables or {}}, headers=headers
        )
        resp.raise_for_status()
        body = resp.json() if hasattr(resp, "json") else {}
        return body.get("data", {}) if isinstance(body, dict) else {}
# ==================== ORDER REPOSITORY ====================
class OrderRepository:
    def __init__(self, shopify_service, db_service):
        self.shopify_service = shopify_service
        self.db = db_service.db  # For v2

    async def get_orders_by_phone(self, phone_number: str) -> List[Dict]:
        # V1: Direct Shopify API lookup
        return await self.shopify_service.search_orders_by_phone(phone_number)
        # V2 (future): MongoDB query when webhooks are implemented
        # return await self.db.orders.find({"phone_numbers": phone_number}).sort("created_at", -1).to_list(50)

# ==================== AI SERVICE ====================
class AIService:
    def __init__(self):
        if settings.gemini_api_key:
            genai.configure(api_key=settings.gemini_api_key)
            self.gemini_client = genai.GenerativeModel('gemini-1.5-flash')
        else: self.gemini_client = None
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        else: self.openai_client = None
        self.visual_matcher = VisualProductMatcher()
        self.circuit_breaker = CircuitBreaker()
        self.openai_breaker = CircuitBreaker()


    async def generate_response(self, message: str, context: Optional[Dict] = None) -> str:
        """Generate AI response for general text-based inquiries."""
        try:
            serializable_context = json.loads(json.dumps(context, default=str)) if context else {}

            if self.gemini_client:
                try:
                    response = await self._generate_gemini_response(message, serializable_context)
                    if response:
                        ai_requests_counter.labels(model="gemini", status="success").inc()
                        return response
                except Exception as e:
                    logger.error("gemini_api_call_failed", error=str(e))
                    ai_requests_counter.labels(model="gemini", status="error").inc()

            if self.openai_client:
                try:
                    response = await self.openai_breaker.call(self._generate_openai_response, message, serializable_context)
                    if response:
                        ai_requests_counter.labels(model="openai", status="success").inc()
                        return response
                except Exception as e:
                    logger.error("openai_fallback_failed", error=str(e))
                    ai_requests_counter.labels(model="openai", status="error").inc()

            # Fallback response if both AI models fail
            return "I'm sorry, I'm having trouble connecting to my knowledge base. Could you please rephrase your question?"

        except Exception as e:
            logger.error("ai_generation_error", error=str(e))
            ai_requests_counter.labels(model="error", status="error").inc()
            return "I apologize, but I'm having some technical difficulties. How can I help with our products?"

    async def _generate_openai_response(self, message: str, context: Dict = None) -> str:
        """Generate response using OpenAI with the full brand persona."""
        system_prompt = """You are FeelOri's friendly and expert fashion shopping assistant. Your persona is warm, knowledgeable, and passionate about helping women express themselves.

        **Your Brand Story & Founder:**
        FeelOri has a rich heritage of over 65 years in jewelry craftsmanship, rooted in a family tradition from Telangana that began in the 1950s. Our founder, Pooja Tunk, grew up surrounded by this artistry and launched FeelOri.com to blend timeless tradition with modern trends. We now offer a wide range of handcrafted jewelry and lightweight hair extensions.

        **Your Mission:**
        Our mission is to empower every woman to "Feel Original. Feel Beautiful. Feel You." We do this by providing ethically sourced, comfortable, and affordable luxury accessories.

        **Instructions:**
        - When asked about the owner or founder, proudly mention our founder, Pooja Tunk, and her vision for the brand.
        - When asked about the brand's history, mention our 65+ years of craftsmanship and roots in Telangana.
        - If asked what you sell, remember to mention both jewelry and hair extensions.
        - Always steer the conversation back towards helping the customer find products.
        - NEVER say "As a large language model" or "I don't have access to...". You are a knowledgeable assistant from the FeelOri team.
        - Keep responses concise, friendly, and use emojis where appropriate (✨, 💖, 💍).
        """
        messages = [{"role": "system", "content": system_prompt}]
        if context and context.get("conversation_history"):
            for exchange in context["conversation_history"]:
                messages.append({"role": "user", "content": exchange.get("message", "")})
                messages.append({"role": "assistant", "content": exchange.get("response", "")})
        messages.append({"role": "user", "content": message})

        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.7
        )
        return response.choices[0].message.content.strip()

    async def _generate_gemini_response(self, message: str, context: Dict = None) -> str:
        """Generate response using Gemini with the full brand persona."""
        system_prompt = """You are FeelOri's friendly and expert fashion shopping assistant. Your persona is warm, knowledgeable, and passionate about helping women express themselves.

        **Your Brand Story & Founder:**
        FeelOri has a rich heritage of over 65 years in jewelry craftsmanship, rooted in a family tradition from Telangana that began in the 1950s. Our founder, Pooja Tunk, grew up surrounded by this artistry and launched FeelOri.com to blend timeless tradition with modern trends. We now offer a wide range of handcrafted jewelry and lightweight hair extensions.

        **Your Mission:**
        Our mission is to empower every woman to "Feel Original. Feel Beautiful. Feel You." We do this by providing ethically sourced, comfortable, and affordable luxury accessories.

        **Instructions:**
        - When asked about the owner or founder, proudly mention our founder, Pooja Tunk, and her vision for the brand.
        - When asked about the brand's history, mention our 65+ years of craftsmanship and roots in Telangana.
        - If asked what you sell, remember to mention both jewelry and hair extensions.
        - Always steer the conversation back towards helping the customer find products.
        - NEVER say "As a large language model" or "I don't have access to...". You are a knowledgeable assistant from the FeelOri team.
        - Keep responses concise, friendly, and use emojis where appropriate (✨, 💖, 💍).
        """
        full_prompt = f"{system_prompt}\n\nContext: {json.dumps(context)}\n\nMessage: {message}"
        response = await self.gemini_client.generate_content_async(full_prompt)
        return response.text.strip()

    # --- NEW HYBRID VISUAL SEARCH ORCHESTRATOR ---
    async def get_keywords_from_image_for_reranking(self, image_bytes: bytes, mime_type: str) -> List[str]:
        """Generates text keywords from an image for re-ranking."""
        if not self.gemini_client: return []
        try:
            prompt = ("Analyze this jewelry photo. Return ONLY a comma-separated list of 3-4 of the most relevant lowercase keywords for a product search.\n"
                      "1. Start with the single most accurate primary category from this list: `[necklace, earrings, bangle, ring, set, choker, haram, jhumka, stud]`.\n"
                      "2. Add 2-3 other dominant keywords (e.g., primary stone, color, style).\n"
                      "Example: `set, ruby, gold plated, traditional`")
            vision_model = genai.GenerativeModel('gemini-1.5-flash')
            image_part = {"mime_type": mime_type, "data": image_bytes}
            resp = await vision_model.generate_content_async([prompt, image_part])
            text = (resp.text or "").strip().lower()
            return [k.strip() for k in text.split(',') if k.strip()]
        except Exception as e:
            logger.error("get_keywords_for_reranking_error", error=str(e)); return []
            
    def _calculate_keyword_relevance(self, keywords: List[str], candidate: Dict) -> float:
        """
        Final, enhanced relevance calculation. It gives a huge bonus if the primary
        category keyword matches, making it more resilient to messy tags.
        """
        if not keywords:
            return 0.0

        relevance_score = 0.0
        matched_keywords = set()

        title_lower = candidate['title'].lower()
        tags_lower = [tag.lower() for tag in candidate.get('tags', [])]

        # --- NEW: Prioritize the primary category keyword ---
        primary_category = keywords[0]

        # Huge bonus if the main category is in the title. This is a strong signal.
        if primary_category in title_lower:
            relevance_score += 1.5
            matched_keywords.add(primary_category)
        # Good bonus if it's in the tags.
        elif primary_category in tags_lower:
            relevance_score += 1.0
            matched_keywords.add(primary_category)

        # Check the rest of the descriptive keywords
        for keyword in keywords[1:]:
            if keyword in title_lower:
                relevance_score += 0.5
                matched_keywords.add(keyword)
            elif keyword in tags_lower:
                relevance_score += 0.3
                matched_keywords.add(keyword)

        # Bonus for multiple keyword matches
        if len(matched_keywords) > 1:
            relevance_score += 0.5 * (len(matched_keywords) - 1)

        return relevance_score


    async def find_exact_product_by_image(self, image_bytes: bytes, mime_type: str) -> Dict:
        """Final hybrid visual search with weighted re-ranking and detailed response."""
        try:
            visual_candidates_task = self.visual_matcher.find_matching_products(image_bytes, top_k=15)
            keyword_task = self.get_keywords_from_image_for_reranking(image_bytes, mime_type)
            visual_candidates, keywords = await asyncio.gather(visual_candidates_task, keyword_task)

            if not visual_candidates:
                return {'success': False, 'message': 'No products found in the visual index.'}

            logger.info(f"Generated keywords for re-ranking: {keywords}")

            ranked_products = []
            for candidate in visual_candidates:
                relevance_score = self._calculate_keyword_relevance(keywords, candidate)
                final_score = (candidate['similarity_score'] * 0.4) + (relevance_score * 0.6)
                candidate.update({'final_score': final_score, 'relevance_score': relevance_score})
                ranked_products.append(candidate)

            ranked_products.sort(key=lambda x: x['final_score'], reverse=True)
            
            # --- THIS IS THE FIX ---
            # 1. Prepare the log data separately.
            top_matches_log = [(p['title'][:30], f"final:{p['final_score']:.2f}") for p in ranked_products[:3]]
            # 2. Log the prepared data in a simple f-string.
            logger.info(f"Top 3 matches: {top_matches_log}")
            # --- END OF FIX ---
     
            
            best_match = ranked_products[0]
            match_type = 'similar' # Default type
            if best_match['similarity_score'] >= 0.92 and best_match['relevance_score'] >= 1.0:
                match_type = 'exact'
            elif best_match['final_score'] >= 0.8:
                 match_type = 'very_similar'

            final_products = []
            for match in ranked_products:
                if match['final_score'] >= 0.6: # Final score threshold
                    product = await services.shopify_service.get_product_by_handle(match['handle'])
                    if product: final_products.append(product)
            
            if final_products:
                return {'success': True, 'match_type': match_type, 'products': final_products[:5]}

            return {'success': False, 'message': 'No sufficiently matching products found.'}
            
        except Exception as e:
            logger.error("Error in enhanced hybrid visual search", error=str(e), exc_info=True)
            return {'success': False, 'error': str(e)}

# ==================== SERVICE CONTAINER ====================
class ServiceContainer:
    def __init__(self):
        self.db_service = None
        self.cache_service = None
        self.state_service = None
        self.message_queue = None
        self.whatsapp_service = None
        self.shopify_service = None
        self.ai_service = None
        self.order_repository = None
        self.login_tracker = None
        self.rate_limiter = None
        self.migration_service = None
        self.db_health_monitor = None
        self.resource_monitor = None
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, read=30.0, connect=5.0)
        )
    

    async def initialize(self):
        """Initialize all services"""
        try:
            # --- THIS IS THE FIX ---
            # 1. Initialize Cache and essential services FIRST
            self.cache_service = CacheService(settings.redis_url)
            self.state_service = StateManagementService(self.cache_service.redis)
            self.rate_limiter = AdvancedRateLimiter(self.cache_service.redis)
            self.login_tracker = RedisLoginAttemptTracker(self.cache_service.redis)
            # --- END OF FIX ---

            # 2. Initialize other services that DEPEND on the cache
            
            # Database Service with Redis circuit breaker
            self.db_service = DatabaseService(settings.mongo_atlas_uri)
            self.db_service.circuit_breaker = RedisCircuitBreaker(
                self.cache_service.redis, "database"
            )
            await self.db_service.create_indexes()
            
            # Migration (for v2 readiness)
            self.migration_service = MigrationService(self.db_service)
            await self.migration_service.run_migrations()
            
            # Message Queue
            self.message_queue = RedisMessageQueue(self.cache_service.redis)
            await self.message_queue.start_workers()
            
            # WhatsApp Service with Redis circuit breaker
            self.whatsapp_service = WhatsAppService(
                settings.whatsapp_access_token,
                settings.whatsapp_phone_id,
                self.http_client,
                settings.whatsapp_business_account_id # Pass the new setting here
            )
            self.whatsapp_service.circuit_breaker = RedisCircuitBreaker(
                self.cache_service.redis, "whatsapp"
            )

            # Shopify Service with Redis circuit breaker
            self.shopify_service = ShopifyService(
                settings.shopify_store_url,
                settings.shopify_access_token,
                RedisCircuitBreaker(self.cache_service.redis, "shopify"),
                storefront_token=settings.shopify_storefront_access_token
            )
            
            # Order Repository
            self.order_repository = OrderRepository(
                self.shopify_service,
                self.db_service
            )
            
            # AI
            self.ai_service = AIService()
            
            # Monitors
            self.db_health_monitor = DatabaseHealthMonitor(self.db_service, self.db_service.circuit_breaker)
            asyncio.create_task(self.db_health_monitor.monitor_connection())
            
            self.resource_monitor = ResourceMonitor()
            asyncio.create_task(self.resource_monitor.check_memory_usage())
            
            logger.info("services_initialized_successfully")
        except Exception as e:
            logger.error("service_initialization_failed", error=str(e))
            await alerting.send_critical_alert("Service initialization failed", {"error": str(e)})
            raise
    
    async def cleanup(self):
        """Cleanup all services"""
        try:
            # Stop message queue workers
            if self.message_queue:
                await self.message_queue.stop_workers()
            
            # Close HTTP client
            await self.http_client.aclose()
            
            # Close database
            if self.db_service:
                self.db_service.client.close()
            
            # Cleanup alerting
            await alerting.cleanup()
            
            logger.info("services_cleaned_up_successfully")
            
        except Exception as e:
            logger.error("service_cleanup_error", error=str(e))

# ==================== GLOBAL SERVICES ====================
services = ServiceContainer()

# Initialize rate limiter
limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[f"{settings.rate_limit_per_minute}/minute"]
)

# OAuth2 scheme for JWT
oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/{settings.api_version}/auth/login")

# ==================== PRODUCTION CONFIG VALIDATOR ====================
class ProductionConfigValidator:
    @staticmethod
    def validate_ssl_config():
        if settings.https_only:
            if not settings.ssl_cert_path or not settings.ssl_key_path:
                raise ValueError("SSL cert and key paths required for HTTPS")
            
    @staticmethod
    def validate_security_config():
        if settings.environment == "production":
            if settings.cors_allowed_origins == "*":
                raise ValueError("Wildcard CORS not allowed in production")



# ==================== APPLICATION LIFECYCLE ====================

# ✅ 1. Initialize the scheduler globally
scheduler = AsyncIOScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management."""
    global ADMIN_PASSWORD_HASH

    # This logic guarantees the correct password is used for tests.
    if settings.environment == "test":
        test_password = "a_secure_test_password_123"
        print(f'--- RUNNING IN TEST MODE: Forcing admin password to "{test_password}" ---')
        ADMIN_PASSWORD_HASH = SecurityService.hash_password(test_password)
    else:
        ADMIN_PASSWORD_HASH = SecurityService.hash_password(settings.admin_password)

    logger.info("application_starting", 
               version="2.0.0",
               environment=settings.environment)
    
    sentry_enabled = initialize_sentry()
    tracing_enabled = setup_tracing()
    
    try:
        await services.initialize()
        
        # ✅ 2. Start the scheduler after services are ready
        scheduler.start()
        logger.info("scheduler_started")

        logger.info("application_ready", 
                   sentry_enabled=sentry_enabled,
                   tracing_enabled=tracing_enabled)
        yield
    except Exception as e:
        logger.error("application_startup_failed", error=str(e))
        await alerting.send_critical_alert("Application startup failed", {"error": str(e)})
        raise
    finally:
        # ✅ 3. Shut down the scheduler gracefully
        scheduler.shutdown()
        logger.info("scheduler_shutdown")

        logger.info("application_shutting_down")
        await services.cleanup()

# Setup signal handlers for graceful shutdown
def setup_signal_handlers(app):
    async def graceful_shutdown():
        logger.info("starting_graceful_shutdown")
        await services.cleanup()
        loop = asyncio.get_running_loop()
        loop.stop()

    def signal_handler(signum, frame):
        logger.info("shutdown_signal_received", signal=signum)
        asyncio.create_task(graceful_shutdown())

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

# ==================== FASTAPI APPLICATION ====================
app = FastAPI(
    title="Feelori AI WhatsApp Assistant",
    version="2.0.0",
    description="Production-ready AI WhatsApp assistant with enterprise features",
    lifespan=lifespan,
    openapi_url=f"/api/{settings.api_version}/openapi.json" if settings.environment != "production" else None,
    docs_url=f"/api/{settings.api_version}/docs" if settings.environment != "production" else None,
    redoc_url=f"/api/{settings.api_version}/redoc" if settings.environment != "production" else None,
)

# Create a directory named "static" in your project's root folder
# --- START: Corrected Static Files Configuration ---

# Get the absolute path to the directory where server.py is located
basedir = os.path.abspath(os.path.dirname(__file__))

# Create the full path to the 'static' directory
static_dir = os.path.join(basedir, "static")

# Mount the static directory using the full, unambiguous path
app.mount("/static", StaticFiles(directory=static_dir), name="static")

# --- END: Corrected Static Files Configuration ---

# Setup signals
setup_signal_handlers(app)

# ==================== MIDDLEWARE SETUP ====================
# This middleware runs in all environments
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response

# --- START OF FINAL FIX ---
# Define allowed origins based on the environment
cors_origins = [origin.strip() for origin in settings.cors_allowed_origins.split(",") if origin.strip()]
if settings.environment != "production":
    # For tests and development, we absolutely need to allow the frontend's origin
    print(f"--- {settings.environment.upper()} Mode: Allowing additional CORS origins ---")
    cors_origins.extend(["http://localhost:3000", "http://127.0.0.1:3000"])

# CORS middleware is now configured with the correct origins for the environment
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"], # OPTIONS is needed for preflight requests
    allow_headers=["*"],
)

# Add security-related middleware ONLY for non-test environments
if settings.environment != "test":
    print("--- Production/Dev Mode: Adding security middleware ---")
    allowed_hosts = [host.strip() for host in settings.allowed_hosts.split(",") if host.strip()]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)
else:
    print("--- Test Mode: Skipping TrustedHostMiddleware ---")

# These middleware are safe for all environments
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
# --- END OF FINAL FIX ---

# Performance middleware
@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    
    response_time_histogram.labels(
        endpoint=request.url.path
    ).observe(process_time)
    
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Custom timeout middleware (simple version)
@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

# ==================== AUTHENTICATION DEPENDENCIES ====================
async def verify_jwt_token(token: str = Depends(oauth2_scheme)) -> dict:
    """Verify JWT token with enhanced security"""
    try:
        payload = jwt_service.verify_token(token)
        
        # Verify admin role
        if payload.get("sub") != "admin":
            auth_attempts_counter.labels(status="unauthorized", method="jwt").inc()
            logger.warning("unauthorized_access_attempt", 
                         user=payload.get("sub"), 
                         token_type=payload.get("type"))
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not authorized for this resource"
            )
        
        # Verify token type
        if payload.get("type") != "access":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Invalid token type"
            )
        
        auth_attempts_counter.labels(status="success", method="jwt").inc()
        return payload
    except HTTPException:
        raise
    except JWTError as e:
        auth_attempts_counter.labels(status="failure", method="jwt").inc()
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token verification failed: {str(e)}",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def verify_webhook_signature(request: Request):
    """Webhook signature verification dependency"""
    try:
        body = await request.body()
        signature = request.headers.get("x-hub-signature-256", "")
        
        if not EnhancedSecurityService.verify_webhook_signature(
            body, signature, settings.whatsapp_webhook_secret
        ):
            webhook_signature_counter.labels(status="invalid").inc()
            
            await services.db_service.log_security_event(
                "invalid_webhook_signature",
                get_remote_address(request),
                {
                    "signature": signature[:50],
                    "body_length": len(body),
                    "user_agent": request.headers.get("user-agent", "")
                }
            )
            
            logger.warning("webhook_signature_invalid",
                         signature_prefix=signature[:20],
                         ip=get_remote_address(request))
            
            raise HTTPException(status_code=403, detail="Invalid signature")
        
        webhook_signature_counter.labels(status="valid").inc()
        request.state.verified_body = body
        return body
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("webhook_signature_verification_error", error=str(e))
        raise HTTPException(status_code=500, detail="Signature verification failed")

async def verify_metrics_access(request: Request):
    """Optional API key protection for metrics endpoint"""
    if settings.api_key:
        auth_header = request.headers.get("authorization", "")
        api_key = request.headers.get("x-api-key", "")
        
        if auth_header.startswith("Bearer "):
            provided_key = auth_header[7:]
        elif api_key:
            provided_key = api_key
        else:
            raise HTTPException(
                status_code=401, 
                detail="API key required for metrics access"
            )
        
        if not secrets.compare_digest(provided_key, settings.api_key):
            logger.warning("invalid_metrics_access_attempt", 
                         ip=get_remote_address(request))
            raise HTTPException(status_code=403, detail="Invalid API key")
    
    return True


async def verify_shopify_signature(request: Request) -> bytes:
    """Verify Shopify webhook using X-Shopify-Hmac-Sha256 header. Returns raw body if valid."""
    # --- NEW: Safety check for when the secret is not configured ---
    if not settings.shopify_webhook_secret:
        logger.error("shopify_webhook_received_but_secret_not_configured")
        raise HTTPException(
            status_code=501, 
            detail="Shopify webhook processing is not configured on the server."
        )
    # --- END OF NEW CODE ---

    try:
        body = await request.body()
        header_val = request.headers.get("X-Shopify-Hmac-Sha256") or ""
        if not header_val:
            logger.warning("shopify_webhook_missing_hmac")
            raise HTTPException(status_code=403, detail="Missing HMAC header")

        expected = base64.b64encode(
            hmac.new(
                settings.shopify_webhook_secret.encode("utf-8"),
                body,
                hashlib.sha256
            ).digest()
        ).decode()

        if not hmac.compare_digest(expected, header_val):
            logger.warning("shopify_webhook_signature_invalid")
            await services.db_service.log_security_event(
                "invalid_shopify_signature",
                get_remote_address(request),
                {"header_prefix": header_val[:20], "body_length": len(body)}
            )
            raise HTTPException(status_code=403, detail="Invalid signature")

        request.state.verified_body = body
        return body

    except HTTPException:
        raise
    except Exception as e:
        logger.error("verify_shopify_signature_error", error=str(e))
        raise HTTPException(status_code=500, detail="Shopify signature verification failed")
# ==================== ROUTERS ====================
v1_router = APIRouter(prefix=f"/api/{settings.api_version}")

# ==================== MESSAGE PROCESSING ====================
async def process_message(phone_number: str, message: str, message_type: str = "text", quoted_wamid: Optional[str] = None) -> Optional[str]:
    """
    Processes a message, gets a response, but does not send it.
    Returns None if a message was sent directly by a handler.
    """
    try:
        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)

        # --- 1. PACKING DEPARTMENT WORKFLOW (DEPRECATED) ---
        # All packing operations are now handled by dedicated API endpoints for the dashboard.
        # This block now simply catches any messages from the packing dept number and redirects them.
        if clean_phone == settings.packing_dept_whatsapp_number:
            logger.info("message_from_packing_dept_ignored", message=message)
            # Provide a helpful message in case a packer tries to use the old chat commands.
            return "All packing operations are now handled on the dashboard. Please use the dashboard link to manage orders."

        # --- 2. REGULAR CUSTOMER CONVERSATION FLOW ---
        AFFIRMATIVE_RESPONSES = {"yes", "sure", "ok", "okay", "yep", "y"}
        NEGATIVE_RESPONSES = {"no", "nope", "nah", "not really"}
        clean_message_for_check = message.lower().strip()
        last_question = await services.state_service.get_last_bot_question(clean_phone)

        if last_question:
            await services.state_service.clear_last_bot_question(clean_phone)
            customer = await get_or_create_customer(clean_phone)

            if last_question == "offer_bestsellers":
                if clean_message_for_check in AFFIRMATIVE_RESPONSES:
                    await handle_bestsellers(customer)
                    return None
            elif last_question == "offer_unfiltered_products":
                if clean_message_for_check in AFFIRMATIVE_RESPONSES:
                    await handle_show_unfiltered_products(customer)
                    return None
            
            if clean_message_for_check in NEGATIVE_RESPONSES:
                return "No problem! Let me know if there's anything else I can help you find. ✨"
        
        customer = await get_or_create_customer(clean_phone)
        intent = await analyze_intent(message, message_type, customer, quoted_wamid)
        response = await route_message(intent, clean_phone, message, customer, quoted_wamid)
        
        if response is None:
            return None
        
        return response[:4096]
        
    except Exception as e:
        logger.error("message_processing_error", phone=phone_number, error=str(e), exc_info=True)
        return "I apologize, but I'm experiencing technical difficulties."

async def get_or_create_customer(phone_number: str) -> Dict:
    """Enhanced customer management with proper caching and serialization."""
    try:
        cache_key = f"customer:v2:{phone_number}"
        cached_customer = await services.cache_service.get(cache_key)
        if cached_customer:
            # ✅ FIX: Parse the JSON string from Redis back into a Python dictionary before returning.
            return json.loads(cached_customer)
        
        customer_data = await services.db_service.get_customer(phone_number)
        if customer_data:
            # Use json.dumps with default=str to handle non-serializable types like ObjectId
            serialized_customer = json.dumps(customer_data, default=str)
            await services.cache_service.set(cache_key, serialized_customer, ttl=1800)
            # Return the parsed dictionary, not the string
            return json.loads(serialized_customer)
        
        new_customer = {
            "id": str(uuid.uuid4()), "phone_number": phone_number,
            "created_at": datetime.utcnow(), "conversation_history": [],
            "preferences": {}, "last_interaction": datetime.utcnow(),
            "total_messages": 0, "favorite_categories": []
        }
        await services.db_service.create_customer(new_customer)
        
        # Also serialize and cache the new customer profile
        serialized_new_customer = json.dumps(new_customer, default=str)
        await services.cache_service.set(cache_key, serialized_new_customer, ttl=1800)
        active_customers_gauge.inc()
        return json.loads(serialized_new_customer)

    except Exception as e:
        logger.error("get_or_create_customer_error", phone=phone_number, error=str(e))
        # Return a default, safe structure in case of error
        return {
            "id": str(uuid.uuid4()), "phone_number": phone_number,
            "created_at": datetime.utcnow(), "conversation_history": [],
            "preferences": {}, "last_interaction": datetime.utcnow()
        }



async def analyze_intent(message: str, message_type: str, customer: Dict, quoted_wamid: Optional[str] = None) -> str:
    """
    Main intent analysis orchestrator.
    Prioritizes specific actions (interactive taps, visual search) before falling back to text analysis.
    """
    if not message:
        return "general"
        
    message_lower = message.lower().strip()

    # 1. Check for specific interactive button taps FIRST.
    if message_type == "interactive":
        return _analyze_interactive_intent(message)

    # 2. Check for a visual search command.
    if message.startswith("visual_search_"):
        return "visual_search"

    # 3. THEN, check for a general text reply to a product.
    if quoted_wamid:
        last_product = await services.state_service.get_last_single_product(customer["phone_number"])
        if last_product:
            return "contextual_product_question"
    
    # 4. Fallback to analyzing the text content using the rules engine.
    return analyze_text_intent(message_lower)


def _analyze_interactive_intent(message: str) -> str:
    """Analyze intent for interactive messages based on their prefix."""
    # This dictionary should already exist in your file, but is included for completeness
    INTERACTIVE_PREFIXES = {
        "buy_": "interactive_button_reply",
        "more_": "interactive_button_reply",
        "similar_": "interactive_button_reply",
        "option_": "interactive_button_reply",
        "product_": "product_detail"
    }
    for prefix, intent in INTERACTIVE_PREFIXES.items():
        if message.startswith(prefix):
            return intent
    return "interactive_response"

# ==============================================================================
# ---- INTENT ANALYSIS RULES ENGINE ----
# ==============================================================================
# This block defines the primary "rules engine" for understanding text messages.
# The 'INTENT_RULES' list is processed in order from top to bottom, so the
# placement of each rule defines its priority.
#
# Each rule is a tuple with three parts:
# 1. A set of single-word tokens (fast check using set intersection).
# 2. A list of multi-word phrases (slower check using 'in' operator).
# 3. The final intent string to be returned if a match is found.
#
# This structure separates the intent logic from the execution code, making it
# highly maintainable and easy to update.

# Precompiled regex for performance
WORD_RE = re.compile(r'\w+')

# Organized intent rules with single-word tokens vs multi-word phrases
# In server.py, replace your INTENT_RULES list with this one.

INTENT_RULES = [
    # =================================================================
    # Group 1: High-Priority Commands (Directly trigger product lists)
    # =================================================================
    # 1. Latest Arrivals
    ({
        "latest", "new", "newest", "recent", "arrivals", "fresh", "just", "added"
    }, [], "latest_arrivals_inquiry"),
    
    # 2. Bestsellers
    ({
        "bestseller", "popular", "trending", "top", "selling", "favorite"
    }, [
        "best selling", "best sellers"
    ], "bestseller_inquiry"),
    
    # =================================================================
    # Group 2: Transactional & Support Inquiries (High user importance)
    # =================================================================
    
    # ✅ NEW: High-priority rule for human escalation.
    # 3. Human Escalation
    ({
        "human", "agent", "person", "representative", "someone"
    }, [
        "talk to human", "speak to a person", "talk to someone", "customer service"
    ], "human_escalation"),

    # 4. General Customer Support & Complaints
    ({
        "help", "support", "problem", "issue", "complaint", "refund", "return", 
        "exchange", "cancel", "payment", "billing", "damaged", "broken", "defective",
        "wrong", "incorrect", "bad", "poor", "dull", "unfortunate", "nonsense"
    }, [
        "speak to someone", "talk to agent", "not the same",
        "wrong item", "bad delivery", "poor quality"
    ], "support"),

    # 5. Shipping & Delivery Inquiry
    ({
        "shipping", "shipped", "ship", "deliver", "courier", "cost", "charge", 
        "charges", "fee", "fees", "policy", "policies", "before", "urgent", 
        "asap", "tomorrow", "today", "rush", "express", "fast", "quick", "deadline"
    }, [
        "shipping cost", "shipping policy", "delivery time", "how long", "when it will be delivered",
        "needed by", "out for delivery", "in transit", "on the way"
    ], "shipping_inquiry"),
    
    # 6. Existing Order Status
    ({
        "tracking", "dispatched", "delayed", "cancelled", 
        "processing", "confirmed", "pending", "status"
    }, [
        "where is my order", "order status", "track my order", "shipping status", "my order"
    ], "order_inquiry"),
    
    # =================================================================
    # Group 3: Informational Inquiries (Company & Product Details)
    # =================================================================
    # 7. Contact Details
    ({
        "contact", "phone", "email", "address", "location", "store", "visit"
    }, [
        "how to contact", "get in touch", "customer care"
    ], "contact_inquiry"),
    
    # 8. Customer Reviews
    ({
        "review", "reviews", "rating", "ratings", "feedback", "testimonial", "testimonials"
    }, [
        "what do people say", "customer reviews", "google reviews"
    ], "review_inquiry"),
    
    # 9. Discounts & Offers
    ({
        "discount", "offer", "sale", "coupon", "deal", "promo", "code"
    }, [
        "any offers", "current deals", "discount code"
    ], "discount_inquiry"),

    # 10. Reseller & WhatsApp Group Inquiry
    ({
        "reseller", "reselling", "broadcast", "group"
    }, [
        "reseller group", "whatsapp group"
    ], "reseller_inquiry"),

    # 11. Bulk & Wholesale Inquiry
    ({
        "wholesale", "bulk"
    }, [
        "bulk order", "buy in bulk", "bulk pricing"
    ], "bulk_order_inquiry"),
    
    # 12. Price Feedback
    ({
        "expensive", "cheap", "costly", "affordable", "budget", "pricey", "reasonable",
        "high", "low", "fair"
    }, [
        "too much", "worth it", "value for money", 
        "overpriced", "good deal"
    ], "price_feedback"),
    
    # 13. Price Inquiry
    ({
        "price", "cost", "much", "rate", "rupees", "rs", "₹"
    }, [
        "how much", "what is the price", "whats the price"
    ], "price_inquiry"),
    
    # 14. Product Details
    ({
        "size", "fit", "adjustable", "length", "diameter", "measurement", "loose", "tight"
    }, [
        "too big", "too small", "what size", "ring size", "chain length"
    ], "product_inquiry"),
    
    # 15. Stock & Availability
    ({
        "stock", "available", "inventory", "restock"
    }, [
        "in stock", "out of stock", "sold out", "back in stock", "when available"
    ], "stock_inquiry"),
    
    # =================================================================
    # Group 4: Search & Conversational Flow (Lower priority, broad)
    # =================================================================
    # 16. More Results
    ({
        "more", "other", "different", "alternatives", "similar", "else"
    }, [
        "show more", "any other", "something else", "more options"
    ], "more_results"),
    
    # 17. Product Search (Broadest - keep last)
    ({
        "earring", "earrings", "necklace", "necklaces", "ring", "rings", 
        "bracelet", "bracelets", "bangle", "bangles", "pendant", "pendants", 
        "chain", "chains", "jhumka", "jhumkas", "set", "sets", "gold", 
        "silver", "diamond", "ruby", "emerald", "sapphire", "pearl", 
        "navaratna", "jewelry", "jewellery"
    }, [], "product_search"),
]

# Conversational patterns (handled separately due to context requirements)
GREETING_KEYWORDS_SET = {
    "hi", "hello", "hey", "morning", "afternoon", "evening", "namaste"
}

THANK_KEYWORDS_SET = {
    "thanks", "thank", "grateful", "appreciate", "thankyou"
}

HIGH_PRIORITY_CONTEXT_WORDS = {
    "order", "track", "return", "refund", "shipping", "delivery", 
    "payment", "cancel", "exchange", "problem", "issue", "help",
    "urgent", "deadline", "before", "by", "needed"
}

def analyze_text_intent(message_lower: str) -> str:
    """
    Analyzes intent for text messages with optimized performance and enhanced context awareness.
    Uses precompiled regex, set operations, and comprehensive pattern matching.
    """
    # Cache message_words for reuse - significant performance improvement
    message_words = set(WORD_RE.findall(message_lower))
    
    # Process intent rules in priority order
    for single_word_patterns, multi_word_patterns, intent in INTENT_RULES:
        # Check single-word patterns using fast set intersection
        if single_word_patterns and not message_words.isdisjoint(single_word_patterns):
            logger.debug("intent_detected_single_word", 
                        message=message_lower, 
                        intent=intent,
                        matched_words=message_words & single_word_patterns)
            return intent
        
        # Check multi-word patterns using substring search
        if multi_word_patterns and any(pattern in message_lower for pattern in multi_word_patterns):
            matched_patterns = [p for p in multi_word_patterns if p in message_lower]
            logger.debug("intent_detected_multi_word", 
                        message=message_lower, 
                        intent=intent,
                        matched_patterns=matched_patterns)
            return intent
    
    # Handle conversational responses with context awareness
    is_greeting = not message_words.isdisjoint(GREETING_KEYWORDS_SET)
    has_high_priority_context = not message_words.isdisjoint(HIGH_PRIORITY_CONTEXT_WORDS)
    
    if is_greeting and not has_high_priority_context:
        logger.debug("intent_detected", message=message_lower, intent="greeting")
        return "greeting"
    
    if not message_words.isdisjoint(THANK_KEYWORDS_SET):
        logger.debug("intent_detected", message=message_lower, intent="thank_you")
        return "thank_you"
    
    # Final fallback
    logger.debug("intent_detected", message=message_lower, intent="general")
    return "general"

# ==============================================================================
# ---- END OF INTENT ANALYSIS RULES ENGINE ----
# ==============================================================================


async def route_message(intent: str, phone_number: str, message: str, customer: Dict, quoted_wamid: Optional[str] = None) -> Optional[str]:
    """
    Directs the user's message to the appropriate handler with the corrected priority order.
    """
    try:
        # High-priority product and context intents
        if intent == "product_search":
            return await handle_product_search(message, customer)
        if intent == "contextual_product_question":
            return await handle_contextual_product_question(message, customer)
        if intent == "latest_arrivals_inquiry":
            return await handle_latest_arrivals(customer)
        if intent == "bestseller_inquiry":
            return await handle_bestsellers(customer)
        if intent == "visual_search":
            return await handle_visual_search(message, customer)
        if intent == "interactive_button_reply":
            return await handle_interactive_button_response(message, customer)
        if intent == "product_detail":
            return await handle_product_detail(message, customer)
        if intent == "more_results":
            return await handle_more_results(message, customer)

        # Primary user inquiries
        if intent == "human_escalation":
            return await handle_human_escalation(customer)
        if intent == "order_inquiry":
            return await handle_order_inquiry(phone_number, customer)
        if intent == "support":
            return await handle_support_request(message, customer)
        if intent == "shipping_inquiry":
            return await handle_shipping_inquiry(message, customer)
            
        # Informational and feedback intents
        if intent == "contact_inquiry":
            return await handle_contact_inquiry(message, customer)
        if intent == "review_inquiry":
            return await handle_review_inquiry(message, customer)
        if intent == "reseller_inquiry":
            return await handle_reseller_inquiry(message, customer)
        if intent == "bulk_order_inquiry":
            return await handle_bulk_order_inquiry(message, customer)
        if intent == "discount_inquiry":
            return await handle_discount_inquiry(message, customer)
        if intent == "price_feedback":
            return await handle_price_feedback(message, customer)
        if intent == "price_inquiry":
            return await handle_price_inquiry(message, customer)

        # ✅ FIX: Placed conversational intents just before the final AI fallback.
        # Conversational filler
        if intent == "greeting":
            return await handle_greeting(phone_number, customer)
        if intent == "thank_you":
            return await handle_thank_you(customer)
            
        # Final fallback for any other general inquiry
        if intent == "general":
             return await handle_general_inquiry(message, customer)

        # Ultimate fallback if intent is somehow not covered
        return await handle_general_inquiry(message, customer)

    except Exception as e:
        logger.error("route_message_error", intent=intent, error=str(e), exc_info=True)
        return "I'm sorry, I encountered a technical issue. Please try your request again in a moment."


# ==============================================================================
# ---- FINAL, CORRECTED SEARCH AND AI HANDLER BLOCK ----
# ==============================================================================

# In server.py, replace your existing SearchConfig and QueryBuilder classes with this full block.

  

# This is your controlled vocabulary of known keywords.
# IMPORTANT: You must update this list if you add new product types or materials.
VALID_KEYWORDS = [
    "ruby", "necklace", "earring", "bangle", "bracelet", "ring", "pendant",
    "choker", "chain", "set", "jhumka", "kundan", "victorian", "oxidised",
    "matte", "cz", "nakshi", "layered", "short", "long", "bridal", "stone",
    "diamond", "emerald", "sapphire", "pearl", "gold", "silver", "jewelry"
]

@dataclass
class SearchConfig:
    """Configuration for search behavior"""
    customer: Optional[Dict] = None
    STOP_WORDS: Set[str] = None
    QUESTION_INDICATORS: Set[str] = None
    CATEGORY_EXCLUSIONS: Dict[str, List[str]] = None
    MIN_WORD_LENGTH: int = 2
    MAX_SEARCH_RESULTS: int = 5
    QA_RESULT_LIMIT: int = 1
    
    def __post_init__(self):
        if self.STOP_WORDS is None:
            self.STOP_WORDS = {
                "a", "about", "an", "any", "are", "authentic", "buy", "can", "do", 
                "does", "find", "for", "get", "genuine", "give", "have", "help", 
                "how", "i", "im", "is", "looking", "material", "me", "need", 
                "please", "quality", "real", "send", "show", "some", "tell", 
                "to", "want", "what", "when", "where", "which", "why", "you"
            }
        if self.QUESTION_INDICATORS is None:
            self.QUESTION_INDICATORS = {
                "what", "are", "is", "does", "do", "how", "why", "which", 
                "when", "where", "real", "genuine", "authentic", "quality", "material"
            }
        if self.CATEGORY_EXCLUSIONS is None:
            self.CATEGORY_EXCLUSIONS = {
                "earring": ["necklace", "set", "haram", "bracelet"],
                "necklace": ["earring", "bracelet", "ring"],
                "bangle": ["necklace", "set", "earring", "ring"],
                "ring": ["necklace", "set", "earring", "bracelet"]
            }

class QueryBuilder:
    """Final QueryBuilder with fuzzy keyword correction, price filters, and flexible search logic."""
    def __init__(self, config: SearchConfig, customer: Optional[Dict] = None):
        self.config = config
        self.customer = customer
        self.plural_mappings = {
            "necklaces": "necklace", "earrings": "earring", "bangles": "bangle",
            "bracelets": "bracelet", "rings": "ring", "pendants": "pendant", "charms": "charm", 
            "chains": "chain", "anklets": "anklet", "chokers": "choker", "sets": "set", 
            "collections": "collection", "pieces": "piece", "jhumkas": "jhumka", "diamonds": "diamond",
            "rubies": "ruby", "emeralds": "emerald", "sapphires": "sapphire", "pearls": "pearl", 
            "stones": "stone", "accessories": "accessory"
        }

    def _fuzzy_correct_keywords(self, keywords: List[str]) -> List[str]:
        corrected = []
        for kw in keywords:
            match, score, _ = process.extractOne(kw, VALID_KEYWORDS, scorer=fuzz.token_sort_ratio)
            if score >= 80:
                if kw != match:
                    logger.info("fuzzy_keyword_correction", original=kw, corrected=match, score=score)
                corrected.append(match)
            else:
                corrected.append(kw)
        return corrected

    def _extract_keywords(self, message: str) -> List[str]:
        words = [
            word.lower().strip() for word in re.findall(r'\b\w+\b', message.lower())
            if len(word) >= self.config.MIN_WORD_LENGTH and word.lower() not in self.config.STOP_WORDS
        ]
        normalized = self._normalize_keywords(words)
        return self._deduplicate_keywords(normalized)

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        mapped = [self.plural_mappings.get(keyword, keyword) for keyword in keywords]
        return self._fuzzy_correct_keywords(mapped)

    def _deduplicate_keywords(self, words: List[str]) -> List[str]:
        return list(dict.fromkeys(words))

    def _build_prioritized_query(self, keywords: List[str]) -> str:
        if not keywords:
            return ""
        field_clauses = [f"(title:{kw}* OR tag:{kw} OR product_type:{kw})" for kw in keywords]
        return " OR ".join(field_clauses)

    def _apply_exclusions(self, query: str, keywords: List[str]) -> str:
        if not query or not keywords:
            return query

        primary_category = keywords[0]
        exclusions = self.config.CATEGORY_EXCLUSIONS.get(primary_category, [])
        
        if exclusions:
            exclusion_query = ' AND NOT ' + ' AND NOT '.join([f"(title:{ex} OR tag:{ex} OR product_type:{ex})" for ex in exclusions])
            query += exclusion_query
            logger.info("applied_query_exclusions", category=primary_category, exclusions=exclusions)

        return query

    def build_query_parts(self, message: str) -> Tuple[str, Optional[Dict]]:
        price_filter, price_words = self._parse_price_filter(message)
        
        message_for_text_search = message
        if price_words:
            for word in price_words:
                message_for_text_search = message_for_text_search.replace(word, "")

        keywords = self._extract_keywords(message_for_text_search)
        text_query = self._build_prioritized_query(keywords)
        
        text_query_with_exclusions = self._apply_exclusions(text_query, keywords)
        
        return text_query_with_exclusions, price_filter

    def _parse_price_filter(self, message: str) -> Tuple[Optional[Dict], List[str]]:
        less_than_match = re.search(r'\b(under|below|less than|<)\s*₹?(\d+k?)\b', message, re.IGNORECASE)
        if less_than_match:
            price_str = less_than_match.group(2).replace('k', '000')
            return {"price": {"lessThan": float(price_str)}}, less_than_match.group(0).split()

        greater_than_match = re.search(r'\b(over|above|more than|>)\s*₹?(\d+k?)\b', message, re.IGNORECASE)
        if greater_than_match:
            price_str = greater_than_match.group(2).replace('k', '000')
            return {"price": {"greaterThan": float(price_str)}}, greater_than_match.group(0).split()
            
        return None, []

class QuestionDetector:
    def __init__(self, config: SearchConfig):
        self.config = config
    def is_question(self, message: str) -> bool:
        message_lower = message.lower()
        words = set(message_lower.split())
        return not words.isdisjoint(self.config.QUESTION_INDICATORS) or '?' in message


class AIAnswerGenerator:
    """Generates AI-powered answers for product questions"""
    def create_qa_prompt(self, product, user_question: str) -> str:
        return f"""You are FeelOri's jewelry expert assistant. Answer the customer's question using ONLY the product information provided below. Be helpful, accurate, and concise.

PRODUCT INFORMATION:
Title: {product.title}
Description: {product.description or 'No description available'}
Tags: {', '.join(product.tags) if product.tags else 'No tags available'}
Price: {getattr(product, 'price', 'Contact for pricing')}

CUSTOMER QUESTION: "{user_question}"

INSTRUCTIONS:
- Answer based ONLY on the provided product information
- If the information isn't available, say "I don't have that specific information, but I can help you contact our team"
- Be friendly and professional
- Keep the answer concise (2-3 sentences max)

ANSWER:"""


async def handle_product_search(message: str, customer: Dict) -> Optional[str]:
    """Final, corrected product search handler with intelligent filter responses."""
    try:
        config = SearchConfig()
        query_builder = QueryBuilder(config, customer=customer)
        
        text_query, price_filter = query_builder.build_query_parts(message)
        
        if not text_query and not price_filter:
            await _handle_unclear_request(customer, message)
            return None

        filtered_products, unfiltered_count = await services.shopify_service.get_products(
            query=text_query,
            filters=price_filter,
            limit=config.MAX_SEARCH_RESULTS
        )
        
        if not filtered_products:
            if unfiltered_count > 0 and price_filter:
                price_str = ""
                price_cond = price_filter.get("price", {})
                if "lessThan" in price_cond:
                    price_str = f"under ₹{price_cond['lessThan']}"
                elif "greaterThan" in price_cond:
                    price_str = f"over ₹{price_cond['greaterThan']}"

                response = (f"I found {unfiltered_count} necklace(s), but none are {price_str}. 😔\n\n"
                            f"Would you like to see the ones I found, regardless of price?")
                
                await services.state_service.set_last_search(customer["phone_number"], query=message, page=1)
                await services.state_service.set_last_bot_question(customer["phone_number"], "offer_unfiltered_products")
                await services.whatsapp_service.send_message(customer["phone_number"], response)
                return None
            else:
                await _handle_no_results(customer, message)
                return None

        await _handle_standard_search(filtered_products, message, customer)
        return None
        
    except Exception as e:
        logger.error(f"Error in product search for customer {customer.get('phone_number', 'unknown')}: {e}", exc_info=True)
        await _handle_error(customer)
        return None

async def handle_show_unfiltered_products(customer: Dict) -> Optional[str]:
    """
    Retrieves the last search query from state and re-runs it without price filters.
    """
    try:
        phone_number = customer["phone_number"]
        last_search = await services.state_service.get_last_search(phone_number)
        
        if not last_search or not last_search.get("query"):
            return "I'm sorry, I've lost the context of your last search. Could you please try searching again?"

        original_message = last_search["query"]
        
        config = SearchConfig()
        query_builder = QueryBuilder(config, customer=customer)
        text_query, _ = query_builder.build_query_parts(original_message)

        products, _ = await services.shopify_service.get_products(
            query=text_query,
            filters=None,
            limit=config.MAX_SEARCH_RESULTS
        )

        if not products:
            return f"I'm sorry, I still couldn't find any results for '{original_message}'."

        await _handle_standard_search(products, original_message, customer)
        return None

    except Exception as e:
        logger.error(f"Error in handle_show_unfiltered_products: {e}", exc_info=True)
        return "I encountered a problem trying to retrieve those results. Please try your search again."


async def _handle_no_results(customer: Dict, original_query: str):
    """Provide intelligent, multi-step response when no products are found."""
    try:
        config = SearchConfig()
        query_builder = QueryBuilder(config, customer=customer)
        keywords = query_builder._extract_keywords(original_query)
        
        search_category = _identify_search_category(keywords)
        
        # This sends the first message, e.g., "we don't have bangles, but we have sets..."
        initial_response = _generate_contextual_response(search_category, keywords)
        await services.whatsapp_service.send_message(customer["phone_number"], initial_response)
        
        customer_context = {} 
        # This checks if a second, follow-up suggestion message should be sent
        if _should_offer_alternatives(search_category, customer_context):
            await asyncio.sleep(1.5) 
            
            alternative_suggestion = _generate_alternative_suggestions(search_category, keywords)
            await services.whatsapp_service.send_message(customer["phone_number"], alternative_suggestion)

        # --- THIS IS THE CORRECTED PART ---
        # After making the suggestion, save the state. This prepares the bot for a "yes/no" reply.
        await services.state_service.set_last_bot_question(customer["phone_number"], "offer_bestsellers")
        
    except Exception as e:
        logger.error(f"Error in _handle_no_results: {e}", exc_info=True)
        await _handle_error(customer)

def _generate_contextual_response(search_category: str, keywords: List[str]) -> str:
    """Generate a helpful, contextual response based on what was searched for."""
    main_term = keywords[0] if keywords else "that item"

    responses = {
        "bangles": (
            f"That's a fantastic question! 💖 While we don't carry bangles in our collection right now, "
            f"we do have some stunning **earrings and necklace sets** that would complement your look beautifully.\n\n"
            f"Would you like me to show you some of our bestselling sets? ✨"
        ),
        "rings": (
            f"That's a fantastic question! 💖 While we don't carry rings in our collection right now, "
            f"we do have some stunning **earrings and necklace sets** that would complement your look beautifully.\n\n"
            f"Would you like me to show you some of our bestselling sets? ✨"
        ),
        "bracelets": (
            f"That's a fantastic question! 💖 While we don't carry bracelets in our collection right now, "
            f"we do have some stunning **earrings and necklace sets** that would complement your look beautifully.\n\n"
            f"Would you like me to show you some of our bestselling sets? ✨"
        ),
        "hair_extensions": (
            f"That's a fantastic question! 💖 While we don't carry hair extensions in our collection right now, "
            f"we do have some stunning **earrings and necklace sets** that would complement your look beautifully.\n\n"
            f"Would you like me to show you some of our bestselling sets? ✨"
        ),
    }

    return responses.get(search_category,
        (
            f"I couldn't find any {main_term} in our collection right now. 🔍\n\n"
            f"However, we have an amazing selection of **handcrafted necklaces and earrings** that I'd love to show you!\n\n"
            f"Would you like to see our **bestsellers**? ⭐"
        )
    )

def _generate_alternative_suggestions(search_category: str, original_keywords: List[str]) -> str:
    """Generate smart alternative suggestions based on what the customer was looking for."""
    suggestions = {
        "bangles": {
            "alternatives": ["matching earrings", "layered necklace sets", "statement pieces"],
            "message": "create a coordinated festive look"
        },
        "bracelets": {
            "alternatives": ["delicate chains", "charm necklaces", "minimalist earrings"],
            "message": "achieve that same elegant, delicate style"
        },
        "rings": {
            "alternatives": ["statement earrings", "cocktail necklaces", "bold pendants"],
            "message": "get that same eye-catching sparkle"
        },
        "watches": {
            "alternatives": ["chain necklaces", "layering pieces", "everyday earrings"],
            "message": "add sophisticated finishing touches to your look"
        }
    }
    if search_category in suggestions:
        alt_data = suggestions[search_category]
        alt_list = ", ".join(alt_data["alternatives"])
        return f"Instead, might I suggest our **{alt_list}** to {alt_data['message']}? 💫"
    return "Let me show you our **bestselling collections** that might interest you! ⭐"

def _should_offer_alternatives(search_category: str, customer_context: Dict) -> bool:
    """Determine if we should offer alternatives based on customer context and search intent."""
    unavailable_categories = {"bangles", "bracelets", "rings", "watches", "anklets"}
    if search_category in unavailable_categories:
        return True
    
    if customer_context:
        recent_searches = customer_context.get("recent_searches", [])
        if any("set" in search or "matching" in search for search in recent_searches):
            return True
    return False

def _identify_search_category(keywords: List[str]) -> str:
    """
    Prefer specific product categories (bangles, earrings, rings, etc.)
    over generic set/matching terms. Only return 'sets' when no specific
    category is present.
    """
    specific = {
        "bangles": {"bangle", "bangles", "kada", "kadaa"},
        "bracelets": {"bracelet", "bracelets"},
        "rings": {"ring", "rings"},
        "necklaces": {"necklace", "necklaces", "haram", "choker", "mala"},
        "earrings": {"earring", "earrings", "jhumka", "jhumkas", "stud", "studs"},
        "anklets": {"anklet", "anklets", "payal", "paayal"},
        "hair_extensions": {"hair extension", "hair extensions", "hair extn"},
    }

    generic_sets = {"set", "sets", "matching", "combo", "pair"}

    for cat, terms in specific.items():
        if any(w in terms for w in keywords):
            return cat

    if any(w in generic_sets for w in keywords):
        return "sets"

    return "unknown"


async def _handle_unclear_request(customer: Dict, original_message: str):
    """Handle cases where the search intent is unclear."""
    response = (
        "I'd love to help you find the perfect jewelry! 💎\n\n"
        "Could you tell me what type of piece you're looking for? For example:\n"
        "• Necklaces ✨\n"
        "• Earrings 💫\n"
        "• Jewelry sets 🎊"
    )
    await services.whatsapp_service.send_message(customer["phone_number"], response)

async def _handle_question_response(product, message: str, customer: Dict, answer_generator: AIAnswerGenerator):
    """Handle AI-powered question responses"""
    await services.state_service.set_last_search(customer["phone_number"], query=message, page=1)
    prompt = answer_generator.create_qa_prompt(product, message)
    ai_answer = await services.ai_service.generate_response(prompt)
    await services.whatsapp_service.send_message(customer["phone_number"], ai_answer)
    await _send_product_card(
        products=[product], customer=customer,
        header_text="Here's the product I was referring to ✨",
        body_text="Tap to see full details"
    )

async def _handle_standard_search(products: List, message: str, customer: Dict):
    """
    Handles standard product search results and resolves pending questions with context.
    """
    await services.state_service.set_last_search(customer["phone_number"], query=message, page=1)
    if products:
        await services.state_service.set_last_product_list(customer["phone_number"], products)
    
    header_text=f"Found {len(products)} match{'es' if len(products) != 1 else ''} for you ✨"
    body_text="Tap any product for details or ask me questions!"
    
    await _send_product_card(
        products=products, 
        customer=customer,
        header_text=header_text,
        body_text=body_text
    )

    # ✅ FIX: Check for and resolve pending questions using the saved city context.
    pending_question = await services.state_service.get_pending_question(customer["phone_number"])
    if pending_question and pending_question.get("question_type") == "delivery_time_inquiry":
        await asyncio.sleep(1.5)
        
        context = pending_question.get("context", {})
        city = context.get("city")

        if city:
            # Provide a specific answer if we know the city
            contextual_answer = (
                f"Regarding your question about delivery to **{city.title()}**: once you place your order, it typically takes **3-5 business days** for delivery, as it is a metro city. 🚚✨"
            )
        else:
            # Provide a general answer if no city was found
            contextual_answer = (
                "Regarding your delivery question: once you place your order, it typically takes **3-5 business days** for metro cities and **5-7 business days** for other areas. 🚚✨"
            )
            
        await services.whatsapp_service.send_message(customer["phone_number"], contextual_answer)

async def _send_product_card(products: List, customer: Dict, header_text: str, body_text: str):
    """Send product card with fallback handling"""
    catalog_id = await services.whatsapp_service.get_catalog_id()
    product_items = [{"product_retailer_id": p.sku} for p in products if p.sku]
    await services.whatsapp_service.send_multi_product_message(
        to=customer["phone_number"], header_text=header_text, body_text=body_text,
        footer_text="Powered by FeelOri", catalog_id=catalog_id,
        section_title="Products", product_items=product_items, fallback_products=products
    )

async def _handle_error(customer: Dict):
    """Handle unexpected errors gracefully"""
    error_message = (
        "Sorry, I'm having trouble searching right now. 😔\n"
        "Please try again in a moment or contact our support team."
    )
    await services.whatsapp_service.send_message(customer["phone_number"], error_message)


# --- HANDLER FUNCTIONS ---

# Add this new function in the "HANDLER FUNCTIONS" section
async def handle_price_feedback(message: str, customer: Dict) -> str:
    """
    Handles user feedback about pricing with empathy and offers solutions.
    """
    return (
        "Thank you for your feedback on our prices. We believe in using high-quality materials and handcrafted techniques to create unique, lasting pieces. ✨\n\n"
        "To help you find the perfect item that fits your budget, I can show you:\n\n"
        "1. Our products currently **on sale**.\n"
        "2. Items under a certain price (e.g., *'show me earrings under ₹1000'*).\n\n"
        "What would you prefer?"
    )

async def handle_contextual_product_question(message: str, customer: Dict) -> Optional[str]:
    """
    Handles questions asked in reply to a specific product message.
    If the question is a new search, it re-routes to the search handler.
    """
    phone_number = customer["phone_number"]
    last_product = await services.state_service.get_last_single_product(phone_number)

    if not last_product:
        return "I'm sorry, I've lost the context of which product you're asking about. Could you search for it again?"

    # Use the QueryBuilder to get clean keywords from the user's message
    config = SearchConfig()
    query_builder = QueryBuilder(config, customer=customer)
    keywords = query_builder._extract_keywords(message)
    
    # Use our new "specific-first" category detector
    category = _identify_search_category(keywords)

    # Re-route to the main search handler if the user is asking for a different product type.
    # We exclude "sets" because a request for a "matching set" is about the current product.
    excluded_search_categories = {"sets"}
    if category != "unknown" and category not in excluded_search_categories:
        logger.info("contextual_question_rerouted_to_search",
                    phone=phone_number, category=category, product=last_product.title)
        return await handle_product_search(message, customer)

    # --- Fallback to AI for genuine questions about the current product ---
    
    # Handle direct price queries
    if any(keyword in message.lower() for keyword in ["price", "cost", "how much", "rate"]):
        return f"The price for the *{last_product.title}* is ₹{last_product.price:,.2f}. ✨"

    # Handle direct availability queries
    if "available" in message.lower() or "stock" in message.lower():
        availability_text = last_product.availability.replace('_', ' ').title()
        return f"Yes, the *{last_product.title}* is currently {availability_text}!"

    # If it's a true question, use the AI
    answer_generator = AIAnswerGenerator()
    prompt = answer_generator.create_qa_prompt(last_product, message)
    ai_answer = await services.ai_service.generate_response(prompt)

    await services.whatsapp_service.send_message(phone_number, ai_answer)
    await _send_product_card(
        products=[last_product],
        customer=customer,
        header_text="This is the product we're discussing:",
        body_text="Tap to view details."
    )
    return None

async def handle_interactive_button_response(message: str, customer: Dict) -> Optional[str]:
    """Handles replies from the 'Buy Now', 'More Info', and 'Similar Items' buttons."""
    phone_number = customer["phone_number"]
    
    if message.startswith("buy_"):
        product_id = message.replace("buy_", "")
        return await handle_buy_request(product_id, customer)
        
    elif message.startswith("more_"):
        product_id = message.replace("more_", "")
        product = await services.shopify_service.get_product_by_id(product_id)
        if product:
            return product.description # Returns the full, clean description
        return "I'm sorry, I can't find the details for that product anymore."

    elif message.startswith("similar_"):
        product_id = message.replace("similar_", "")
        product = await services.shopify_service.get_product_by_id(product_id)
        if product and product.tags:
            return await handle_product_search(product.tags[0], customer)
        return "What kind of similar items are you looking for?"
    
    elif message.startswith("option_"):
        # The message format will be "option_{variant_gid}"
        variant_id = message.replace("option_", "")
        cart_url = services.shopify_service.get_add_to_cart_url(variant_id)
        return f"Perfect! I've added that to your cart. You can complete your purchase here:\n{cart_url}"
    
    return "I didn't understand that selection. How can I help?"


async def handle_buy_request(product_id: str, customer: Dict) -> Optional[str]:
    """Handles a buy request, checking for variants and asking for selection if needed."""
    product = await services.shopify_service.get_product_by_id(product_id)
    if not product:
        return "Sorry, that product is no longer available."

    variants = await services.shopify_service.get_product_variants(product.id)
    
    # If there are multiple options (e.g., sizes, colors)
    if len(variants) > 1:
        # Get the titles for the first 3 variants to use as button labels
        variant_options = {v['title']: v['id'] for v in variants[:3]}
        await services.whatsapp_service.send_quick_replies(
            customer["phone_number"],
            f"Please select an option for *{product.title}*:",
            variant_options # Pass the dictionary of options
        )
        return None # No text response needed, buttons were sent
        
    # If there's only one variant
    elif variants:
        cart_url = services.shopify_service.get_add_to_cart_url(variants[0]["id"])
        return f"Great choice! Add the *{product.title}* directly to your cart here:\n{cart_url}"
        
    # If there are no variants available at all
    else:
        product_url = services.shopify_service.get_product_page_url(product.handle)
        return f"This product is currently unavailable online. You can view it here: {product_url}"

# In server.py, add this new function in the "HANDLER FUNCTIONS" section.

async def handle_price_inquiry(message: str, customer: Dict) -> Optional[str]:
    """
    Handles direct questions about price, including ambiguity when a list was shown.
    """
    phone_number = customer["phone_number"]
    
    # 1. Check if the user is asking about a LIST of products
    product_list = await services.state_service.get_last_product_list(phone_number)
    if product_list and len(product_list) > 1:
        logger.info("price_query_ambiguous", phone=phone_number, list_size=len(product_list))
        # Send the clarification message and stop
        clarification_message = (
            "I just showed you a few different designs. To make sure I get you the right price, "
            "could you please tap **'View Details'** on the specific product you're interested in? "
            "From there, I can give you all the information you need! 👍"
        )
        await services.whatsapp_service.send_message(phone_number, clarification_message)
        return None

    # 2. If not a list, check for a SINGLE product
    product_to_price = None
    if product_list and len(product_list) == 1:
        product_to_price = product_list[0]
    else:
        product_to_price = await services.state_service.get_last_single_product(phone_number)

    if product_to_price:
        logger.info("price_query_intercepted", phone=phone_number, product_id=product_to_price.id)
        if not product_to_price.price or product_to_price.price <= 0:
            await services.whatsapp_service.send_message(phone_number, "The price for this item isn't available right now. Would you like me to show you similar products?")
            return None
        
        await services.whatsapp_service.send_product_detail_with_buttons(phone_number, product_to_price)
        return None
    
    # Fallback if no context
    return "I can definitely help with prices! Which product are you interested in? You can search for something like 'gold necklaces' first."

async def handle_product_detail(message: str, customer: Dict) -> Optional[str]:
    """Handles product detail requests by sending a message with interactive buttons."""
    product_id = message.replace("product_", "")
    product = await services.shopify_service.get_product_by_id(product_id)
    if product:
        # Use the new function to explicitly save a SINGLE product context.
        await services.state_service.set_last_single_product(customer["phone_number"], product)
        # Send the rich detail card
        await services.whatsapp_service.send_product_detail_with_buttons(customer["phone_number"], product)
        return None
    else:
        return "Sorry, I couldn't find details for that product."

async def handle_latest_arrivals(customer: Dict) -> Optional[str]:
    """Handles requests for latest arrivals by searching for all products and sorting by creation date."""
    phone_number = customer["phone_number"]
    
    products, _ = await services.shopify_service.get_products(
        query="", 
        limit=5, 
        sort_key="CREATED_AT"
    )

    if not products:
        return "I couldn't fetch the latest arrivals at the moment. Please try again shortly."

    await services.whatsapp_service.send_multi_product_message(
        to=phone_number,
        header_text="Here are our latest arrivals! ✨",
        body_text="Freshly added to our collection, just for you.",
        footer_text="Powered by FeelOri",
        catalog_id=None,
        section_title="New Arrivals",
        product_items=[],
        # ✅ FIX: Pass the `products` list directly
        fallback_products=products
    )
    return None

async def handle_human_escalation(customer: Dict) -> str:
    """
    Provides direct contact information when a user asks to speak to a human.
    """
    contact_number = "+91 9967680579"
    contact_email = "support@feelori.com"
    support_hours = "Monday–Saturday, 10 AM – 7 PM IST"

    return (
        "Of course. I understand you'd like to speak with a person. 🧑‍💻\n\n"
        "You can reach our dedicated customer support team here:\n\n"
        f"📞 **Call/WhatsApp:** {contact_number}\n"
        f"📧 **Email:** {contact_email}\n\n"
        f"Our support hours are **{support_hours}**. Our team will be happy to assist you!"
    )

async def handle_bestsellers(customer: Dict) -> Optional[str]:
    """Handles requests for bestsellers by searching for all products and sorting by popularity."""
    phone_number = customer["phone_number"]

    products, _ = await services.shopify_service.get_products(
        query="", 
        limit=5, 
        sort_key="BEST_SELLING"
    )

    if not products:
        return "I couldn't fetch our bestsellers at the moment. Please try again shortly."

    await services.whatsapp_service.send_multi_product_message(
        to=phone_number,
        header_text="Check out our bestsellers! 🌟",
        body_text="These are the items our customers love the most.",
        footer_text="Powered by FeelOri",
        catalog_id=None,
        section_title="Top Selling",
        product_items=[],
        # ✅ FIX: Pass the `products` list directly
        fallback_products=products
    )
    return None

async def handle_more_results(message: str, customer: Dict) -> Optional[str]:
    """
    Handles requests for "more" results intelligently by prioritizing the last viewed
    product and correctly rebuilding the search query from memory.
    """
    phone_number = customer["phone_number"]
    search_query = None
    price_filter = None # Initialize price_filter
    header_text = "Here are some more designs ✨"
    raw_query_for_display = "" # For user-facing messages

    # 1. Prioritize context from the last viewed product
    last_product = await services.state_service.get_last_single_product(phone_number)
    if last_product and last_product.tags:
        search_query = last_product.tags[0] 
        raw_query_for_display = search_query
        header_text = f"More items similar to {last_product.title}"

    # 2. If no product context, fall back to the last search query
    if not search_query:
        last_search = await services.state_service.get_last_search(phone_number)
        if last_search:
            raw_query = last_search["query"]
            raw_query_for_display = raw_query

            # --- ✅ THIS IS THE FIX ---
            config = SearchConfig()
            query_builder = QueryBuilder(config)
            # Use the correct method which returns two values
            search_query, price_filter = query_builder.build_query_parts(raw_query)
            # --- END OF FIX ---
            
            header_text = f"Here are some more designs ✨"

    # 3. If no context at all, guide the user
    if not search_query:
        return "More of what? Please search for a type of product first (e.g., 'show me necklaces')."

    # Pass both the query and any reconstructed filters to the service
    products = await services.shopify_service.get_products(search_query, limit=5, filters=price_filter)

    if not products:
        # Use the raw query for a more user-friendly message
        return f"I couldn't find any more designs for '{raw_query_for_display}'. Try searching for something else."

    # Send the new batch of products as a rich message
    await services.whatsapp_service.send_multi_product_message(
        to=phone_number,
        header_text=header_text,
        body_text="Here are a few more options based on what you were looking at.",
        footer_text="Powered by FeelOri",
        catalog_id=None,
        section_title="More Products",
        product_items=[],
        fallback_products=products
    )
    
    return None


async def handle_reseller_inquiry(message: str, customer: Dict) -> str:
    """Provides information about the reseller program and WhatsApp group."""
    contact_number = "7337294499"
    contact_email = "support@feelori.com"
    
    response = (
        "We're excited you're interested in partnering with us! ✨\n\n"
        "We offer a fantastic **Reseller Program** and an exclusive **WhatsApp Broadcast Group** for updates on new arrivals and special offers.\n\n"
        "To get more details on joining, please contact our team directly:\n\n"
        f"📞 **WhatsApp/Call:** {contact_number}\n"
        f"📧 **Email:** {contact_email}\n\n"
        "We look forward to hearing from you!"
    )
    return response

async def handle_bulk_order_inquiry(message: str, customer: Dict) -> str:
    """Provides information about placing bulk or wholesale orders."""
    contact_number = "7337294499"
    contact_email = "support@feelori.com"
    
    response = (
        "Yes, we absolutely cater to bulk and wholesale orders! 📦\n\n"
        "This is a perfect option for corporate gifting, events, or if you're looking to make a large purchase.\n\n"
        "To discuss your requirements and receive a personalized quote, please contact our team:\n\n"
        f"📞 **WhatsApp/Call:** {contact_number}\n"
        f"📧 **Email:** {contact_email}\n\n"
        "Let us know what you need, and we'll be happy to assist!"
    )
    return response

# In server.py, replace the existing handle_review_inquiry function

async def handle_review_inquiry(message: str, customer: Dict) -> str:
    """Handles inquiries about customer reviews and ratings."""
    # New Google Review link from your business profile
    google_review_url = "https://g.page/r/CbA6KqXz4_UpEBM/review"

    response_text = (
        "Thank you for asking! We'd love for you to read our reviews or leave one of your own. ✨\n\n"
        "You have two great options:\n\n"
        "1️⃣ **Google Reviews:**\n"
        "Leave a review directly on our Google Business Profile using this link:\n"
        f"➡️ {google_review_url}\n\n"
        "2️⃣ **Website Reviews:**\n"
        "You can also read and write product reviews directly on our website, **Feelori.com**.\n\n"
        "Your feedback helps us and other customers so much! 💖"
    )
    return response_text

# Add this new function in the "HANDLER FUNCTIONS" section
async def handle_discount_inquiry(message: str, customer: Dict) -> str:
    """
    Provides a brand-aligned response for questions about discounts and coupons.
    """
    return (
        "Thanks for asking! While we don't have any active discount codes at the moment, we believe in offering the best value through our commitment to quality. ✨\n\n"
        "Each of our pieces is handcrafted with premium materials and unique designs, ensuring you receive a beautiful item that lasts.\n\n"
        "Would you like to continue browsing our latest arrivals or bestsellers?"
    )

async def handle_contact_inquiry(message: str, customer: Dict) -> str:
    """
    Provides detailed contact, location, and social media information for FeelOri.
    """
    # Details extracted from your contact information page
    address = (
        "FeelOri\n"
        "Sai Nidhi, Plot 9, Krishnapuri Colony,\n"
        "Lakshmi Nagar, West Marredpally,\n"
        "Secunderabad, Hyderabad, Telangana 500026, India"
    )
    address_note = (
        "Please note: This is our registered business address. "
        "We are an online-only store and do not offer in-person shopping or order pickups."
    )
    email = "support@feelori.com"
    phone_number = "+91 9967680579"
    support_hours = "Monday – Saturday: 11:00 AM – 8:00 PM IST"

    instagram_url = "https://www.instagram.com/FeeloriOfficial/"
    facebook_url = "https://www.facebook.com/FeeloriOfficial/"
    youtube_url = "https://www.youtube.com/@FeeloriOfficial/"
    twitter_url = "https://twitter.com/FeeloriOfficial/"
    pinterest_url = "https://www.pinterest.com/FeeloriOfficial/"

    return (
        f"👋 Here is our official contact information:\n\n"
        f"📍 **Business Address:**\n{address}\n\n"
        f"*{address_note}*\n\n"
        f"📧 **Email Support:**\n{email}\n\n"
        f"📞 **Phone / WhatsApp:**\n{phone_number}\n\n"
        f"⏰ **Support Hours:**\n{support_hours}\n\n"
        f"📲 **Connect with us on Social Media:**\n"
        f"• Instagram: {instagram_url}\n"
        f"• Facebook: {facebook_url}\n"
        f"• YouTube: {youtube_url}\n"
        f"• X (Twitter): {twitter_url}\n"
        f"• Pinterest: {pinterest_url}\n\n"
        f"We're here to help! ✨"
    )

async def handle_reseller_inquiry(message: str, customer: Dict) -> str:
    """Provides information about reseller, broadcast, and bulk order opportunities."""
    contact_number = "7337294499"
    contact_email = "support@feelori.com"
    
    response = (
        "Yes, we do! We're excited you're interested in partnering with us. ✨\n\n"
        "We offer several opportunities for resellers and bulk orders:\n\n"
        "📦 **Bulk Orders:** Perfect for corporate gifting, events, or large purchases.\n"
        "🛍️ **Reseller Program:** Join our network and bring FeelOri's unique designs to your customers.\n"
        "📲 **WhatsApp Groups:** Get exclusive updates on new arrivals and special offers in our broadcast group.\n\n"
        "For all inquiries regarding reselling, bulk pricing, and joining our groups, please contact us directly:\n\n"
        f"📞 **WhatsApp/Call:** {contact_number}\n"
        f"📧 **Email:** {contact_email}\n\n"
        "Our team will be happy to assist you!"
    )
    return response

async def handle_greeting(phone_number: str, customer: Dict) -> str:
    """
    Generate personalized greeting based on customer history and context.
    """
    try:
        conversation_history = customer.get("conversation_history", [])
        conversation_count = len(conversation_history)
        
        customer_name = customer.get("name", "").strip()
        name_greeting = f"{customer_name}, " if customer_name else ""
        
        last_conversation = get_last_conversation_date(conversation_history)
        is_recent_return = (
            last_conversation and 
            (datetime.utcnow().replace(tzinfo=None) - last_conversation.replace(tzinfo=None)).days <= 7
        )
        
        if conversation_count == 0:
            return (
                f"Hello {name_greeting}welcome to Feelori! 👋 "
                f"I'm your AI shopping assistant, here to help you discover amazing fashion finds. "
                f"What style are you in the mood for today?"
            )
        elif conversation_count <= 3:
            return (
                f"Hi {name_greeting}great to see you back! 👋 "
                f"I'm here to help you find the perfect fashion pieces. "
                f"What can I help you discover today?"
            )
        elif is_recent_return:
            previous_interest = get_previous_interest(conversation_history)
            interest_context = f" Still looking for {previous_interest}?" if previous_interest else ""
            return (
                f"Welcome back, {name_greeting.rstrip(', ')}! 👋 "
                f"Ready to continue our fashion journey?{interest_context}"
            )
        else:
            return (
                f"Hey {name_greeting}welcome back to Feelori! 👋 "
                f"I've missed helping you with your fashion finds. "
                f"What's catching your eye today?"
            )
            
    except Exception as e:
        logger.error(f"Error generating greeting for {phone_number}: {e}")
        # Fallback greeting
        return (
            "Hello! Welcome to Feelori! 👋 "
            "I'm your AI shopping assistant, ready to help you find amazing fashion. "
            "What are you looking for today?"
        )

def get_last_conversation_date(conversation_history: list) -> Optional[datetime]:
    """Extract the date of the last conversation."""
    if not conversation_history:
        return None
    
    try:
        last_entry = conversation_history[-1]
        if isinstance(last_entry, dict) and "timestamp" in last_entry:
            ts = last_entry["timestamp"]
            if isinstance(ts, str):
                # Use utcfromisoformat for timezone-aware comparison
                return datetime.fromisoformat(ts.replace('Z', '+00:00'))
            elif isinstance(ts, datetime):
                return ts
    except (KeyError, ValueError, AttributeError):
        pass
    
    return None

def get_previous_interest(conversation_history: list) -> Optional[str]:
    """Extract customer's previous shopping interest from conversation history."""
    if not conversation_history:
        return None
    
    # Use the master list of keywords for consistency
    interest_keywords = {
        "earring", "necklace", "ring", "bracelet", "bangle", "pendant", 
        "chain", "jhumka", "set", "jewelry", "jewellery"
    }
    
    try:
        # Check last few conversations for interests
        for conversation in reversed(conversation_history[-3:]):
            if isinstance(conversation, dict):
                message = conversation.get("message", "").lower()
                for keyword in interest_keywords:
                    if keyword in message:
                        return f"{keyword}s" # Return a pluralized version
    except (KeyError, AttributeError):
        pass
    
    return None

async def handle_shipping_inquiry(message: str, customer: Dict) -> Optional[str]:
    """
    Provides shipping info and intelligently saves the context (including city) if more info is needed.
    """
    message_lower = message.lower()
    
    general_keywords = {"policy", "cost", "charge", "fee", "charges", "fees"}
    date_specific_keywords = {"before", "until", "by", "deadline"}

    if any(keyword in message_lower for keyword in general_keywords) or any(keyword in message_lower for keyword in date_specific_keywords):
        city_info = ""
        if "delhi" in message_lower:
            city_info = "For Delhi and other metro cities, delivery is typically within **3-5 business days!** 🏙️\n\n"
        elif any(city in message_lower for city in ["mumbai", "bangalore", "chennai", "kolkata", "hyderabad"]):
            city_info = "For metro cities, delivery is typically within **3-5 business days!** 🏙️\n\n"
        
        return (
            f"🚚 **FeelOri Shipping Policy**\n\n"
            f"{city_info}"
            f"Here are our delivery timelines for all of India:\n"
            f"📍 **Metro & Tier-1 Cities:** 3-5 business days\n"
            f"📍 **Remote & Rural Areas:** 5-7 business days\n\n"
            f"📦 **Shipping Costs**\n"
            f"🎉 **Free Shipping** on all orders above ₹999!\n"
            f"🛒 For orders below ₹999, charges are calculated at checkout.\n\n"
            # ✅ FIX: Added India Post to the list of couriers.
            f"🛡️ **Secure & Tracked**\n"
            f"All orders are securely packed and shipped with premium couriers like **Blue Dart, Delhivery, and India Post**. You'll receive a tracking link via email once your order is on its way!\n\n"
            f"For our full policy, please visit:\n"
            f"https://feelori.com/policies/shipping-policy"
        )

    cities = ["hyderabad", "delhi", "mumbai", "bangalore", "chennai", "kolkata"]
    found_city = next((city for city in cities if city in message_lower), None)

    await services.state_service.set_pending_question(
        customer["phone_number"], 
        "delivery_time_inquiry",
        context={"city": found_city}
    )
    
    return (
        "To give you an accurate delivery estimate, I'll need a little more information. 💖\n\n"
        "Could you please tell me which items you're interested in ordering? For example, you can say 'ruby necklace'."
    )
async def handle_visual_search(message: str, customer: Dict) -> Optional[str]:
    """Final visual search handler with enhanced UX and error handling."""
    try:
        media_id = message.replace("visual_search_", "").strip()
        phone_number = customer["phone_number"]
        if not media_id:
            return "I'm sorry, I couldn't read the image reference. Please try uploading the image again."

        await services.whatsapp_service.send_message(phone_number, "🔍 Analyzing your image and searching our catalog... ✨")

        image_bytes, mime_type = await services.whatsapp_service.get_media_content(media_id)
        if not image_bytes or not mime_type:
            return "I'm sorry, I had trouble downloading your image. Please try uploading it again."

        result = await services.ai_service.find_exact_product_by_image(image_bytes, mime_type)

        if not result.get('success') or not result.get('products'):
            return ("I couldn't find a good match for your image. 😔\n\n"
                    "💡 **Tips for better results:**\n"
                    "• Use clear, well-lit photos\n"
                    "• Show the jewelry clearly without distractions\n\n"
                    "Or feel free to describe what you're looking for!")

        products = result['products']
        match_type = result.get('match_type', 'similar')
        
        if match_type == 'exact':
            header_text = "🎯 Perfect Match Found!"
            body_text = "This looks like the exact product from your image!"
        elif match_type == 'very_similar':
            header_text = f"🌟 Found {len(products)} Excellent Matches"
            body_text = "These products are very similar to your image!"
        else: # similar
            header_text = f"✨ Found {len(products)} Similar Products"
            body_text = "Here are some products that match your style!"

        # Use the existing _send_product_card helper for consistency
        await _send_product_card(
            products=products,
            customer=customer,
            header_text=header_text,
            body_text=body_text
        )
        
        # Send a context-aware follow-up message
        await asyncio.sleep(1.5)
        if match_type in ['exact', 'very_similar']:
            follow_up = "❤️ Found what you were looking for? Tap a product to see more details!"
        else:
            follow_up = "🤔 Not quite right? Try uploading another image or just ask me for 'red necklaces'!"
        await services.whatsapp_service.send_message(phone_number, follow_up)

        return None
        
    except Exception as e:
        logger.error("Critical error in visual search handler", error=str(e), exc_info=True)
        return "Something went wrong during the visual search. Please try again. 😔"


# ==============================================================================
# ---- REPLACE your handle_order_inquiry function with this ENTIRE block ----
# ==============================================================================

async def handle_order_inquiry(phone_number: str, customer: Dict) -> str:
    """
    Handles order inquiries with enhanced security and error handling.
    Now correctly answers informational questions about finding an order number.
    """
    try:
        # --- NEW LOGIC TO ANSWER INFORMATIONAL QUESTIONS ---
        message_lower = customer.get("conversation_history", [{}])[-1].get("message", "").lower()
        informational_keywords = {"where", "find", "how", "get", "what is my"}
        
        if any(keyword in message_lower for keyword in informational_keywords) and "order number" in message_lower:
            return (
                "You can find your order number in the confirmation email and SMS we sent you right after you completed your purchase. 📧\n\n"
                "It typically looks like #12345. If you can't find it, I can try to look up your recent orders for you!"
            )
        # --- END OF NEW LOGIC ---

        # Security check for phone number manipulation
        security_message = _perform_security_check(phone_number, customer)
        if security_message:
            return security_message
        
        # Fetch orders (existing logic)
        orders = await services.order_repository.get_orders_by_phone(phone_number)
        
        if not orders:
            return _format_no_orders_response()
        
        return _format_orders_response(orders)
        
    except Exception as e:
        return await _handle_order_inquiry_error(e, phone_number)



def _perform_security_check(phone_number: str, customer: Dict) -> Optional[str]:
    """
    Perform security check to prevent unauthorized order access.
    Fixed to handle phone number format variations properly.
    """
    conversation_history = customer.get("conversation_history", [])
    if not conversation_history:
        return None
    
    # Get the most recent user message to check its content
    latest_message = conversation_history[-1].get("message", "")
    if not latest_message:
        return None
    
    # Find potential phone numbers mentioned in the message
    found_numbers = re.findall(r'\b\d{8,15}\b', latest_message)
    if not found_numbers:
        return None
    
    # Sanitize the sender's phone number for comparison
    sanitized_sender_phone = re.sub(r'\D', '', phone_number)
    
    # Handle Indian numbers - remove country code if present
    if sanitized_sender_phone.startswith('91') and len(sanitized_sender_phone) == 12:
        sanitized_sender_phone = sanitized_sender_phone[2:]
    
    # Check if the user mentioned a phone number that is NOT their own
    for number in found_numbers:
        sanitized_found_number = re.sub(r'\D', '', number)
        
        # Handle different formats of the same number
        is_same_number = (
            sanitized_found_number == sanitized_sender_phone or
            sanitized_found_number == sanitized_sender_phone[-10:] or
            sanitized_sender_phone.endswith(sanitized_found_number) or
            sanitized_found_number.endswith(sanitized_sender_phone)
        )
        
        if not is_same_number:
            return (
                "For your security and privacy, I can only check the order status "
                "for the phone number you are currently using to chat with me."
            )
    
    return None


def _format_no_orders_response() -> str:
    """Format response when no orders are found."""
    return (
        "I couldn't find any recent orders for this number. "
        "If you have an order number, please reply with it (e.g. #12345) or "
        "contact support@feelori.com for assistance."
    )


def _format_orders_response(orders: List[Dict]) -> str:
    """
    Format the orders response with proper sorting and limiting.
    """
    sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)
    recent_orders = sorted_orders[:3]
    
    response_parts = [f"I found {len(orders)} order(s) linked to this phone number:\n\n"]
    
    for order in recent_orders:
        order_info = _format_single_order(order)
        response_parts.append(order_info)
    
    response_parts.append("Reply with the order number if you want more details or a tracking link.")
    
    return "".join(response_parts)


def _format_single_order(order: Dict) -> str:
    """Format a single order's information."""
    order_date = _format_order_date(order.get("created_at", ""))
    
    order_number = order.get('order_number') or order.get('id')
    fulfillment_status = (order.get("fulfillment_status") or "unfulfilled").replace("_", " ").title()
    total_price = order.get("current_total_price") or order.get("total_price") or "N/A"
    currency = order.get("currency") or order.get("currency_code") or ""
    
    order_info = [
        f"🛍️ Order #{order_number}",
        f"📅 Placed: {order_date}",
        f"💰 Total: {total_price} {currency}",
        f"📋 Status: *{fulfillment_status}*"
    ]
    
    tracking_info = _format_tracking_info(order.get("fulfillments", []))
    if tracking_info:
        order_info.append(tracking_info)
    
    return "\n".join(order_info) + "\n"


def _format_order_date(created_at: str) -> str:
    """Format order creation date safely."""
    if not created_at:
        return "Unknown date"
    try:
        clean_date = created_at.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean_date)
        return dt.strftime("%d %b %Y")
    except (ValueError, AttributeError):
        return created_at


def _format_tracking_info(fulfillments: List[Dict]) -> Optional[str]:
    """Format tracking information from fulfillments, with India Post detection."""
    if not fulfillments:
        return None
    
    tracking_lines = []
    for fulfillment in fulfillments:
        tracking_number = fulfillment.get("tracking_number")
        tracking_company = fulfillment.get("tracking_company")
        
        if tracking_number:
            carrier = tracking_company or "carrier"
            
            # ✅ FIX: Automatically detect India Post if the company isn't set in Shopify.
            if not tracking_company and tracking_number.upper().endswith('IN'):
                carrier = "India Post"

            tracking_lines.append(f"🚚 Tracking: {tracking_number} via {carrier}")
    
    return "\n".join(tracking_lines) if tracking_lines else None

async def _handle_order_inquiry_error(error: Exception, phone_number: str) -> str:
    """Handle errors in order inquiry processing."""
    logger.error(
        "handle_order_inquiry_error", 
        error=str(error), 
        phone=phone_number,
        exc_info=True
    )
    
    await alerting.send_critical_alert(
        "order_lookup_failed", 
        {"phone": phone_number, "error": str(error)}
    )
    
    return (
        "I'm having trouble checking order details right now. "
        "Please contact support@feelori.com and we'll help you promptly."
    )

# ==============================================================================
# ---- END OF handle_order_inquiry REPLACEMENT BLOCK ----
# ==============================================================================

async def handle_support_request(message: str, customer: Dict) -> str:
    """
    Handles support requests including returns, refunds, exchanges, and general issues.
    """
    message_lower = message.lower().strip()
    
    # Keywords to detect a complaint about a received item
    complaint_keywords = {
        "damaged", "broken", "defective", "wrong", "incorrect", "bad", 
        "poor", "dull", "nonsense", "unfortunate", "not the same"
    }

    # Check if the message is a complaint about a product
    if any(keyword in message_lower for keyword in complaint_keywords):
        return (
            "I'm truly sorry to hear about your experience. It's very unfortunate, and we sincerely apologize that your order did not meet your expectations. Let's resolve this for you right away. 🙏\n\n"
            "For issues like damaged items or products that are not in the right condition, our policy requires a continuous unboxing video to process a return or refund. This helps us verify the issue with our courier partners.\n\n"
            "📋 **Next Steps:**\n"
            "1. Please email our support team at **support@feelori.com**.\n"
            "2. Include your **Order ID** and a description of the issue.\n"
            "3. **Attach the unboxing video** and clear photos of the item.\n\n"
            "Our team will review it and get back to you with a solution promptly. We appreciate your patience."
        )
    
    # Fallback for other general support questions
    return (
        "I understand you need assistance. For the fastest help with your issue, please email our dedicated support team with your order details.\n\n"
        "📧 **Email:** support@feelori.com\n\n"
        "They are equipped to handle all support requests, including returns, refunds, and other problems. We'll do our best to help you!"
    )


def _is_return_refund_request(message_lower: str) -> bool:
    """Check if message is about returns, refunds, or damaged items."""
    return any(keyword in message_lower for keyword in [
        "return", "refund", "damaged", "broken", "defective", 
        "exchange", "replacement", "money back", "want a return",
        "want refund", "quality issue", "not working", "faulty"
    ])


def _is_wrong_item_request(message_lower: str) -> bool:
    """Check if message is about receiving wrong item."""
    return any(phrase in message_lower for phrase in [
        "wrong item", "incorrect item", "different item", 
        "not what i ordered", "received wrong", "wrong product"
    ])


def _is_shipping_address_request(message_lower: str) -> bool:
    """Check if message is about changing shipping address."""
    return any(phrase in message_lower for phrase in [
        "change address", "modify address", "update address",
        "shipping address", "delivery address", "wrong address"
    ])


def _is_general_policy_question(message_lower: str) -> bool:
    """Check if message is asking about policies."""
    return any(word in message_lower for word in [
        "policy", "how long", "when", "timeline", "process"
    ])


def _handle_return_refund_request(message_lower: str) -> str:
    """Handle return, refund, and damage-related requests."""
    response_parts = []
    
    if "damaged" in message_lower or "broken" in message_lower or "defective" in message_lower:
        response_parts.append("😔 I'm sorry to hear your product arrived damaged. We'll help you resolve this right away!")
    else:
        response_parts.append("I understand you need assistance with a return or refund. Let me help you with that.")
    
    response_parts.extend([
        "\n📋 **FeelOri Return & Refund Process:**",
        "",
        "✅ **What We Cover:** Products damaged during transit",
        "✅ **Timeline:** Report within 3 days of delivery",
        "✅ **Required:** Continuous unboxing video (mandatory)",
        "",
        "🎥 **Unboxing Video Requirements:**",
        "• Must be continuous and unedited",
        "• Show both outer packaging and product clearly",
        "• Demonstrate any damage clearly",
        "",
        "📧 **Next Steps:**",
        "1. Email: support@feelori.com within 3 days",
        "2. Include: Order ID, your name, contact info",
        "3. Attach: Your unboxing video + clear damage photos",
        "",
        "🔄 **Resolution Options (based on availability):**",
        "• **Replacement:** Same product sent free",
        "• **Alternative:** Different color/similar item", 
        "• **Full Refund:** 5-10 business days to original payment",
        "",
        "💡 **Important:** Without a valid unboxing video, we cannot process returns/refunds for transit damage.",
        "",
        "Need immediate help? Email support@feelori.com now! 📨"
    ])
    
    return "\n".join(response_parts)


def _handle_wrong_item_request() -> str:
    """Handle wrong item delivery requests."""
    return """🚫 Received the wrong item? We'll fix this immediately!

📧 **Quick Resolution:**
1. Email: support@feelori.com within 3 days
2. Include: Order ID + photo of received item
3. We'll send the correct item at NO extra cost

✅ **What Happens Next:**
• We provide return instructions (we cover shipping)
• Correct item ships immediately
• Issue resolved within 5-10 business days

Contact support@feelori.com for instant assistance! 📨"""


def _handle_shipping_address_request() -> str:
    """Handle shipping address change requests."""
    return """📦 **Address Change Request:**

✅ **If Order NOT Shipped Yet:**
Email support@feelori.com immediately with:
• Order ID
• New complete address
• We'll update it for you!

❌ **If Order Already Shipped:**
Unfortunately, address changes aren't possible once shipped.

⏰ **Time-Sensitive:** Contact us ASAP for the best chance of updating your address.

Email: support@feelori.com 📨"""


def _handle_policy_question(message_lower: str) -> str:
    """Handle general policy questions."""
    if "how long" in message_lower or "timeline" in message_lower:
        return """⏰ **FeelOri Timelines:**

🚨 **Report Issues:** Within 3 days of delivery
💰 **Refund Processing:** 5-10 business days
📦 **Replacement:** 5-10 business days after approval
📧 **Response Time:** Contact support@feelori.com

Need specific timeline info? Email support@feelori.com! 📨"""
    
    return """📋 **FeelOri Policy Summary:**

✅ **Returns:** Only for transit damage (with unboxing video)
❌ **Change of Mind:** Not accepted
🎥 **Video Required:** Mandatory for all damage claims
⏰ **Report Window:** 3 days from delivery
💰 **Refunds:** 5-10 business days to original payment

📧 **Questions?** Email support@feelori.com
🔗 **Full Policy:** feelori.com/policies/refund-policy"""


def _handle_general_support_request() -> str:
    """Handle general support requests."""
    return """👋 **FeelOri Customer Support**

I'm here to help! For the fastest resolution:

📧 **Email:** support@feelori.com
⏰ **For Damage/Wrong Items:** Report within 3 days
🎥 **Remember:** Unboxing video required for damage claims

**Common Issues:**
• Product damage → Email with unboxing video
• Wrong item → Email with photo
• Address changes → Contact immediately if not shipped
• Refund questions → Full refunds for valid claims

What specific issue can I help you with today? 🤔

For detailed assistance: support@feelori.com 📨"""


async def handle_thank_you(customer: Dict) -> str:
    return "You're very welcome! Happy to help anytime! ✨"

async def handle_general_inquiry(message: str, customer: Dict) -> str:
    try:
        context = {"conversation_history": customer.get("conversation_history", [])[-5:]}
        ai_response = await services.ai_service.generate_response(message, context)
        return ai_response
    except Exception as e:
        logger.error("handle_general_inquiry_error", error=str(e))
        return "Thanks for your message! How can I help you with our products today?"

async def update_conversation_history_safe(phone_number: str, message: str, response: Optional[str], wamid: Optional[str] = None):
    """Safely update conversation history, handling None responses."""
    if response is None:
        return
    try:
        await services.db_service.update_conversation_history(phone_number, message, response, wamid)
        cache_key = f"customer:v2:{phone_number}"
        await services.cache_service.redis.delete(cache_key)
    except Exception as e:
        logger.error("conversation_history_update_error", phone=phone_number, error=str(e))

#-------------------------------------------------------------------------------------------------------------------------------------
import numpy as np
import pickle
from PIL import Image
import io
import torch
from transformers import AutoProcessor, AutoModel
import sqlite3
from sklearn.metrics.pairwise import cosine_similarity

class VisualProductMatcher:
    """Manages exact visual product matching using image embeddings."""
    
    def __init__(self, db_path: str = "product_embeddings.db"):
        self.db_path = db_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Visual matcher will use device: {self.device}")
        self._setup_database()
        
    def _setup_database(self):
        """Initializes the SQLite database, now including a tags column."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    product_id TEXT PRIMARY KEY,
                    product_handle TEXT NOT NULL,
                    product_title TEXT,
                    image_url TEXT,
                    tags TEXT,
                    embedding BLOB NOT NULL
                )
            ''')
            conn.commit()

    async def _initialize_vision_model(self):
        """Loads the AI model for creating image embeddings."""
        if self.model and self.processor: return
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info("Visual matching model initialized successfully.")
        except Exception as e:
            logger.error("Failed to initialize vision model", error=str(e)); raise

    async def generate_image_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Generates a numerical fingerprint (embedding) for a single image."""
        try:
            await self._initialize_vision_model()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                embedding = image_features.cpu().numpy().flatten()
                return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.error("Error generating image embedding", error=str(e)); return None

    async def index_all_products(self):
        """Fetches all product images and tags from Shopify and saves their embeddings."""
        try:
            logger.info("Starting product indexing for visual search...")
            all_products = await services.shopify_service.get_products(query="", limit=250)
            indexed_count = 0
            with sqlite3.connect(self.db_path) as conn:
                for product in all_products:
                    cursor = conn.cursor()
                    cursor.execute("SELECT 1 FROM product_embeddings WHERE product_id = ?", (product.id,))
                    if cursor.fetchone() or not product.image_url: continue
                    try:
                        async with httpx.AsyncClient() as client:
                            response = await client.get(product.image_url, timeout=10.0)
                            response.raise_for_status()
                            image_bytes = response.content
                        embedding = await self.generate_image_embedding(image_bytes)
                        if embedding is not None:
                            tags_str = ",".join(product.tags)
                            cursor.execute('INSERT OR REPLACE INTO product_embeddings (product_id, product_handle, product_title, image_url, tags, embedding) VALUES (?, ?, ?, ?, ?, ?)', (product.id, product.handle, product.title, product.image_url, tags_str, pickle.dumps(embedding)))
                            indexed_count += 1
                            if indexed_count % 10 == 0: logger.info(f"Indexed {indexed_count} products...")
                    except Exception as e:
                        logger.error(f"Error indexing product {product.id}", error=str(e)); continue
                conn.commit()
            logger.info(f"Successfully indexed {indexed_count} new products.")
            return indexed_count
        except Exception as e:
            logger.error("Error during full product indexing", error=str(e)); return 0

    async def find_matching_products(self, query_image_bytes: bytes, top_k: int = 15) -> List[Dict]:
        """Finds the most visually similar products and returns them with their tags."""
        try:
            query_embedding = await self.generate_image_embedding(query_image_bytes)
            if query_embedding is None: return []
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT product_id, product_handle, product_title, tags, embedding FROM product_embeddings")
                all_products = cursor.fetchall()
                if not all_products:
                    logger.warning("No products found in the visual search index."); return []
                product_ids = [row[0] for row in all_products]; product_handles = [row[1] for row in all_products]; product_titles = [row[2] for row in all_products]; product_tags_str = [row[3] for row in all_products]; stored_embeddings = np.array([pickle.loads(row[4]) for row in all_products])
            similarities = cosine_similarity(query_embedding.reshape(1, -1), stored_embeddings)[0]
            top_k_indices = np.argsort(similarities)[-top_k:][::-1]
            results = []
            for i in top_k_indices:
                results.append({'product_id': product_ids[i], 'handle': product_handles[i], 'title': product_titles[i], 'tags': product_tags_str[i].split(',') if product_tags_str[i] else [], 'similarity_score': float(similarities[i])})
            return results
        except Exception as e:
            logger.error("Error finding similar products", error=str(e)); return []


# ==================== WEBHOOK ENDPOINTS ====================
@v1_router.get("/webhook")
async def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """WhatsApp webhook verification"""
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        logger.info("webhook_verified_successfully")
        return PlainTextResponse(hub_challenge)
    
    logger.warning("webhook_verification_failed", 
                  mode=hub_mode, 
                  token=hub_verify_token[:10] if hub_verify_token else None)
    
    raise HTTPException(status_code=403, detail="Forbidden")

async def handle_status_update(status_data: Dict):
    """Processes a status update webhook from WhatsApp with a retry mechanism."""
    try:
        wamid = status_data.get("id")
        status = status_data.get("status")
        recipient_phone_raw = status_data.get("recipient_id")
        recipient_phone = EnhancedSecurityService.sanitize_phone_number(recipient_phone_raw)
        
        if not all([wamid, status, recipient_phone]):
            logger.warning("incomplete_status_update", data=status_data)
            return

        # Retry logic to handle the race condition
        for attempt in range(3):
            result = await services.db_service.db.customers.update_one(
                {
                    "phone_number": recipient_phone, 
                    "conversation_history.wamid": wamid
                },
                {
                    "$set": {"conversation_history.$.status": status}
                }
            )
            
            if result.modified_count > 0:
                logger.info("message_status_updated", wamid=wamid, status=status, attempt=attempt + 1)
                return # Success, exit the function
            
            # If not found, wait a bit and try again
            await asyncio.sleep(1.0)  # Wait 1 second before the next attempt

        # If it still fails after all retries, then log the warning
        logger.warning("message_for_status_update_not_found", wamid=wamid, phone=recipient_phone)
            
    except Exception as e:
        logger.error("handle_status_update_error", error=str(e))

@v1_router.post("/webhook")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def handle_webhook(
    request: Request,
    verified_body: bytes = Depends(verify_webhook_signature)
):
    """Enhanced webhook handler for both messages and status updates."""
    with response_time_histogram.labels(endpoint="webhook_handler").time():
        try:
            data = json.loads(verified_body.decode())
            client_ip = get_remote_address(request)

            if not await services.rate_limiter.check_ip_rate_limit(client_ip, limit=100, window=60):
                return JSONResponse({"status": "rate_limited"}, status_code=429)

            for entry in data.get("entry", []):
                for change in entry.get("changes", []):
                    if change.get("field") != "messages":
                        continue
                    
                    value = change.get("value", {})
                    
                    # --- START OF NEW LOGIC ---
                    # Check if it's a STATUS update from WhatsApp
                    if "statuses" in value:
                        for status_data in value.get("statuses", []):
                            # Process each status update in the background
                            asyncio.create_task(handle_status_update(status_data))
                        continue # Move to the next change
                    
                    # Check if it's a MESSAGE update from a user
                    if "messages" in value:
                        for message in value.get("messages", []):
                            try:
                                if not message.get("from"):
                                    continue
                                clean_phone = EnhancedSecurityService.sanitize_phone_number(message["from"])
                                if not await services.rate_limiter.check_phone_rate_limit(clean_phone):
                                    continue
                                await process_webhook_message_enhanced(message, value)
                            except Exception as e:
                                logger.error("message_processing_loop_error", error=str(e), message_id=message.get("id"))
                    # --- END OF NEW LOGIC ---
            
            return JSONResponse({"status": "success"})
            
        except Exception as e:
            logger.error("webhook_handler_error", error=str(e))
            return JSONResponse({"status": "error"}, status_code=500)

async def process_webhook_message_enhanced(message: Dict, webhook_data: Dict):
    """Enhanced webhook message processing with reply context and name capture."""
    try:
        message_id = message.get("id")
        from_number = message.get("from")
        timestamp = message.get("timestamp")
        
        if not all([message_id, from_number, timestamp]):
            logger.warning("incomplete_webhook_message", message_id=message_id)
            return
        
        clean_phone = EnhancedSecurityService.sanitize_phone_number(from_number)
        
        if await services.message_queue.is_duplicate_message(message_id, clean_phone):
            logger.info("duplicate_message_detected", message_id=message_id, phone=clean_phone)
            return
        
        context = message.get("context", {})
        quoted_wamid = context.get("id") if context else None

        # ✅ NEW: Capture the customer's name from the webhook payload
        profile_name = webhook_data.get("contacts", [{}])[0].get("profile", {}).get("name")

        message_type = message.get("type", "text")
        message_text = ""
        
        if message_type == "text":
            message_text = EnhancedSecurityService.validate_message_content(message.get("text", {}).get("body", ""))
        elif message_type == "interactive":
            interactive = message.get("interactive", {})
            if interactive.get("type") == "list_reply":
                message_text = interactive.get("list_reply", {}).get("id", "")
            elif interactive.get("type") == "button_reply":
                message_text = interactive.get("button_reply", {}).get("id", "")
        elif message_type == "image":
            media_id = (message.get("image") or {}).get("id")
            message_text = f"visual_search_{media_id}" if media_id else "[IMAGE] received"
            caption = (message.get("image") or {}).get("caption", "")
            if caption:
                message_text += f"_caption_{caption}"
        else:
            logger.info("unsupported_message_type", type=message_type)
            return
        
        await services.message_queue.add_message({
            "message_id": message_id,
            "from_number": clean_phone,
            "message_text": message_text,
            "message_type": message_type,
            "timestamp": timestamp,
            "quoted_wamid": quoted_wamid,
            "profile_name": profile_name # Pass the name to the message queue
        })
        
    except Exception as e:
        logger.error("process_webhook_message_enhanced_error", error=str(e), message=message)



# In server.py, replace your existing function with this one.
async def send_packing_alert_background(order_payload: Dict):
    try:
        order_number = order_payload.get("order_number")
        packing_phone = settings.packing_dept_whatsapp_number
        
        # --- THIS IS THE FIX ---
        # It now checks for the number and logs a warning if it's missing.
        if not packing_phone:
            logger.warning("packing_alert_not_sent", reason="PACKING_DEPT_WHATSAPP_NUMBER is not set in the environment file.")
            return

        # The URL is now correctly read from your settings.
        dashboard_url = settings.dashboard_url 

        message_body = (
            f"🎉 New Order Received! #{order_number}\n\n"
            f"This has been added to your queue. View the full packing list here:\n"
            f"{dashboard_url}"
        )
        await services.whatsapp_service.send_message(packing_phone, message_body)
    except Exception as e:
        logger.error("send_packing_alert_background_error", error=str(e))

#  PACKING ENDPOINTS

# In server.py, add this full block of new endpoints

@v1_router.get("/packing/orders", response_model=APIResponse)
async def get_packing_orders(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides the list of all orders for the packing dashboard."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        statuses = ["Pending", "Needs Stock Check", "In Progress", "On Hold", "Completed"]
        orders_cursor = services.db_service.db.orders.find(
            {"fulfillment_status_internal": {"$in": statuses}},
        ).sort("created_at", 1)
        
        orders_list = await orders_cursor.to_list(length=200)
        
        formatted_orders = []
        for order in orders_list:
            raw_order = order.get("raw", {})
            customer = raw_order.get("customer", {})
            shipping_address = raw_order.get("shipping_address", {})
            
            line_items = []
            for item in order.get("line_items_with_images", []):
                line_items.append({
                    "quantity": item.get("quantity"),
                    "title": item.get("title"),
                    "sku": item.get("sku"),
                    "image_url": item.get("image_url", "https://placehold.co/80") 
                })

            formatted_orders.append({
                # --- BUG FIX: Use the correct numeric ID from the raw payload ---
                "order_id": raw_order.get("id"), 
                "order_number": order.get("order_number"),
                "status": order.get("fulfillment_status_internal"),
                "created_at": order.get("created_at"),
                "packer_name": order.get("packed_by"),
                "customer": {
                    "name": f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip(),
                    "phone": shipping_address.get("phone") or customer.get("phone")
                },
                "items": line_items,
                "notes": order.get("notes"),
                "hold_reason": order.get("hold_reason"),
                "problem_item_skus": order.get("problem_item_skus", []),
                # --- NEW FEATURE: Send the new data to the dashboard ---
                "previously_on_hold_reason": order.get("previously_on_hold_reason"),
                "previously_problem_skus": order.get("previously_problem_skus", [])
            })

        return APIResponse(success=True, message="Orders retrieved", data={"orders": formatted_orders})
    except Exception as e:
        logger.error("get_packing_orders_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get packing orders.")

@v1_router.post("/packing/orders/{order_id}/start", response_model=APIResponse)
async def start_packing_order(order_id: int, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from Pending to In Progress."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        # Find the order in a pending state
        order = await services.db_service.db.orders.find_one(
            {"id": order_id, "fulfillment_status_internal": {"$in": ["Pending", "Needs Stock Check"]}}
        )

        if not order:
            raise HTTPException(status_code=404, detail="Order not found or is not in a pending state.")

        # Update the status and set the timestamp for when work began
        await services.db_service.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "In Progress",
                "in_progress_at": datetime.utcnow() # Track when packing started for metrics
            }}
        )
        return APIResponse(success=True, message="Order moved to In Progress.")
    except Exception as e:
        logger.error("start_packing_error", error=str(e), order_id=order_id)
        raise HTTPException(status_code=500, detail="Could not start packing order.")

# In server.py

@v1_router.post("/packing/orders/{order_id}/requeue", response_model=APIResponse)
async def requeue_packing_order(order_id: int, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from On Hold back to Pending and records the previous hold reason and SKUs."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        order_on_hold = await services.db_service.db.orders.find_one(
            {"id": order_id, "fulfillment_status_internal": "On Hold"}
        )
        
        if not order_on_hold:
            raise HTTPException(status_code=404, detail="Order not found in 'On Hold' status.")

        # --- NEW FEATURE: Get both the reason and the SKUs from the held order ---
        previous_reason = order_on_hold.get("hold_reason", "Unknown reason")
        previous_skus = order_on_hold.get("problem_item_skus", [])
        
        await services.db_service.db.orders.update_one(
            {"id": order_id},
            {
                "$set": {
                    "fulfillment_status_internal": "Pending",
                    "previously_on_hold_reason": previous_reason,
                    "previously_problem_skus": previous_skus # Save the problem SKUs
                }, 
                "$unset": {"hold_reason": "", "problem_item_skus": "", "notes": ""}
            }
        )
        return APIResponse(success=True, message="Order moved back to Pending queue.")
    except Exception as e:
        logger.error("requeue_packing_error", error=str(e), order_id=order_id)
        raise HTTPException(status_code=500, detail="Could not re-queue order.")

@v1_router.post("/packing/orders/{order_id}/hold", response_model=APIResponse)
async def hold_packing_order(order_id: int, hold_data: HoldOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order to On Hold."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        await services.db_service.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "On Hold",
                "hold_reason": hold_data.reason,
                "notes": hold_data.notes, # Make sure notes are saved too
                # CHANGE THIS LINE
                "problem_item_skus": hold_data.problem_item_skus # Use the new plural key
            }}
        )
        return APIResponse(success=True, message="Order moved to On Hold.")
    except Exception as e:
        logger.error("hold_packing_error", error=str(e), order_id=order_id)
        raise HTTPException(status_code=500, detail="Could not hold order.")

@v1_router.post("/packing/orders/{order_id}/requeue", response_model=APIResponse)
async def requeue_packing_order(order_id: int, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from On Hold back to Pending."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        await services.db_service.db.orders.update_one(
            {"id": order_id, "fulfillment_status_internal": "On Hold"},
            {"$set": {"fulfillment_status_internal": "Pending"}, "$unset": {"hold_reason": "", "problem_item_sku": ""}}
        )
        return APIResponse(success=True, message="Order moved back to Pending queue.")
    except Exception as e:
        logger.error("requeue_packing_error", error=str(e), order_id=order_id)
        raise HTTPException(status_code=500, detail="Could not re-queue order.")


@v1_router.post("/packing/orders/{order_id}/fulfill", response_model=APIResponse)
async def fulfill_packing_order(order_id: int, fulfill_data: FulfillOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Fulfills the order in Shopify and marks it as complete."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        success, fulfillment_id = await services.shopify_service.fulfill_order(
            order_id, fulfill_data.tracking_number, fulfill_data.packer_name, fulfill_data.carrier
        )
        
        if not success:
            raise HTTPException(status_code=502, detail="Failed to fulfill order in Shopify.")

        await services.db_service.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "Completed", # Changed from "Fulfilled" to "Completed"
                "packed_by": fulfill_data.packer_name,
                "fulfillment_id": fulfillment_id,
                "fulfilled_at": datetime.utcnow()
            }}
        )

        # FINAL STEP: Send customer notification
        order_doc = await services.db_service.db.orders.find_one({"id": order_id})
        
        # --- THIS IS THE FIX ---
        # It now checks that the phone_numbers list exists AND is not empty.
        if order_doc and order_doc.get("phone_numbers"):
            customer_phone = order_doc["phone_numbers"][0]
            customer_name = order_doc.get("raw", {}).get("customer", {}).get("first_name", "")
            
            notification_message = (
                f"Great news, {customer_name}! ✨ Your FeelOri order #{order_doc.get('order_number')} has been packed and is on its way!\n\n"
                f"🚚 Tracking Number: {fulfill_data.tracking_number}\n"
                f"🏢 Carrier: {fulfill_data.carrier}\n\n"
                "We're so excited for you to receive your items! 💖"
            )
            # This task now runs safely in the background
            asyncio.create_task(
                services.whatsapp_service.send_message(customer_phone, notification_message)
            )
        # --- END OF FIX ---

        return APIResponse(success=True, message="Order fulfilled and customer notified.")
    except Exception as e:
        logger.error("fulfill_packing_error", error=str(e), order_id=order_id)
        raise HTTPException(status_code=500, detail=f"Could not fulfill order: {e}")

@v1_router.get("/packing/metrics", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packing_metrics(
    request: Request,
    current_user: dict = Depends(verify_jwt_token)
):
    """
    Provides a collection of key performance indicators (KPIs) for the packing workflow.
    """
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)

        # Define time ranges for "today"
        now = datetime.utcnow()
        start_of_day = now.replace(hour=0, minute=0, second=0, microsecond=0)

        # A single, powerful pipeline to get all metrics at once
        pipeline = [
            {
                "$facet": {
                    # Metric 1: Counts of current statuses
                    "status_counts": [
                        {"$group": {"_id": "$fulfillment_status_internal", "count": {"$sum": 1}}}
                    ],
                    # Metric 2: Packer leaderboard for today
                    "packer_leaderboard": [
                        {"$match": {"fulfillment_status_internal": "Fulfilled", "fulfilled_at": {"$gte": start_of_day}}},
                        {"$group": {"_id": "$packed_by", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ],
                    # Metric 3: Average time to pack for orders completed today
                    "avg_pack_time_seconds": [
                        {"$match": {
                            "fulfilled_at": {"$gte": start_of_day},
                            "in_progress_at": {"$exists": True}
                        }},
                        {"$project": {
                            "duration": {"$subtract": ["$fulfilled_at", "$in_progress_at"]}
                        }},
                        {"$group": {
                            "_id": None,
                            "avg_duration_ms": {"$avg": "$duration"}
                        }},
                        {"$project": { # Convert from milliseconds to seconds
                            "avg_seconds": {"$divide": ["$avg_duration_ms", 1000]},
                            "_id": 0
                        }}
                    ]
                }
            }
        ]

        results = await services.db_service.db.orders.aggregate(pipeline).to_list(1)
        
        # Format the results into a clean object
        if not results:
            return APIResponse(success=True, message="No metrics data available yet.", data={})

        data = results[0]
        
        # Process status counts
        status_counts = {item['_id']: item['count'] for item in data.get('status_counts', [])}
        
        # Process leaderboard
        leaderboard = [{"packer": item['_id'], "count": item['count']} for item in data.get('packer_leaderboard', [])]

        # Process average pack time
        avg_time_data = data.get('avg_pack_time_seconds', [{}])[0]
        avg_pack_time = round(avg_time_data.get('avg_seconds', 0), 2)

        metrics = {
            "status_counts": {
                "pending": status_counts.get("Pending", 0),
                "in_progress": status_counts.get("In Progress", 0),
                "on_hold": status_counts.get("On Hold", 0),
                "needs_stock_check": status_counts.get("Needs Stock Check", 0),
            },
            "packer_leaderboard_today": leaderboard,
            "avg_pack_time_today_seconds": avg_pack_time
        }
        
        return APIResponse(
            success=True,
            message="Packing metrics retrieved successfully.",
            data=metrics
        )

    except Exception as e:
        logger.error("get_packing_metrics_error", error=str(e), exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve packing metrics.")


# --- SHOPIFY WEBHOOK HANDLERS ---
@v1_router.post("/webhooks/shopify/orders/create")
async def shopify_orders_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    try:
        payload = json.loads(verified_body.decode("utf-8"))
        order_id = payload.get("id")

        # --- START: New Logic to get images ---
        line_items_with_images = []
        for item in payload.get("line_items", []):
            image_url = None
            if item.get("product_id"):
                image_url = await services.shopify_service.get_product_image_url(item["product_id"])
            
            line_items_with_images.append({
                "title": item.get("title"),
                "quantity": item.get("quantity"),
                "sku": item.get("sku"),
                "image_url": image_url
            })
        # --- END: New Logic ---

        needs_stock_check = False
        for item in payload.get("line_items", []):
            required_qty = item.get("quantity", 1)
            variant_id = item.get("variant_id")
            if variant_id:
                available_qty = await services.shopify_service.get_inventory_for_variant(variant_id)
                if available_qty is None or available_qty < required_qty:
                    needs_stock_check = True
                    break
        
        initial_status = "Needs Stock Check" if needs_stock_check else "Pending"

        phones = set()

        if (payload.get("customer") or {}).get("phone"):
            phones.add(payload["customer"]["phone"])
        if (payload.get("shipping_address") or {}).get("phone"):
            phones.add(payload["shipping_address"]["phone"])
        if (payload.get("billing_address") or {}).get("phone"):
            phones.add(payload["billing_address"]["phone"])

        clean_phones = []
        for p in phones:
            if p:
                try:
                    clean_phones.append(EnhancedSecurityService.sanitize_phone_number(p))
                except Exception:
                    pass

        # --- Construct the Full Document for MongoDB ---
        order_doc = {
            "id": order_id,
            "order_number": payload.get("order_number"),
            "created_at": payload.get("created_at"),
            "raw": payload,
            "line_items_with_images": line_items_with_images, # Save the items with images
            "phone_numbers": clean_phones,
            "fulfillment_status_internal": initial_status,
            "packed_by": None,
            "fulfillment_id": None,
            "in_progress_at": None,
            "fulfilled_at": None,
            "hold_reason": None,
            "notes": None,
            "problem_item_skus": [],
            "last_synced": datetime.utcnow()
        }
        
        await services.db_service.db.orders.update_one(
            {"id": order_id}, {"$set": order_doc}, upsert=True
        )
        
        asyncio.create_task(send_packing_alert_background(payload))
        return JSONResponse({"status": "ok"})
    except Exception as e:
        logger.error("shopify_orders_create_error", error=str(e), exc_info=True)
        return JSONResponse({"status": "error"}, status_code=500)



@v1_router.post("/webhooks/shopify/orders/updated")
async def shopify_orders_updated_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Shopify orders/updated webhook — update order metadata in DB."""
    try:
        payload = json.loads(verified_body.decode("utf-8"))
        order = payload
        order_id = order.get("id")
        if not order_id:
            logger.warning("shopify_order_update_missing_id")
            return JSONResponse({"status": "ignored", "reason": "missing id"}, status_code=400)

        # Build minimal update patch
        patch = {}
        if order.get("fulfillment_status") is not None:
            patch["fulfillment_status"] = order.get("fulfillment_status")
        if order.get("financial_status") is not None:
            patch["financial_status"] = order.get("financial_status")
        if order.get("total_price") is not None:
            patch["total_price"] = order.get("total_price")
        if order.get("currency") is not None:
            patch["currency"] = order.get("currency")
        # update phones similarly
        phones = set()
        cust_phone = (order.get("customer") or {}).get("phone")
        if cust_phone:
            phones.add(cust_phone)
        billing_phone = (order.get("billing_address") or {}).get("phone")
        if billing_phone:
            phones.add(billing_phone)
        shipping_phone = (order.get("shipping_address") or {}).get("phone")
        if shipping_phone:
            phones.add(shipping_phone)

        clean_phones = []
        for p in phones:
            try:
                clean_phones.append(EnhancedSecurityService.sanitize_phone_number(p))
            except Exception:
                pass

        if clean_phones:
            patch["phone_numbers"] = clean_phones

        if patch:
            patch["raw"] = order
            patch["last_synced"] = datetime.utcnow()
            await services.db_service.db.orders.update_one(
                {"id": order_id},
                {"$set": patch},
                upsert=True
            )

        logger.info("shopify_order_updated", order_id=order_id, update_keys=list(patch.keys()))
        return JSONResponse({"status": "ok"})

    except Exception as e:
        logger.error("shopify_orders_updated_error", error=str(e))
        return JSONResponse({"status": "error"}, status_code=500)

@v1_router.post("/webhooks/shopify/fulfillments/create")
async def shopify_fulfillments_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Shopify fulfillments/create webhook — store tracking numbers on the order."""
    try:
        payload = json.loads(verified_body.decode("utf-8"))
        # payload may contain 'fulfillment' or array depending on webhook config
        fulfillment = payload.get("fulfillment") or payload.get("fulfillments") or payload

        order_id = fulfillment.get("order_id") or fulfillment.get("order") and fulfillment["order"].get("id")
        if not order_id:
            logger.warning("shopify_fulfillment_missing_order_id")
            return JSONResponse({"status": "ignored", "reason": "missing order_id"}, status_code=400)

        tracking_numbers = fulfillment.get("tracking_numbers") or []
        if isinstance(tracking_numbers, str):
            tracking_numbers = [tracking_numbers]

        # Append tracking numbers and fulfillment details to order document
        await services.db_service.db.orders.update_one(
            {"id": order_id},
            {
                "$addToSet": {"tracking_numbers": {"$each": tracking_numbers}},
                "$set": {
                    "fulfillment_status": fulfillment.get("status") or "fulfilled",
                    "last_fulfillment": fulfillment,
                    "last_synced": datetime.utcnow()
                }
            },
            upsert=True
        )

        logger.info("shopify_fulfillment_processed", order_id=order_id, track_count=len(tracking_numbers))
        return JSONResponse({"status": "ok"})

    except Exception as e:
        logger.error("shopify_fulfillments_create_error", error=str(e))
        return JSONResponse({"status": "error"}, status_code=500)



@v1_router.post("/webhooks/shopify/checkouts/update")
async def shopify_checkouts_update_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """
    Handles abandoned checkout events from Shopify by scheduling a reminder message.
    """
    try:
        payload = json.loads(verified_body.decode("utf-8"))
        
        if payload.get("completed_at") is not None:
            return JSONResponse({"status": "ignored", "reason": "checkout completed"})

        phone_number = (payload.get("phone") or 
                        (payload.get("shipping_address") or {}).get("phone") or
                        (payload.get("customer") or {}).get("phone"))

        checkout_url = payload.get("abandoned_checkout_url")

        if not phone_number or not checkout_url:
            logger.info("abandoned_checkout_ignored", reason="missing phone or url")
            return JSONResponse({"status": "ignored", "reason": "missing phone or url"})

        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        
        customer = await get_or_create_customer(clean_phone)
        customer_name = customer.get("name", "there")

        message = (
            f"Hi {customer_name}! 👋\n\n"
            "It looks like you left some beautiful items in your cart at FeelOri. ✨\n\n"
            "Would you like to complete your purchase? You can return to your cart here:\n"
            f"{checkout_url}\n\n"
            "These pieces are waiting for you! 💖"
        )
        
        # ✅ FIX: Instead of sending now, schedule the message for 1 hour in the future.
        run_time = datetime.now() + timedelta(hours=1)
        
        scheduler.add_job(
            services.whatsapp_service.send_message,
            'date',
            run_date=run_time,
            args=[clean_phone, message],
            id=f"abandoned_checkout_{payload.get('id')}", # Give the job a unique ID
            replace_existing=True # Prevents duplicate jobs for the same checkout
        )
        
        logger.info("abandoned_checkout_reminder_scheduled", phone=clean_phone, run_time=run_time.isoformat())
        return JSONResponse({"status": "scheduled"})

    except Exception as e:
        logger.error("shopify_checkouts_update_error", error=str(e))
        return JSONResponse({"status": "error"}, status_code=500)

# ==================== AUTHENTICATION ENDPOINTS ====================
@v1_router.post("/auth/login", response_model=TokenResponse)
@limiter.limit(f"{settings.auth_rate_limit_per_minute}/minute")
async def login(request: Request, login_data: LoginRequest):
    """JWT-based authentication with Redis-backed security"""
    
    client_ip = get_remote_address(request)
    
    try:
        # Check if IP is locked out (Redis-backed)
        if await services.login_tracker.is_locked_out(client_ip):
            logger.warning("login_attempt_lockout", ip=client_ip)
            auth_attempts_counter.labels(status="lockout", method="password").inc()
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many failed attempts. Please try again later."
            )
        
        if not EnhancedSecurityService.verify_password(login_data.password, ADMIN_PASSWORD_HASH):
            # Record failed attempt in Redis
            await services.login_tracker.record_attempt(client_ip)
            
            logger.warning("login_attempt_failed", ip=client_ip)
            auth_attempts_counter.labels(status="failure", method="password").inc()
            
            await services.db_service.log_security_event(
                "failed_login", client_ip, {"reason": "invalid_password"}
            )
            
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Create JWT token
        access_token = jwt_service.create_access_token(
            data={"sub": "admin", "type": "access", "ip": client_ip}
        )
        
        logger.info("login_successful", ip=client_ip)
        auth_attempts_counter.labels(status="success", method="password").inc()
        
        await services.db_service.log_security_event(
            "successful_login", client_ip, {"method": "jwt"}
        )
        
        return TokenResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=settings.jwt_access_token_expire_hours * 3600
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("login_error", error=str(e), ip=client_ip)
        await alerting.send_critical_alert("Login system error", {"error": str(e)})
        raise HTTPException(status_code=500, detail="Authentication system error")

@v1_router.post("/auth/logout")
@limiter.limit("10/minute")
async def logout(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Logout endpoint with session cleanup"""
    try:
        client_ip = get_remote_address(request)
        
        # Log security event
        await services.db_service.log_security_event(
            "logout", client_ip, {"user": current_user.get("sub")}
        )
        
        logger.info("user_logout", ip=client_ip, user=current_user.get("sub"))
        
        return APIResponse(
            success=True,
            message="Logged out successfully"
        )
        
    except Exception as e:
        logger.error("logout_error", error=str(e))
        raise HTTPException(status_code=500, detail="Logout failed")

# ==================== ADMIN ENDPOINTS ====================
@v1_router.get("/admin/stats")
@limiter.limit("10/minute")
async def get_admin_stats(
    request: Request, 
    current_user: dict = Depends(verify_jwt_token)
):
    """Get system statistics with an optimized aggregation pipeline."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        # Define time windows for our queries
        now = datetime.utcnow()
        last_24_hours = now - timedelta(hours=24)
        one_week_ago = now - timedelta(days=7)
        previous_week_24h_start = one_week_ago - timedelta(hours=24)

        # A single, powerful aggregation pipeline to get all stats in one DB trip
        pipeline = [
            {
                "$facet": {
                    "customer_stats": [
                        {
                            "$group": {
                                "_id": None,
                                "total_customers": {"$sum": 1},
                                "active_24h": {
                                    "$sum": {
                                        "$cond": [{"$gte": ["$last_interaction", last_24_hours]}, 1, 0]
                                    }
                                },
                                "total_previous_week": {
                                    "$sum": {
                                        "$cond": [{"$lt": ["$created_at", one_week_ago]}, 1, 0]
                                    }
                                },
                            }
                        }
                    ],
                    "message_stats": [
                        {"$unwind": "$conversation_history"},
                        {
                            "$group": {
                                "_id": None,
                                "total_24h": {
                                    "$sum": {
                                        "$cond": [{"$gte": ["$conversation_history.timestamp", last_24_hours]}, 1, 0]
                                    }
                                },
                                "total_24h_previous_week": {
                                    "$sum": {
                                        "$cond": [
                                            {"$and": [
                                                {"$gte": ["$conversation_history.timestamp", previous_week_24h_start]},
                                                {"$lt": ["$conversation_history.timestamp", one_week_ago]}
                                            ]},
                                            1, 0
                                        ]
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        ]

        # Execute the aggregation
        results = await services.db_service.db.customers.aggregate(pipeline).to_list(1)
        
        # Extract results, providing default values if no data exists
        customer_stats = results[0]['customer_stats'][0] if results and results[0]['customer_stats'] else {"total_customers": 0, "active_24h": 0, "total_previous_week": 0}
        message_stats = results[0]['message_stats'][0] if results and results[0]['message_stats'] else {"total_24h": 0, "total_24h_previous_week": 0}

        # Assemble the final response object
        stats = {
            "customers": {
                "total": customer_stats["total_customers"],
                "total_previous_week": customer_stats["total_previous_week"],
                "active_24h": customer_stats["active_24h"],
                "active_percentage": round((customer_stats["active_24h"] / customer_stats["total_customers"] * 100) if customer_stats["total_customers"] > 0 else 0, 2)
            },
            "messages": {
                "total_24h": message_stats["total_24h"],
                "total_24h_previous_week": message_stats["total_24h_previous_week"],
                "avg_per_customer": round(message_stats["total_24h"] / customer_stats["active_24h"] if customer_stats["active_24h"] > 0 else 0, 2)
            },
            # System stats can be fetched separately as they are quick
            "system": {
                "database_status": "connected",
                "cache_status": "connected",
                "queue_size": await services.cache_service.redis.xlen("webhook_messages") if services.cache_service else 0
            },
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }
        
        return APIResponse(
            success=True,
            message="Statistics retrieved successfully",
            data=stats
        )
        
    except Exception as e:
        logger.error("admin_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve statistics")

@v1_router.get("/admin/products", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_admin_products(
    request: Request,
    query: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get product list from Shopify for the admin dashboard."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)

        products = await services.shopify_service.get_products(query or "", limit=limit)

        return APIResponse(
            success=True,
            message=f"Retrieved {len(products)} products",
            data={"products": [p.dict() for p in products]}
        )
    except Exception as e:
        logger.error("admin_products_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve products")

@v1_router.get("/admin/me", response_model=APIResponse, status_code=status.HTTP_200_OK)
async def read_current_user(
    request: Request,
    current_user: dict = Depends(verify_jwt_token)
):
    """Endpoint to verify a token is valid and get basic user info."""
    EnhancedSecurityService.validate_admin_session(request, current_user)

    # Create the user data object that the frontend expects
    user_data = {"username": current_user.get("sub")}

    # Return the user data wrapped in the consistent APIResponse model
    return APIResponse(
        success=True,
        message="User authenticated successfully.",
        data={"user": user_data}
    )

@v1_router.get("/health", summary="Comprehensive Health Check for UI", status_code=status.HTTP_200_OK)
async def comprehensive_health_check(
    request: Request,
    _: dict = Depends(verify_jwt_token) # Use underscore if current_user is not needed
):
    """Provides detailed health status of all services for the admin dashboard."""
    # The original function logic goes here, but a simplified version is better
    # for a clear API response model.
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.api_version,
        "environment": settings.environment,
        "services": {}
    }

    # Database Check
    try:
        await services.db_service.db.command("ping")
        health_status["services"]["database"] = "connected"
    except Exception:
        health_status["services"]["database"] = "error"
        health_status["status"] = "degraded"

    # Cache Check
    try:
        await services.cache_service.redis.ping()
        health_status["services"]["cache"] = "connected"
    except Exception:
        health_status["services"]["cache"] = "error"
        health_status["status"] = "degraded"

    # AI Models Check
    health_status["services"]["ai_models"] = {
        "gemini": "available" if services.ai_service.gemini_client else "not_available",
        "openai": "available" if services.ai_service.openai_client else "not_available"
    }

    # External APIs (simple check)
    health_status["services"]["whatsapp"] = "configured" if settings.whatsapp_access_token else "not_configured"
    health_status["services"]["shopify"] = "configured" if settings.shopify_access_token else "not_configured"

    # Update overall status if any service has an error
    if any(val == "error" for val in health_status["services"].values()):
        health_status["status"] = "unhealthy"

    return APIResponse(
        success=True,
        message="Comprehensive health status retrieved.",
        data=health_status
    )

@v1_router.get("/health/detailed")
async def detailed_health():
    checks = {
        "database": await check_database_health(),
        "redis": await check_redis_health(),
        "whatsapp_api": await check_whatsapp_health(),
        "shopify_api": await check_shopify_health()
    }
    overall_status = "healthy" if all(checks.values()) else "unhealthy"
    return {"status": overall_status, "checks": checks}

async def check_database_health():
    try:
        await services.db_service.db.command("ping")
        return True
    except:
        return False

async def check_redis_health():
    try:
        await services.cache_service.redis.ping()
        return True
    except:
        return False

async def check_whatsapp_health():
    # Simple config check or ping if possible
    return bool(settings.whatsapp_access_token)

async def check_shopify_health():
    # Simple config check or ping if possible
    return bool(settings.shopify_access_token)

@v1_router.get("/admin/customers")
@limiter.limit("5/minute")
async def get_customers(
    request: Request,
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get paginated customer list"""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        # Validate pagination parameters
        page = max(1, page)
        limit = max(1, min(limit, 100))  # Cap at 100
        skip = (page - 1) * limit
        
        # Get customers with pagination
        cursor = services.db_service.db.customers.find(
            {},
            {
                "preferences": 0,
                "conversation_history": {"$slice": -1}  
            }
        ).sort("last_interaction", -1).skip(skip).limit(limit)
        
        customers = await cursor.to_list(length=limit)
        total_count = await services.db_service.db.customers.count_documents({})
        
        # Convert ObjectIds to strings
        for customer in customers:
            customer["_id"] = str(customer["_id"])
            # Mask phone numbers partially for privacy
            if "phone_number" in customer:
                phone = customer["phone_number"]
                if len(phone) > 8:
                    customer["phone_number"] = phone[:4] + "****" + phone[-4:]
        
        return APIResponse(
            success=True,
            message=f"Retrieved {len(customers)} customers",
            data={
                "customers": customers,
                "pagination": {
                    "page": page,
                    "limit": limit,
                    "total": total_count,
                    "pages": (total_count + limit - 1) // limit
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_customers_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve customers")

@v1_router.get("/admin/security/events")
@limiter.limit("5/minute")
async def get_security_events(
    request: Request,
    limit: int = 50,
    event_type: Optional[str] = None,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get recent security events"""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        query = {}
        if event_type:
            query["event_type"] = event_type
        
        cursor = services.db_service.db.security_events.find(query) \
            .sort("timestamp", -1) \
            .limit(min(limit, 100))
        
        events = await cursor.to_list(length=limit)
        
        # Sanitize sensitive data
        for event in events:
            event["_id"] = str(event["_id"])
            # Mask IP addresses partially
            if "ip_address" in event:
                ip = event["ip_address"]
                if "." in ip:  # IPv4
                    parts = ip.split(".")
                    event["ip_address"] = f"{parts[0]}.{parts[1]}.*.{parts[3]}"
        
        return APIResponse(
            success=True,
            message=f"Found {len(events)} security events",
            data={"events": events}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("get_security_events_error", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve security events")

@v1_router.post("/admin/broadcast")
@limiter.limit("1/minute")
async def broadcast_message(
    request: Request,
    broadcast_data: BroadcastRequest, # Use the new Pydantic model
    current_user: dict = Depends(verify_jwt_token)
):
    """Broadcast message to all, active, recent, or a specific list of customers."""
    try:
        EnhancedSecurityService.validate_admin_session(request, current_user)
        
        message = broadcast_data.message
        if not message or len(message.strip()) == 0:
            raise HTTPException(status_code=400, detail="Message cannot be empty")
        if len(message) > 1000:
            raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

        customers_to_message = []
        
        # New Logic: Prioritize the specific list of phones if provided
        if broadcast_data.target_phones:
            customers_to_message = await services.db_service.db.customers.find(
                {"phone_number": {"$in": broadcast_data.target_phones}}, 
                {"phone_number": 1}
            ).to_list(length=None)
        else:
            # Fallback to the original target_type logic
            now = datetime.utcnow()
            query = {}
            if broadcast_data.target_type == "active":
                query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
            elif broadcast_data.target_type == "recent":
                query["last_interaction"] = {"$gte": now - timedelta(days=7)}
            
            customers_to_message = await services.db_service.db.customers.find(
                query, 
                {"phone_number": 1}
            ).to_list(length=None)

        if not customers_to_message:
            return APIResponse(success=True, message="No customers found for the selected target", data={"sent_count": 0})

        sent_count, failed_count = 0, 0
        for customer in customers_to_message:
            success = await services.whatsapp_service.send_message(customer["phone_number"], message)
            if success:
                sent_count += 1
            else:
                failed_count += 1
            await asyncio.sleep(0.1)

        # Log the security event
        await services.db_service.log_security_event(
            "message_broadcast",
            get_remote_address(request),
            {
                "target": f"{len(broadcast_data.target_phones)} specific users" if broadcast_data.target_phones else broadcast_data.target_type,
                "sent_count": sent_count,
                "failed_count": failed_count
            }
        )
        
        return APIResponse(
            success=True,
            message=f"Broadcast completed: {sent_count} sent, {failed_count} failed",
            data={"sent_count": sent_count, "failed_count": failed_count}
        )
        
    except Exception as e:
        logger.error("broadcast_message_error", error=str(e))
        raise HTTPException(status_code=500, detail="Broadcast failed")

# ==================== PUBLIC ENDPOINTS ====================
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "Feelori AI WhatsApp Assistant",
        "version": "2.0.0",
        "status": "operational",
        "environment": settings.environment
    }

@app.get("/health")
async def health_check():
    """Basic health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "version": "2.0.0"
    }

@app.get("/health/ready")
async def readiness_check():
    """Kubernetes/Docker readiness probe"""
    try:
        # Quick checks that must pass for the app to be ready
        await services.db_service.db.command("ping")
        await services.cache_service.redis.ping()
        
        return {
            "status": "ready",
            "timestamp": datetime.utcnow(),
            "version": "2.0.0"
        }
    except Exception as e:
        logger.error("readiness_check_failed", error=str(e))
        raise HTTPException(status_code=503, detail="Service not ready")

@app.get("/health/live")
async def liveness_check():
    """Kubernetes/Docker liveness probe"""
    return {
        "status": "alive",
        "timestamp": datetime.utcnow(),
        "version": "2.0.0"
    }

@app.get("/metrics")
async def metrics(request: Request, _: bool = Depends(verify_metrics_access)):
    """Secured Prometheus metrics endpoint"""
    return PlainTextResponse(
        generate_latest(),
        media_type="text/plain"
    )

# Include the v1 router
app.include_router(v1_router)

# ==================== MAIN ENTRY POINT ====================
if __name__ == "__main__":
    import uvicorn

    # Run validation
    ProductionConfigValidator.validate_ssl_config()
    ProductionConfigValidator.validate_security_config()
    
    # Store start time for uptime calculation
    app.state.start_time = time.time()
    
    # Production configuration
    workers = int(os.getenv("UVICORN_WORKERS", "1"))  # Default to 1 for development
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    
    # SSL configuration
    ssl_keyfile = settings.ssl_key_path if settings.https_only else None
    ssl_certfile = settings.ssl_cert_path if settings.https_only else None
    
    if settings.environment == "production":
        if settings.https_only and ssl_keyfile and ssl_certfile:
            logger.info("starting_https_server", 
                       port=port, 
                       workers=workers,
                       ssl_enabled=True)
            uvicorn.run(
                "server:app",
                host=host,
                port=port,
                ssl_keyfile=ssl_keyfile,
                ssl_certfile=ssl_certfile,
                workers=workers if workers > 1 else None,  # Only use workers in production
                reload=False,
                access_log=False,  # Use structured logging instead
                log_config=None,   # Disable uvicorn's logging config
                server_header=False,
                date_header=False,
            )
        else:
            logger.info("starting_http_server", 
                       port=port, 
                       workers=workers,
                       ssl_enabled=False)
            uvicorn.run(
                "server:app",
                host=host,
                port=port,
                workers=workers if workers > 1 else None,
                reload=False,
                access_log=False,
                log_config=None,
                server_header=False,
                date_header=False,
            )
    else:
        # Development mode
        logger.info("starting_development_server", port=port)
        uvicorn.run(
            "server:app",
            host=host,
            port=port,
            reload=True,  # Enable reload in development
            access_log=True,
            log_level="info"
        )