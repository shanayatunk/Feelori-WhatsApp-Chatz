# /app/main.py

import os
import time
import uvicorn
import asyncio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from starlette.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from app.config.settings import settings
from app.utils.lifecycle import lifespan
from app.utils.metrics import response_time_histogram
from app.utils.rate_limiter import limiter
# --- FIX 1: Import the new 'packing' router ---
from app.routes import auth, admin, webhooks, public, dashboard, packing

# Initialize the FastAPI application 1
app = FastAPI(
    title="FeelOri AI WhatsApp Assistant",
    version="2.0.0",
    description="Refactored AI WhatsApp assistant with enterprise features",
    lifespan=lifespan,
    openapi_url=f"/api/{settings.api_version}/openapi.json" if settings.environment != "production" else None,
    docs_url=f"/api/{settings.api_version}/docs" if settings.environment != "production" else None,
    redoc_url=f"/api/{settings.api_version}/redoc" if settings.environment != "production" else None,
)

# --- Static Files ---
basedir = os.path.abspath(os.path.dirname(__file__))
static_dir_path = os.path.join(basedir, "static")
os.makedirs(static_dir_path, exist_ok=True)
app.mount("/static", StaticFiles(directory=static_dir_path), name="static")


# --- Rate Limiting ---
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- Middleware ---
cors_origins = []
if settings.cors_allowed_origins:
    cors_origins.extend([origin.strip() for origin in settings.cors_allowed_origins.split(",")])

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.environment != "test":
    allowed_hosts = [host.strip() for host in settings.allowed_hosts.split(",") if host.strip()]
    if allowed_hosts:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=allowed_hosts)

app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SlowAPIMiddleware)

@app.middleware("http")
async def performance_middleware(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response_time_histogram.labels(endpoint=request.url.path).observe(process_time)
    response.headers["X-Process-Time"] = str(process_time)
    return response

@app.middleware("http")
async def timeout_middleware(request: Request, call_next):
    try:
        return await asyncio.wait_for(call_next(request), timeout=30.0)
    except asyncio.TimeoutError:
        return JSONResponse({"detail": "Request timed out"}, status_code=504)

# --- API Routers ---
app.include_router(public.router)
app.include_router(auth.router, prefix=f"/api/{settings.api_version}")
app.include_router(admin.router, prefix=f"/api/{settings.api_version}")
app.include_router(webhooks.router, prefix=f"/api/{settings.api_version}/webhooks")
app.include_router(dashboard.router, prefix=f"/api/{settings.api_version}")
# --- FIX 2: Include the new 'packing' router ---
app.include_router(packing.router, prefix=f"/api/{settings.api_version}")

# --- Main Entry Point for Uvicorn (for local development) ---
if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "127.0.0.1")
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True if settings.environment == "development" else False,
        workers=settings.workers if settings.environment == "production" else 1
    )

