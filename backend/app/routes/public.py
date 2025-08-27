# /app/routes/public.py

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import PlainTextResponse
from prometheus_client import generate_latest
from datetime import datetime

from app.config.settings import settings
from app.utils.dependencies import verify_metrics_access
from app.services import db_service, cache_service
from app.models.api import APIResponse

# This file defines public-facing endpoints that do not require user authentication,
# such as health checks and the main root endpoint. The /metrics endpoint is
# conditionally protected by an API key.

router = APIRouter()

@router.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "Feelori AI WhatsApp Assistant",
        "version": "2.0.0",
        "status": "operational",
        "environment": settings.environment
    }

@router.get("/health", summary="Basic Health Check")
async def health_check():
    """Basic health check for load balancers."""
    return {"status": "healthy", "timestamp": datetime.utcnow()}

@router.get("/health/ready", summary="Readiness Probe")
async def readiness_check():
    """Kubernetes/Docker readiness probe checking essential services."""
    try:
        await db_service.db.command("ping")
        await cache_service.redis.ping()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {e}")

@router.get("/health/live", summary="Liveness Probe")
async def liveness_check():
    """Kubernetes/Docker liveness probe."""
    return {"status": "alive"}

@router.get("/health/detailed", response_model=APIResponse, tags=["Admin"])
async def comprehensive_health_check(request: Request):
    """Provides detailed health status of all services for the admin dashboard."""
    health_status = {"status": "healthy", "services": {}}
    
    try:
        await db_service.db.command("ping")
        health_status["services"]["database"] = "connected"
    except Exception:
        health_status["services"]["database"] = "error"
        health_status["status"] = "degraded"

    try:
        await cache_service.redis.ping()
        health_status["services"]["cache"] = "connected"
    except Exception:
        health_status["services"]["cache"] = "error"
        health_status["status"] = "degraded"

    health_status["services"]["whatsapp"] = "configured" if settings.whatsapp_access_token else "not_configured"
    health_status["services"]["shopify"] = "configured" if settings.shopify_access_token else "not_configured"

    return APIResponse(
        success=True,
        message="Comprehensive health status retrieved.",
        data=health_status,
        version=settings.api_version
    )


@router.get("/metrics", tags=["Monitoring"])
async def metrics(request: Request, _: bool = Depends(verify_metrics_access)):
    """Secured Prometheus metrics endpoint."""
    return PlainTextResponse(generate_latest(), media_type="text/plain")