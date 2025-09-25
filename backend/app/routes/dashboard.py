# /app/routes/dashboard.py

from fastapi import APIRouter, Depends
from app.config.settings import settings
from app.services.db_service import db_service
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse
import structlog

log = structlog.get_logger(__name__)

# This file defines API routes specifically for the main React Admin Dashboard's summary stats.
router = APIRouter(
    prefix="/dashboard",
    tags=["Admin Dashboard"],
    dependencies=[Depends(verify_jwt_token)]
)

@router.get("/stats", response_model=APIResponse)
async def get_dashboard_stats(current_user: dict = Depends(verify_jwt_token)):
    """Provides key metrics for the main admin dashboard."""
    db_stats = await db_service.get_system_stats()

    frontend_stats = {
        "total_customers": db_stats.get("customers", {}).get("total", 0),
        "active_conversations": db_stats.get("customers", {}).get("active_24h", 0),
        "human_escalations": db_stats.get("escalations", {}).get("count", 0),
        "avg_response_time": db_stats.get("messages", {}).get("avg_response_time_minutes", "N/A")
    }
    
    return APIResponse(
        success=True, 
        message="Stats retrieved successfully.", 
        data={"stats": frontend_stats}, 
        version=settings.api_version
    )

# The /escalations and /triage-tickets endpoints have been removed.
# Their logic now resides permanently in /app/routes/admin.py and /app/routes/triage.py respectively.

