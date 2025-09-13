# /app/routes/dashboard.py

from fastapi import APIRouter, Depends
from app.config.settings import settings
from app.services.db_service import db_service
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse

# This file defines all API routes specifically for the new React Admin Dashboard.
router = APIRouter(
    prefix="/dashboard",
    tags=["Admin Dashboard"],
    dependencies=[Depends(verify_jwt_token)]
)

@router.get("/stats", response_model=APIResponse)
async def get_dashboard_stats(current_user: dict = Depends(verify_jwt_token)):
    """Provides key metrics for the main admin dashboard."""
    # 1. Get the nested data from the database service
    db_stats = await db_service.get_system_stats()

    # 2. Transform the data into the flat structure the frontend expects
    #    Note: The DB doesn't calculate escalations or avg response time yet, so we default them.
    frontend_stats = {
        "total_customers": db_stats.get("customers", {}).get("total", 0),
        "active_conversations": db_stats.get("customers", {}).get("active_24h", 0),
        "human_escalations": db_stats.get("escalations", {}).get("count", 0),
        "avg_response_time": db_stats.get("messages", {}).get("avg_response_time_minutes", "N/A")
    }
    
    # 3. Return the correctly formatted data
    return APIResponse(
        success=True, 
        message="Stats retrieved successfully.", 
        data={"stats": frontend_stats}, 
        version=settings.api_version
    )

@router.get("/escalations", response_model=APIResponse)
async def get_human_escalations(current_user: dict = Depends(verify_jwt_token)):
    """Retrieves conversations that have been flagged for human review."""
    escalations = await db_service.get_human_escalation_requests()
    # FIX: Wrap the returned data in a dictionary
    return APIResponse(success=True, message="Escalations retrieved successfully.", data={"escalations": escalations}, version=settings.api_version)



