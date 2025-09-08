# /app/routes/dashboard.py

from fastapi import APIRouter, Depends
# FIX 1: Import the settings to get the API version
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
    stats = await db_service.get_system_stats()
    # FIX 2: Add the required 'version' field to the response
    return APIResponse(success=True, message="Stats retrieved successfully.", data=stats, version=settings.api_version)

@router.get("/escalations", response_model=APIResponse)
async def get_human_escalations(current_user: dict = Depends(verify_jwt_token)):
    """Retrieves conversations that have been flagged for human review."""
    escalations = await db_service.get_human_escalation_requests()
    # FIX 3: Add the required 'version' field to the response
    return APIResponse(success=True, message="Escalations retrieved successfully.", data=escalations, version=settings.api_version)

@router.get("/packing-metrics", response_model=APIResponse)
async def get_react_packing_metrics(current_user: dict = Depends(verify_jwt_token)):
    """Provides packing metrics for the React admin dashboard's performance page."""
    metrics = await db_service.get_packing_dashboard_metrics()
    # FIX 4: Add the required 'version' field to the response
    return APIResponse(success=True, message="Packing metrics retrieved successfully.", data=metrics, version=settings.api_version)

