# /app/routes/dashboard.py

from fastapi import APIRouter, Depends
from app.services.db_service import db_service
# --- THIS IS THE CORRECTED IMPORT ---
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse

# This file defines all API routes specifically for the new React Admin Dashboard.
router = APIRouter(
    prefix="/dashboard",
    tags=["Admin Dashboard"],
    # --- THIS DEPENDENCY IS NOW CORRECT ---
    dependencies=[Depends(verify_jwt_token)]
)

@router.get("/stats", response_model=APIResponse)
async def get_dashboard_stats(current_user: dict = Depends(verify_jwt_token)):
    """Provides key metrics for the main admin dashboard."""
    stats = await db_service.get_dashboard_stats()
    return APIResponse(success=True, message="Stats retrieved successfully.", data=stats)

@router.get("/escalations", response_model=APIResponse)
async def get_human_escalations(current_user: dict = Depends(verify_jwt_token)):
    """Retrieves conversations that have been flagged for human review."""
    escalations = await db_service.get_human_escalations()
    return APIResponse(success=True, message="Escalations retrieved successfully.", data=escalations)

@router.get("/packing-metrics", response_model=APIResponse)
async def get_react_packing_metrics(current_user: dict = Depends(verify_jwt_token)):
    """Provides packing metrics for the React admin dashboard's performance page."""
    metrics = await db_service.get_packing_metrics()
    return APIResponse(success=True, message="Packing metrics retrieved successfully.", data=metrics)

