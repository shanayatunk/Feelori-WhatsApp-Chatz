# /app/routes/dashboard.py

from fastapi import APIRouter, Depends, HTTPException
from app.config.settings import settings
from app.services.db_service import db_service
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse
import structlog  # <-- ADD THIS IMPORT

# Get a logger
log = structlog.get_logger(__name__)

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


# --- ADD THIS NEW ENDPOINT ---

@router.get("/triage-tickets", response_model=APIResponse)
async def get_triage_tickets(current_user: dict = Depends(verify_jwt_token)):
    """
    Fetches all open triage tickets for the admin panel.
    """
    try:
        tickets_cursor = db_service.db.triage_tickets.find(
            {"status": "pending"}
        ).sort("created_at", 1)  # Show oldest first
        
        tickets = await tickets_cursor.to_list(length=100)
        
        # Convert MongoDB ObjectId to string for JSON serialization
        for ticket in tickets:
            ticket["_id"] = str(ticket["_id"])
            
        return APIResponse(
            success=True,
            message="Triage tickets retrieved successfully.",
            data={"tickets": tickets},
            version=settings.api_version
        )
        
    except Exception as e:
        log.error("Error fetching triage tickets", exc_info=True, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to fetch triage tickets")