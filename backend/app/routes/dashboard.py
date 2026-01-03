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
    # Extract tenant_id from JWT payload (conversations use tenant_id, not business_id)
    business_id = current_user.get("tenant_id", current_user.get("business_id", "feelori"))
    
    # 1. Active Conversations (using tenant_id field in conversations collection)
    active_count = await db_service.db.conversations.count_documents({
        "tenant_id": business_id,  # FIXED: Use tenant_id instead of business_id
        "status": {"$in": ["open", "pending", "human_needed"]}
    })

    # 2. AI Handled (using tenant_id field in conversations collection)
    ai_handled_count = await db_service.db.conversations.count_documents({
        "tenant_id": business_id,  # FIXED: Use tenant_id instead of business_id
        "ai_enabled": True
    })

    # 3. Human Intervention (using tenant_id field in conversations collection)
    human_needed_count = await db_service.db.conversations.count_documents({
        "tenant_id": business_id,  # FIXED: Use tenant_id instead of business_id
        "status": "human_needed"
    })

    # 4. Triage Tickets (using business_id field - this collection uses business_id, not tenant_id)
    triage_count = await db_service.db.triage_tickets.count_documents({
        "business_id": business_id,  # Keep business_id - triage_tickets collection uses this field
        "status": "human_needed"
    })

    frontend_stats = {
        "active_conversations": active_count,
        "ai_handled": ai_handled_count,
        "human_needed": human_needed_count,
        "triage_tickets": triage_count
    }
    
    return APIResponse(
        success=True, 
        message="Stats retrieved successfully.", 
        data={"stats": frontend_stats}, 
        version=settings.api_version
    )

# The /escalations and /triage-tickets endpoints have been removed.
# Their logic now resides permanently in /app/routes/admin.py and /app/routes/triage.py respectively.

