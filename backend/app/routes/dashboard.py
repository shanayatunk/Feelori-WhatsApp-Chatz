# /app/routes/dashboard.py

from fastapi import APIRouter, Depends
from app.config.settings import settings
from app.services.db_service import db_service
from app.dependencies.tenant import get_tenant_id
from app.models.api import APIResponse
import structlog

log = structlog.get_logger(__name__)

# This file defines API routes specifically for the main React Admin Dashboard's summary stats.
router = APIRouter(
    prefix="/dashboard",
    tags=["Admin Dashboard"]
)

@router.get("/stats", response_model=APIResponse)
async def get_dashboard_stats(
    tenant_id: str = Depends(get_tenant_id)
):
    """Provides key metrics for the main admin dashboard."""
    # 1. Normalize IDs: Check for "FeelOri", "feelori", etc.
    tenant_candidates = list(set([tenant_id, tenant_id.lower(), tenant_id.strip()]))
    
    # 2. Robust Query: Check BOTH 'tenant_id' AND 'business_id' fields
    match_tenant = {
        "$or": [
            {"tenant_id": {"$in": tenant_candidates}},
            {"business_id": {"$in": tenant_candidates}}
        ]
    }

    # 3. Execute Counts
    active_count = await db_service.db.conversations.count_documents({
        **match_tenant,
        "status": {"$in": ["open", "pending", "human_needed"]}
    })

    ai_handled_count = await db_service.db.conversations.count_documents({
        **match_tenant,
        "ai_enabled": True
    })

    human_needed_count = await db_service.db.conversations.count_documents({
        **match_tenant,
        "status": "human_needed"
    })
    
    # Check triage tickets (robust check)
    attention_needed_count = await db_service.db.triage_tickets.count_documents({
        "$or": [
            {"business_id": {"$in": tenant_candidates}},
            {"tenant_id": {"$in": tenant_candidates}}
        ],
        "status": "human_needed"
    })

    return APIResponse(
        success=True,
        data={
            "active_conversations": active_count,
            "ai_handled": ai_handled_count,
            "human_intervention": human_needed_count,
            "attention_needed": attention_needed_count
        },
        version=settings.api_version
    )

# The /escalations and /triage-tickets endpoints have been removed.
# Their logic now resides permanently in /app/routes/admin.py and /app/routes/triage.py respectively.

