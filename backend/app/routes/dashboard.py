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
    # 1. Normalize IDs
    tenant_candidates = {tenant_id, tenant_id.lower(), tenant_id.strip()}
    
    # 2. SUPER ADMIN HACK: If logged in as 'admin', assume ownership of all businesses
    # This fixes the issue where the JWT says 'admin' but DB has 'feelori'
    if tenant_id.lower() in ["admin", "superadmin", "administrator"]:
        tenant_candidates.add("feelori")
        tenant_candidates.add("goldencollections")
        # Add your 3rd business here later
    
    tenant_list = list(tenant_candidates)
    log.info(f"Dashboard Querying for candidates: {tenant_list}")

    # 3. Robust Query (unchanged logic, just using the expanded list)
    match_tenant = {
        "$or": [
            {"tenant_id": {"$in": tenant_list}},
            {"business_id": {"$in": tenant_list}}
        ]
    }

    # 4. Execute Counts
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
    
    attention_needed_count = await db_service.db.triage_tickets.count_documents({
        "$or": [
            {"business_id": {"$in": tenant_list}},
            {"tenant_id": {"$in": tenant_list}}
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

