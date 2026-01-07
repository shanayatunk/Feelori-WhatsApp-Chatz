# /app/routes/broadcasts.py

import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.config.settings import settings
from app.services.db_service import db_service
from app.services.broadcast_service import broadcast_service, ALLOWED_TEMPLATES
from app.utils.dependencies import verify_jwt_token
from app.dependencies.tenant import get_tenant_id
from app.models.api import APIResponse, BroadcastGroupCreate

logger = logging.getLogger(__name__)

# Template Metadata for UI Previews (Phase 5C)
# Source of Truth: Meta Business Manager
# Last Synced: Manually by Admin
TEMPLATE_METADATA = {
    "new_arrival_showcase": {
        "header": "IMAGE",
        "body": "✨ Hello {{1}}!\n\nOur latest collection is finally here. Handcrafted designs to make you Feel Original. 💎\n\nCheck them out before they are gone! 👇",
        "version": "v1.0",
        "last_verified": "2025-12-29"
    },
    "video_collection_launch": {
        "header": "VIDEO",
        "body": "🎥 Watch our new designs come to life, {{1}}!\n\nDetailed craftsmanship you have to see to believe. Perfect for the upcoming season. ✨\n\nTap below to shop the look!",
        "version": "v1.0",
        "last_verified": "2025-12-29"
    },
    "festival_sale_alert": {
        "header": "TEXT: FESTIVE SALE 🪔",
        "body": "Hi {{1}}, the festive season is here!\n\nEnjoy flat {{2}}% OFF on selected items. 💖\n\nUse code: *{{3}}* at checkout.\nOffer valid until {{4}}.",
        "version": "v1.0",
        "last_verified": "2025-12-29"
    },
    "gentle_greeting_v1": {
        "header": "NONE",
        "body": "Hi {{1}}, we missed you! 👋\n\nWe've added some beautiful new pieces since your last visit. Come take a look at what's new at FeelOri. ✨",
        "version": "v1.0",
        "last_verified": "2025-12-29"
    }
}

router = APIRouter(
    prefix="/broadcasts",
    tags=["Broadcasts"]
)


class BroadcastSendRequest(BaseModel):
    """Request model for sending a broadcast."""
    template_name: str
    audience_type: str  # "all", "active", "recent", "inactive", "custom", "custom_group"
    business_id: str = "feelori"
    params: Dict = {}  # Template variables: body_params, header_text_param, header_image_url, button_url_param
    target_phones: Optional[List[str]] = None  # For "custom" audience type
    target_group_id: Optional[str] = None  # For "custom_group" audience type
    dry_run: bool = False
    is_test: bool = False  # Test mode - doesn't create broadcast job
    test_phone: Optional[str] = None  # Phone number for test mode


@router.get("/config", response_model=APIResponse)
async def get_broadcast_config(current_user: dict = Depends(verify_jwt_token)):
    """Get broadcast configuration (templates and audiences with counts). Admin only."""
    # Admin check
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    # Format templates with metadata for UI previews
    formatted_templates = []
    for t in ALLOWED_TEMPLATES:
        # Convert snake_case to Title Case for readable name
        readable_name = t.replace("_", " ").title()
        meta = TEMPLATE_METADATA.get(t, {})
        formatted_templates.append({
            "id": t,
            "name": readable_name,
            "header": meta.get("header", "NONE"),
            "body": meta.get("body", "Preview not available."),
            "version": meta.get("version", "v0"),
            "last_verified": meta.get("last_verified", "")
        })
    
    # Audience definitions with display names
    audience_definitions = [
        {"id": "all", "name": "All Customers"},
        {"id": "active", "name": "Active (24h)"},
        {"id": "recent", "name": "Recent (7d)"},
        {"id": "inactive", "name": "Inactive (30d+)"},
        {"id": "custom", "name": "Custom List"},
        {"id": "custom_group", "name": "Custom Group"}
    ]
    
    # Get counts for each audience type
    audiences = []
    for audience in audience_definitions:
        try:
            count = await db_service.count_audience(audience["id"])
            audiences.append({
                "id": audience["id"],
                "name": audience["name"],
                "count": count
            })
        except Exception as e:
            logger.error(f"Failed to count audience {audience['id']}: {e}")
            audiences.append({
                "id": audience["id"],
                "name": audience["name"],
                "count": 0
            })
    
    return APIResponse(
        success=True,
        message="Broadcast configuration retrieved",
        data={
            "templates": formatted_templates,
            "audiences": audiences
        },
        version=settings.api_version
    )


@router.post("/send", response_model=APIResponse)
async def send_broadcast(
    request_data: BroadcastSendRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_jwt_token)
):
    """Send a broadcast to an audience. Admin only."""
    # Admin check
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        # Test Mode Logic (Keep as is)
        if request_data.is_test:
            if not request_data.test_phone:
                raise HTTPException(status_code=400, detail="test_phone is required when is_test is True")
            
            result = await broadcast_service.send_broadcast(
                target_business_id=request_data.business_id,
                template_name=request_data.template_name,
                recipients=[request_data.test_phone],
                variables=request_data.params,
                dry_run=request_data.dry_run
            )
            return APIResponse(
                success=True,
                message="Test broadcast sent successfully",
                data={"mode": "test", "sent_count": result.get("sent_count", 0)},
                version=settings.api_version
            )
        
        # --- DEBUG & FIX SECTION ---
        target_group_id = request_data.target_group_id
        audience_type = request_data.audience_type

        # Log incoming request
        logger.info(f"Broadcast Request - Raw Audience: '{audience_type}', GroupID: '{target_group_id}'")

        if audience_type and audience_type.startswith("group:"):
            parts = audience_type.split(":")
            if len(parts) == 2:
                audience_type = "custom_group"
                target_group_id = parts[1].strip() # Strip whitespace just in case
                logger.info(f"Parsed Custom Group - Type: {audience_type}, ID: {target_group_id}")
        
        # Get users
        users = await db_service.get_customers_for_broadcast(
            target_type=audience_type,
            target_phones=request_data.target_phones,
            target_group_id=target_group_id
        )
        
        logger.info(f"DB Query returned {len(users)} users for audience '{audience_type}'")

        # Extract recipients
        recipients = [user.get("phone_number") for user in users if user.get("phone_number") and not user.get("opted_out")]
        
        if not recipients:
            logger.error(f"Recipient list is empty after filtering. Raw users found: {len(users)}")
            raise HTTPException(status_code=400, detail="No recipients found for the specified audience")
        
        # Create broadcast job
        job_id = await db_service.create_broadcast_job(
            message=f"Template: {request_data.template_name}",
            image_url=request_data.params.get("header_image_url"),
            target_type=audience_type,
            total_recipients=len(recipients)
        )
        
        background_tasks.add_task(
            broadcast_service.execute_job,
            job_id=job_id,
            target_business_id=request_data.business_id,
            template_name=request_data.template_name,
            recipients=recipients,
            variables=request_data.params,
            dry_run=request_data.dry_run
        )
        
        return APIResponse(
            success=True,
            message="Broadcast queued successfully",
            data={"job_id": job_id},
            version=settings.api_version
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise 
    except Exception as e:
        logger.error(f"Failed to queue broadcast: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to queue broadcast")


@router.get("/history", response_model=APIResponse)
async def get_broadcast_history(
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get broadcast history. Admin only."""
    # Admin check
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    try:
        jobs, pagination = await db_service.get_broadcast_jobs(page=page, limit=limit)
        
        # Convert ObjectId to string for JSON serialization and ensure stats exist
        for job in jobs:
            if "_id" in job:
                job["_id"] = str(job["_id"])
            
            # Ensure stats are integers, defaulting to 0 if missing
            job["sent_count"] = job.get("sent_count", 0)
            job["delivered_count"] = job.get("delivered_count", 0)
            job["read_count"] = job.get("read_count", 0)
            job["failed_count"] = job.get("failed_count", 0)
            job["total_recipients"] = job.get("total_recipients", 0)
            
            # Ensure all values are integers (handle None/null values)
            job["sent_count"] = int(job["sent_count"]) if job["sent_count"] is not None else 0
            job["delivered_count"] = int(job["delivered_count"]) if job["delivered_count"] is not None else 0
            job["read_count"] = int(job["read_count"]) if job["read_count"] is not None else 0
            job["failed_count"] = int(job["failed_count"]) if job["failed_count"] is not None else 0
            job["total_recipients"] = int(job["total_recipients"]) if job["total_recipients"] is not None else 0
        
        return APIResponse(
            success=True,
            message="Broadcast history retrieved",
            data={
                "jobs": jobs,
                "pagination": pagination
            },
            version=settings.api_version
        )
    except Exception as e:
        logger.error(f"Failed to get broadcast history: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve broadcast history")


@router.get("/groups", response_model=APIResponse)
async def get_broadcast_groups(
    tenant_id: str = Depends(get_tenant_id)
):
    """Fetch all custom audience groups."""
    groups = await db_service.get_broadcast_groups()
    return APIResponse(
        success=True,
        message="Groups retrieved",
        # FIX: Wrap the list in a dictionary to satisfy Pydantic
        data={"groups": groups}, 
        version=settings.api_version
    )


@router.post("/groups", response_model=APIResponse)
async def create_broadcast_group(
    payload: BroadcastGroupCreate,
    tenant_id: str = Depends(get_tenant_id)
):
    """Create a new static audience group."""
    group = await db_service.create_broadcast_group(payload)
    if not group:
        raise HTTPException(status_code=500, detail="Failed to create group")
    
    return APIResponse(
        success=True,
        message="Group created successfully",
        data=group,
        version=settings.api_version
    )

