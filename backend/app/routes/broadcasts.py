# /app/routes/broadcasts.py

import logging
from typing import List, Dict, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel

from app.config.settings import settings
from app.services.db_service import db_service
from app.services.broadcast_service import broadcast_service, ALLOWED_TEMPLATES
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse

logger = logging.getLogger(__name__)

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


@router.get("/config", response_model=APIResponse)
async def get_broadcast_config(current_user: dict = Depends(verify_jwt_token)):
    """Get broadcast configuration (templates and audiences). Admin only."""
    # Admin check
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    
    templates = list(ALLOWED_TEMPLATES)
    audiences = ["all", "active", "recent", "inactive", "custom", "custom_group"]
    
    return APIResponse(
        success=True,
        message="Broadcast configuration retrieved",
        data={
            "templates": templates,
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
        # Get users for broadcast
        users = await db_service.get_customers_for_broadcast(
            target_type=request_data.audience_type,
            target_phones=request_data.target_phones,
            target_group_id=request_data.target_group_id
        )
        
        # Extract phone numbers from user objects
        recipients = [user.get("phone_number") for user in users if user.get("phone_number")]
        
        if not recipients:
            raise HTTPException(status_code=400, detail="No recipients found for the specified audience")
        
        # Create broadcast job
        job_id = await db_service.create_broadcast_job(
            message=f"Template: {request_data.template_name}",
            image_url=request_data.params.get("header_image_url"),
            target_type=request_data.audience_type,
            total_recipients=len(recipients)
        )
        
        # Add background task to send broadcast
        background_tasks.add_task(
            broadcast_service.send_broadcast,
            target_business_id=request_data.business_id,
            template_name=request_data.template_name,
            recipients=recipients,
            variables=request_data.params,
            dry_run=request_data.dry_run
        )
        
        logger.info(f"Broadcast job {job_id} queued by admin {current_user.get('username', 'unknown')}")
        
        return APIResponse(
            success=True,
            message="Broadcast queued successfully",
            data={"job_id": job_id},
            version=settings.api_version
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
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
        
        # Convert ObjectId to string for JSON serialization
        for job in jobs:
            if "_id" in job:
                job["_id"] = str(job["_id"])
        
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

