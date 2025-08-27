# /app/routes/admin.py

from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional

from app.config.settings import settings
from app.models.api import APIResponse, BroadcastRequest
from app.utils.dependencies import verify_jwt_token, get_remote_address
from app.services import security_service, shopify_service, db_service
from app.services.order_service import get_or_create_customer # For broadcast
from app.services.whatsapp_service import whatsapp_service
from app.utils.rate_limiter import limiter
import asyncio

# This file defines the API routes for administrative purposes, such as
# viewing statistics, customers, and broadcasting messages. All routes here
# are protected and require JWT authentication.

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(verify_jwt_token)]
)


@router.get("/stats", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_admin_stats(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get system statistics with an optimized aggregation pipeline."""
    # Implementation moved to db_service for better separation of concerns
    stats = await db_service.get_system_stats()
    return APIResponse(
        success=True,
        message="Statistics retrieved successfully",
        data=stats,
        version=settings.api_version
    )


@router.get("/products", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_admin_products(
    request: Request,
    query: Optional[str] = None,
    limit: int = 50,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get product list from Shopify for the admin dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    products, _ = await shopify_service.get_products(query or "", limit=limit)
    return APIResponse(
        success=True,
        message=f"Retrieved {len(products)} products",
        data={"products": [p.dict() for p in products]},
        version=settings.api_version
    )


@router.get("/customers", response_model=APIResponse)
@limiter.limit("5/minute")
async def get_customers(
    request: Request,
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get paginated customer list."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    customers, pagination_data = await db_service.get_paginated_customers(page, limit)
    return APIResponse(
        success=True,
        message=f"Retrieved {len(customers)} customers",
        data={"customers": customers, "pagination": pagination_data},
        version=settings.api_version
    )


@router.get("/security/events", response_model=APIResponse)
@limiter.limit("5/minute")
async def get_security_events(
    request: Request,
    limit: int = 50,
    event_type: Optional[str] = None,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get recent security events."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    events = await db_service.get_security_events(limit, event_type)
    return APIResponse(
        success=True,
        message=f"Found {len(events)} security events",
        data={"events": events},
        version=settings.api_version
    )


@router.post("/broadcast", response_model=APIResponse)
@limiter.limit("1/minute")
async def broadcast_message(
    request: Request,
    broadcast_data: BroadcastRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    """Broadcast message to all, active, recent, or a specific list of customers."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    message = broadcast_data.message
    if not message or len(message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    if len(message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long (max 1000 characters)")

    customers_to_message = await db_service.get_customers_for_broadcast(
        broadcast_data.target_type, broadcast_data.target_phones
    )

    if not customers_to_message:
        return APIResponse(success=True, message="No customers found for the selected target", data={"sent_count": 0}, version=settings.api_version)

    sent_count, failed_count = 0, 0
    for customer in customers_to_message:
        wamid = await whatsapp_service.send_message(customer["phone_number"], message)
        if wamid:
            sent_count += 1
        else:
            failed_count += 1
        await asyncio.sleep(0.1)

    await db_service.log_security_event(
        "message_broadcast",
        get_remote_address(request),
        {
            "target": f"{len(broadcast_data.target_phones)} specific users" if broadcast_data.target_phones else broadcast_data.target_type,
            "sent_count": sent_count,
            "failed_count": failed_count
        }
    )
    
    return APIResponse(
        success=True,
        message=f"Broadcast completed: {sent_count} sent, {failed_count} failed",
        data={"sent_count": sent_count, "failed_count": failed_count},
        version=settings.api_version
    )