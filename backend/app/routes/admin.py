# /app/routes/admin.py

import io
import csv
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Request
from typing import Optional, List

from app.config.settings import settings
from app.models.api import APIResponse, BroadcastRequest, Rule, StringResource, StringUpdateRequest
from app.utils.dependencies import verify_jwt_token, get_remote_address
from app.services import security_service, shopify_service, cache_service
from app.services.db_service import db_service
from app.services.order_service import get_or_create_customer # For broadcast
from app.services.whatsapp_service import whatsapp_service
from app.services.string_service import string_service
from app.services.rule_service import rule_service
from app.utils.rate_limiter import limiter
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(verify_jwt_token)]
)

@router.get("/stats", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_admin_stats(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get system statistics with an optimized aggregation pipeline."""
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
@limiter.limit("30/minute")
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


@router.get("/customers/{customer_id}", response_model=APIResponse)
@limiter.limit("20/minute")
async def get_customer_details(customer_id: str, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get detailed information for a single customer, including conversation history."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    customer = await db_service.get_customer_by_id(customer_id)
    if not customer:
        raise HTTPException(status_code=404, detail="Customer not found")
        
    return APIResponse(
        success=True,
        message="Customer details retrieved successfully.",
        data={"customer": customer},
        version=settings.api_version
    )


@router.get("/security-events", response_model=APIResponse)
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


async def run_broadcast_task(broadcast_id: str, message: str, image_url: str | None, customers: list):
    """The actual background task that sends messages."""
    for customer in customers:
        phone = customer.get("phone_number")
        if not phone:
            continue

        # In a real template-based broadcast, you'd format the message here
        # For now, we send the raw message
        wamid = await whatsapp_service.send_message(
            to_phone=phone,
            message=message,
            image_url=image_url
        )

        # Log the wamid with the broadcast_id
        if wamid:
            await db_service.db.message_logs.update_one(
                {"wamid": wamid},
                {"$set": {"metadata.broadcast_id": broadcast_id}}
            )
        
        await asyncio.sleep(0.2) # Sleep to avoid hitting API limits too quickly
    
    await db_service.db.broadcasts.update_one(
        {"_id": ObjectId(broadcast_id)},
        {"$set": {"status": "completed"}}
    )

@router.post("/broadcast", response_model=APIResponse)
@limiter.limit("1/minute")
async def broadcast_message(
    request: Request,
    broadcast_data: BroadcastRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_jwt_token)
):
    """Creates a broadcast job and runs it in the background."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    customers_to_message = await db_service.get_customers_for_broadcast(
        broadcast_data.target_type, broadcast_data.target_phones
    )

    if not customers_to_message:
        raise HTTPException(status_code=404, detail="No customers found for the selected target group.")

    # Create the broadcast job record in the database
    job_id = await db_service.create_broadcast_job(
        message=broadcast_data.message,
        image_url=broadcast_data.image_url,
        target_type=broadcast_data.target_type,
        total_recipients=len(customers_to_message)
    )

    # Add the sending process to the background
    background_tasks.add_task(
        run_broadcast_task, 
        job_id, 
        broadcast_data.message, 
        broadcast_data.image_url, 
        customers_to_message
    )

    return APIResponse(
        success=True,
        message=f"Broadcast job created and started for {len(customers_to_message)} customers.",
        data={"job_id": job_id},
        version=settings.api_version
    )


@router.get("/broadcasts", response_model=APIResponse)
async def get_broadcasts(
    request: Request,
    page: int = 1,
    limit: int = 20,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get a list of all past broadcast jobs."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    jobs, pagination = await db_service.get_broadcast_jobs(page, limit)
    return APIResponse(
        success=True,
        message="Broadcast jobs retrieved.",
        data={"broadcasts": jobs, "pagination": pagination},
        version=settings.api_version
    )


@router.get("/broadcasts/{job_id}", response_model=APIResponse)
async def get_broadcast_details(job_id: str, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get detailed stats for a specific broadcast job."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    details = await db_service.get_broadcast_job_details(job_id)
    if not details:
        raise HTTPException(status_code=404, detail="Broadcast job not found.")
    return APIResponse(
        success=True,
        message="Broadcast details retrieved.",
        data={"details": details},
        version=settings.api_version
    )

@router.get("/broadcasts/{job_id}/recipients", response_model=APIResponse)
async def get_broadcast_recipients(
    job_id: str,
    request: Request,
    page: int = 1,
    limit: int = 20,
    search: str = None,
    current_user: dict = Depends(verify_jwt_token)
):
    """Get a paginated list of recipients for a broadcast job."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    recipients, pagination = await db_service.get_broadcast_recipients(job_id, page, limit, search)
    return APIResponse(
        success=True,
        message="Recipients retrieved.",
        data={"recipients": recipients, "pagination": pagination},
        version=settings.api_version
    )


@router.get("/broadcasts/{job_id}/recipients/csv", response_model=None)
async def download_recipients_csv(job_id: str, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Download a CSV of all recipients for a broadcast job."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    recipients = await db_service.get_all_broadcast_recipients_for_csv(job_id)

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=["Name", "Phone Number", "Status", "Timestamp"])
    writer.writeheader()
    writer.writerows(recipients)
    
    headers = {
        'Content-Disposition': f'attachment; filename="broadcast_{job_id}_recipients.csv"'
    }
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)

@router.get("/health", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_system_health(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides a health check of all critical system components."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)

    # --- FIX: Use is_open() method with parentheses ---
    db_status = "connected" if not await db_service.circuit_breaker.is_open() else "error"

    # 2. Check Cache (Redis)
    try:
        await cache_service.redis.ping()
        cache_status = "connected"
    except Exception:
        cache_status = "error"

    # 3. Check WhatsApp Configuration
    whatsapp_status = "configured" if settings.whatsapp_access_token and settings.whatsapp_phone_id else "not_configured"

    # 4. Check Shopify Configuration
    shopify_status = "configured" if settings.shopify_access_token else "not_configured"

    services = {
        "database": db_status,
        "cache": cache_status,
        "whatsapp": whatsapp_status,
        "shopify": shopify_status,
    }

    overall_status = "healthy" if all(s in ["connected", "configured"] for s in services.values()) else "degraded"

    health_data = {
        "status": overall_status,
        "services": services
    }

    return APIResponse(
        success=True,
        message="System health retrieved successfully.",
        data=health_data,
        version=settings.api_version
    )

@router.get("/rules", response_model=APIResponse)
async def get_rules(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get all rules from the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    rules = await db_service.get_all_rules()
    return APIResponse(success=True, message="Rules retrieved successfully", data={"rules": rules}, version=settings.api_version)



@router.put("/rules/{rule_id}", response_model=APIResponse)
async def update_rule(rule_id: str, rule: Rule, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Update an existing rule in the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    updated_rule = await db_service.update_rule(rule_id, rule)
    if not updated_rule:
        raise HTTPException(status_code=404, detail="Rule not found")
    return APIResponse(success=True, message="Rule updated successfully", data={"rule": updated_rule}, version=settings.api_version)

@router.get("/strings", response_model=APIResponse)
async def get_strings(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get all string resources."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    strings = await db_service.get_all_strings()
    return APIResponse(success=True, message="Strings retrieved successfully", data={"strings": strings}, version=settings.api_version)

@router.put("/strings", response_model=APIResponse)
async def update_strings(update_data: StringUpdateRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Update all string resources."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    # Use the 'strings' attribute from the new model
    await db_service.update_strings(update_data.strings)
    await string_service.load_strings() # Reload the cache
    return APIResponse(success=True, message="Strings updated successfully", version=settings.api.version)

@router.get("/escalations", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_escalation_requests(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get recent human escalation requests."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    escalations = await db_service.get_human_escalation_requests()
    return APIResponse(
        success=True,
        message="Escalation requests retrieved successfully",
        data={"escalations": escalations},
        version=settings.api_version
    )

@router.post("/rules", response_model=APIResponse)
async def create_rule(rule: Rule, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Create a new rule in the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    new_rule = await db_service.create_rule(rule)
    await rule_service.load_rules()  # Reload rules after creating
    return APIResponse(success=True, message="Rule created successfully", data={"rule": new_rule}, version=settings.api_version)

@router.get("/packer-performance", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packer_performance(
    request: Request, 
    current_user: dict = Depends(verify_jwt_token),
    days: int = 7 # Default to a 7-day period
):
    """Provides advanced analytics for the packer performance dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    metrics = await db_service.get_packer_performance_metrics(days=days)
    return APIResponse(
        success=True,
        message="Packer performance metrics retrieved successfully.",
        data=metrics,
        version=settings.api_version # <-- THIS LINE IS THE FIX
    )
