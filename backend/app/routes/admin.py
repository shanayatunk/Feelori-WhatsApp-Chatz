# /app/routes/admin.py

import io
import csv
import logging # Added logging import
from bson import ObjectId
from typing import Optional, List # Add List
from fastapi import APIRouter, Depends, HTTPException, Request, BackgroundTasks, status # Add status
from fastapi.responses import StreamingResponse
from app.config.settings import settings
# Import the new request/response models
from app.models.api import (
    APIResponse, BroadcastRequest, Rule, StringUpdateRequest,
    BroadcastGroupCreateRequest, BroadcastGroupResponse # Add new models
)
from app.utils.dependencies import verify_jwt_token
from app.services import security_service, shopify_service, cache_service
from app.services.db_service import db_service
from app.services.whatsapp_service import whatsapp_service
from app.services.string_service import string_service
from app.services.rule_service import rule_service
from app.utils.rate_limiter import limiter
import asyncio

# --- ADD logger ---
logger = logging.getLogger(__name__)
# --- END ADD ---

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

# --- Task for sending broadcasts (INCLUDES DEDUPLICATION) ---
async def run_broadcast_task(broadcast_id: str, message: str, image_url: str | None, customers: list):
    """The actual background task that sends messages, now using unique phones."""

    # --- ADD EXPLICIT DEDUPLICATION ---
    unique_phones_to_message = set()
    for customer in customers:
        phone = customer.get("phone_number") # Extract phone number from dict
        if phone:
            unique_phones_to_message.add(phone)
    # --- END OF DEDUPLICATION ---

    total_unique_recipients = len(unique_phones_to_message)
    logger.info(f"Starting broadcast job {broadcast_id} for {total_unique_recipients} unique recipients.")
    # Optional: Update the total recipients count if it differs significantly
    # await db_service.db.broadcasts.update_one(
    #     {"_id": ObjectId(broadcast_id)},
    #     {"$set": {"stats.total_recipients": total_unique_recipients}}
    # )

    # --- MODIFY LOOP TO USE THE SET ---
    for phone in unique_phones_to_message:
    # --- END OF MODIFICATION ---
        try: # Optional: Add try/except if you want the loop to continue if one send fails
            # TODO: Add name placeholder replacement here if needed using customer data fetched earlier
            # formatted_message = message.replace("{{name}}", customer_name_map.get(phone, "there"))
            wamid = await whatsapp_service.send_message(
                to_phone=phone, # Use the phone from the set
                message=message, # Use potentially formatted message
                image_url=image_url
            )
            if wamid:
                # Log the message send and associate with broadcast ID
                await db_service.db.message_logs.update_one(
                    {"wamid": wamid}, # Assuming wamid is unique identifier for sent message
                    {
                         "$set": {"metadata.broadcast_id": broadcast_id},
                         # Potentially log initial status if not already done in send_message
                         # "$setOnInsert": { ... base log data ... }
                    },
                    # upsert=True # Consider upsert if send_message doesn't log reliably
                )
            await asyncio.sleep(0.2) # Sleep to avoid hitting API limits too quickly
        except Exception as e:
            logger.error(f"Failed to send broadcast message to {phone} for job {broadcast_id}: {e}")
            # Optionally update the recipient status to 'failed' in message_logs if needed

    # Mark the overall job as completed
    await db_service.db.broadcasts.update_one(
        {"_id": ObjectId(broadcast_id)},
        {"$set": {"status": "completed"}}
    )
    logger.info(f"Completed broadcast job {broadcast_id}.")

# --- NEW Broadcast Group Endpoints ---

@router.post("/broadcast-groups", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
@limiter.limit("5/minute")
async def create_broadcast_group(
    request: Request,
    group_data: BroadcastGroupCreateRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    """Creates a new custom broadcast group."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)

    group_id = await db_service.create_broadcast_group(
        name=group_data.name,
        phone_numbers=group_data.phone_numbers
    )
    if not group_id:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Failed to create group. Check name uniqueness or phone numbers.")

    return APIResponse(
        success=True,
        message=f"Broadcast group '{group_data.name}' created successfully.",
        data={"group_id": group_id},
        version=settings.api_version
    )

@router.get("/broadcast-groups", response_model=APIResponse)
@limiter.limit("10/minute")
async def list_broadcast_groups(
    request: Request,
    current_user: dict = Depends(verify_jwt_token)
):
    """Lists all available custom broadcast groups."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    groups = await db_service.get_broadcast_groups()
    # Format response using the Pydantic model
    formatted_groups = [
        BroadcastGroupResponse(_id=g['_id'], name=g['name'], phone_count=g['phone_count']).dict(by_alias=True)
        for g in groups
    ]
    return APIResponse(
        success=True,
        message="Broadcast groups retrieved successfully.",
        data={"groups": formatted_groups}, # Send formatted list
        version=settings.api_version
    )

# --- END NEW Broadcast Group Endpoints ---


# --- Modified Broadcast Endpoint ---
@router.post("/broadcast", response_model=APIResponse)
@limiter.limit("1/minute")
async def broadcast_message(
    request: Request,
    broadcast_data: BroadcastRequest, # Uses the updated model
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(verify_jwt_token)
):
    """Creates a broadcast job and runs it in the background."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)

    # Pass target_group_id to the db_service function
    customers_to_message = await db_service.get_customers_for_broadcast(
        broadcast_data.target_type,
        broadcast_data.target_phones,
        broadcast_data.target_group_id # Pass the group ID if provided
    )

    if not customers_to_message:
        raise HTTPException(status_code=400, detail="No customers found for the selected target group or criteria.") # Changed to 400

    # Determine target description for logging/display
    target_description = broadcast_data.target_type
    if broadcast_data.target_type == "custom_group" and broadcast_data.target_group_id:
         # Optionally fetch group name for better logging/display
        try:
             group_info = await db_service.db.broadcast_groups.find_one(
                 {"_id": ObjectId(broadcast_data.target_group_id)},
                 {"name": 1}
             )
             group_name = group_info.get("name") if group_info else broadcast_data.target_group_id
             target_description = f"custom group '{group_name}'"
        except Exception: # Catch potential InvalidId errors
             logger.warning(f"Could not fetch group name for invalid ID: {broadcast_data.target_group_id}")
             target_description = f"custom group ID {broadcast_data.target_group_id}"
    elif broadcast_data.target_phones:
         target_description = "custom phone list"

    total_recipients = len(customers_to_message)

    # Create the broadcast job record in the database
    job_id = await db_service.create_broadcast_job(
        message=broadcast_data.message,
        image_url=broadcast_data.image_url,
        # Log the specific target for clarity
        target_type=target_description,
        total_recipients=total_recipients # Log the initial count before deduplication
    )

    # Add the sending process to the background
    background_tasks.add_task(
        run_broadcast_task,
        job_id,
        broadcast_data.message,
        broadcast_data.image_url,
        customers_to_message # Pass the fetched customer list
    )

    return APIResponse(
        success=True,
        # Report the number of unique recipients the task will process
        message=f"Broadcast job created (ID: {job_id}) and started for {len(set(c.get('phone_number') for c in customers_to_message if c.get('phone_number')))} unique recipients.",
        data={"job_id": job_id},
        version=settings.api_version
    )

# --- Broadcast History & Reporting Endpoints ---

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

    if not recipients:
         raise HTTPException(status_code=404, detail="No recipients found for this broadcast job.")

    output = io.StringIO()
    # Use the keys from the first recipient dict as headers, ensure order if necessary
    fieldnames = list(recipients[0].keys())
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(recipients)

    headers = {
        'Content-Disposition': f'attachment; filename="broadcast_{job_id}_recipients.csv"'
    }
    return StreamingResponse(iter([output.getvalue()]), media_type="text/csv", headers=headers)

# --- System Health ---

@router.get("/health", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_system_health(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides a health check of all critical system components."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)

    # 1. Check Database (MongoDB)
    db_status = "connected" if await db_service.health_check() else "error"

    # 2. Check Cache (Redis)
    cache_status = "error"
    if cache_service.redis:
        try:
            await cache_service.redis.ping()
            cache_status = "connected"
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            cache_status = "error"
    else:
        cache_status = "not_configured" # Or "error" if it should always be present


    # 3. Check WhatsApp Configuration
    whatsapp_status = "configured" if settings.whatsapp_access_token and settings.whatsapp_phone_id else "not_configured"

    # 4. Check Shopify Configuration
    shopify_status = "configured" if settings.shopify_access_token else "not_configured"

    # 5. Check AI Service Configuration (Optional: Check API keys)
    ai_status = "configured"
    if not settings.gemini_api_key and not settings.openai_api_key:
        ai_status = "not_configured"
    # You could add a ping here if the AI SDKs support it

    services = {
        "database": db_status,
        "cache": cache_status,
        "whatsapp": whatsapp_status,
        "shopify": shopify_status,
        "ai_service": ai_status, # Added AI service status
    }

    # Determine overall status based on critical services (e.g., DB, Cache, WhatsApp)
    critical_services = ["database", "cache", "whatsapp"]
    is_degraded = any(services[s] not in ["connected", "configured"] for s in critical_services)
    overall_status = "degraded" if is_degraded else "healthy"

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

# --- Rules Engine Endpoints ---

@router.get("/rules", response_model=APIResponse)
async def get_rules(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get all rules from the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    rules = await db_service.get_all_rules() # Fetches from DB
    # or use rule_service.get_rules() # Fetches from cache if loaded
    return APIResponse(success=True, message="Rules retrieved successfully", data={"rules": rules}, version=settings.api_version)

@router.post("/rules", response_model=APIResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(rule: Rule, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Create a new rule in the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    new_rule = await db_service.create_rule(rule)
    if not new_rule:
         raise HTTPException(status_code=400, detail="Failed to create rule, name might already exist.")
    await rule_service.load_rules()  # Reload rules cache after creating
    return APIResponse(success=True, message="Rule created successfully", data={"rule": new_rule}, version=settings.api_version)

@router.put("/rules/{rule_id}", response_model=APIResponse)
async def update_rule(rule_id: str, rule: Rule, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Update an existing rule in the rules engine."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    # Ensure the name in the path matches the body if needed, or just use rule_id
    updated_rule = await db_service.update_rule(rule_id, rule)
    if not updated_rule:
        raise HTTPException(status_code=404, detail="Rule not found or update failed")
    await rule_service.load_rules() # Reload rules cache after updating
    return APIResponse(success=True, message="Rule updated successfully", data={"rule": updated_rule}, version=settings.api_version)

# --- Strings Manager Endpoints ---

@router.get("/strings", response_model=APIResponse)
async def get_strings(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Get all string resources."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    strings_list = await db_service.get_all_strings() # Fetches from DB
    # or use string_service._strings_cache # Access cache directly if needed, though service method is safer
    return APIResponse(success=True, message="Strings retrieved successfully", data={"strings": strings_list}, version=settings.api_version)

@router.put("/strings", response_model=APIResponse)
async def update_strings(update_data: StringUpdateRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Update all string resources."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    await db_service.update_strings(update_data.strings)
    await string_service.load_strings() # Reload the string cache
    return APIResponse(success=True, message="Strings updated successfully", version=settings.api_version)

# --- Escalation & Performance Endpoints ---

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
        message=f"Packer performance metrics for the last {days} days retrieved.",
        data=metrics,
        version=settings.api_version # Corrected this line
    )