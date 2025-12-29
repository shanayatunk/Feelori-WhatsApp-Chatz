# /app/routes/webhooks.py

import json
import asyncio
import structlog
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config.settings import settings
from app.utils.dependencies import (
    verify_webhook_signature,
    verify_shopify_signature,
    get_remote_address
)
from app.services import security_service, order_service
from app.services.db_service import db_service
from app.utils.metrics import response_time_histogram
from app.utils.rate_limiter import limiter

# This file defines all webhook endpoints that receive data from external
# services like WhatsApp and Shopify. These routes use special signature
# verification dependencies.

router = APIRouter(
    tags=["Webhooks"]
)

log = structlog.get_logger(__name__)

# --- WhatsApp Webhooks ---

@router.get("/whatsapp")
async def verify_whatsapp_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """WhatsApp webhook verification (GET request)."""
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        log.info("WhatsApp webhook verification successful.")
        return PlainTextResponse(hub_challenge)
    log.error("WhatsApp webhook verification failed.")
    raise HTTPException(status_code=403, detail="Forbidden")

@router.post("/whatsapp")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def handle_whatsapp_webhook(
    request: Request,
    verified_body: bytes = Depends(verify_webhook_signature)
):
    """Enhanced webhook handler for both messages and status updates."""
    with response_time_histogram.labels(endpoint="whatsapp_webhook").time():
        log.info("WhatsApp webhook received a request.")
        data = json.loads(verified_body.decode())
        log.debug("Webhook payload", data=data)

        client_ip = get_remote_address(request)

        if not await security_service.rate_limiter.check_ip_rate_limit(client_ip, limit=100, window=60):
            log.warning("Rate limit exceeded for IP", client_ip=client_ip)
            return JSONResponse({"status": "rate_limited"}, status_code=429)

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages": 
                    log.debug("Ignoring non-message change", change=change)
                    continue
                
                value = change.get("value", {})

                # --- [START] NEW PHONE ID FILTER ---
                # Get the Phone ID from the incoming webhook payload
                metadata = value.get("metadata", {})
                incoming_phone_id = metadata.get("phone_number_id")
                
                # Get the Phone ID this server *expects* (from Doppler/env)
                # This 'settings' object is already imported at the top of the file
                expected_phone_id = settings.whatsapp_phone_id 

                # If they don't match, log it and skip this change completely.
                if incoming_phone_id and expected_phone_id and incoming_phone_id != expected_phone_id:
                    log.info(
                        "Ignored event for different phone ID.",
                        incoming_id=incoming_phone_id,
                        expected_id=expected_phone_id
                    )
                    continue  # <-- This skips to the next 'change'
                # --- [END] NEW PHONE ID FILTER ---

                # --- (Your existing logic continues below) ---
                if "statuses" in value:
                    for status_data in value.get("statuses", []):
                        wamid = status_data.get("id")
                        status_type = status_data.get("status")
                        
                        if wamid and status_type:
                            log.info("Processing status update", wamid=wamid, status=status_type)
                            # The DB service now handles idempotency and job linking internally
                            await db_service.update_message_status(wamid, status_type)
                elif "messages" in value:
                    for message in value.get("messages", []):
                        log.info("Processing incoming message", message=message)
                        asyncio.create_task(order_service.process_webhook_message(message, value))
        
        log.info("Webhook processing complete.")
        return JSONResponse({"status": "success"})


# --- Shopify Webhooks ---

@router.post("/shopify/orders/create")
async def shopify_orders_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles new order creation from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_new_order_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/shopify/orders/updated")
async def shopify_orders_updated_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles order updates from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_updated_order_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/shopify/fulfillments/create")
async def shopify_fulfillments_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles new fulfillment events from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_fulfillment_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/shopify/checkouts/update")
async def shopify_checkouts_update_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles abandoned checkout events from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(order_service.handle_abandoned_checkout(payload))
    return JSONResponse({"status": "scheduled"})