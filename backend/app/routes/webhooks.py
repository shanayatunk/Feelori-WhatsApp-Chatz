# /app/routes/webhooks.py

import json
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request, Query
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config.settings import settings
from app.utils.dependencies import (
    verify_webhook_signature,
    verify_shopify_signature,
    get_remote_address
)
from app.services import security_service, shopify_service, db_service, order_service
from app.utils.queue import message_queue
from app.utils.metrics import response_time_histogram
from app.utils.rate_limiter import limiter

# This file defines all webhook endpoints that receive data from external
# services like WhatsApp and Shopify. These routes use special signature
# verification dependencies.

router = APIRouter(
    tags=["Webhooks"]
)


# --- WhatsApp Webhooks ---

@router.get("/webhook")
async def verify_whatsapp_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):
    """WhatsApp webhook verification (GET request)."""
    if hub_mode == "subscribe" and hub_verify_token == settings.whatsapp_verify_token:
        return PlainTextResponse(hub_challenge)
    raise HTTPException(status_code=403, detail="Forbidden")

@router.post("/webhook")
@limiter.limit(f"{settings.rate_limit_per_minute}/minute")
async def handle_whatsapp_webhook(
    request: Request,
    verified_body: bytes = Depends(verify_webhook_signature)
):
    """Enhanced webhook handler for both messages and status updates."""
    with response_time_histogram.labels(endpoint="whatsapp_webhook").time():
        data = json.loads(verified_body.decode())
        client_ip = get_remote_address(request)

        if not await security_service.rate_limiter.check_ip_rate_limit(client_ip, limit=100, window=60):
            return JSONResponse({"status": "rate_limited"}, status_code=429)

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                if change.get("field") != "messages": continue
                value = change.get("value", {})
                if "statuses" in value:
                    for status_data in value.get("statuses", []):
                        asyncio.create_task(order_service.handle_status_update(status_data))
                elif "messages" in value:
                    for message in value.get("messages", []):
                        asyncio.create_task(order_service.process_webhook_message(message, value))
        
        return JSONResponse({"status": "success"})


# --- Shopify Webhooks ---

@router.post("/webhooks/shopify/orders/create")
async def shopify_orders_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles new order creation from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_new_order_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/webhooks/shopify/orders/updated")
async def shopify_orders_updated_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles order updates from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_updated_order_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/webhooks/shopify/fulfillments/create")
async def shopify_fulfillments_create_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles new fulfillment events from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    asyncio.create_task(db_service.process_fulfillment_webhook(payload))
    return JSONResponse({"status": "ok"})


@router.post("/webhooks/shopify/checkouts/update")
async def shopify_checkouts_update_webhook(request: Request, verified_body: bytes = Depends(verify_shopify_signature)):
    """Handles abandoned checkout events from Shopify."""
    payload = json.loads(verified_body.decode("utf-8"))
    await order_service.handle_abandoned_checkout(payload)
    return JSONResponse({"status": "scheduled"})