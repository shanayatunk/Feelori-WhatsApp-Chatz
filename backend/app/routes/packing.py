# /app/routes/packing.py

import os
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import FileResponse
from app.models.api import APIResponse, HoldOrderRequest, FulfillOrderRequest
from app.utils.dependencies import verify_jwt_token
from app.services.db_service import db_service
from app.services import security_service
from app.services.whatsapp_service import whatsapp_service
from app.utils.rate_limiter import limiter

# This file defines all API routes specifically for the packing team's HTML dashboard.
# All routes are protected and require JWT authentication.

router = APIRouter(
    prefix="/packing",
    tags=["Packing Dashboard"],
    dependencies=[Depends(verify_jwt_token)]
)

# Serve the static HTML dashboard for the packing team
@router.get("/", include_in_schema=False)
async def get_packing_dashboard_page():
    basedir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    file_path = os.path.join(basedir, "static/dashboard.html")
    if os.path.exists(file_path):
        return FileResponse(file_path)
    raise HTTPException(status_code=404, detail="Dashboard not found")

# API endpoint to get all orders for the packing dashboard
@router.get("/orders", response_model=APIResponse)
async def get_packing_orders(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides the list of all orders for the packing dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    formatted_orders = await db_service.get_all_packing_orders()
    return APIResponse(success=True, message="Orders retrieved", data={"orders": formatted_orders}, version="v1")

# API endpoint to start packing an order
@router.post("/orders/{order_id}/start", response_model=APIResponse)
async def start_packing(order_id: str, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Marks an order as 'In Progress'."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    # --- FIX ---
    # The 'details' argument was missing. We provide an empty dictionary.
    # We also convert order_id to an integer.
    await db_service.update_order_packing_status(int(order_id), "In Progress", {})
    # --- END OF FIX ---
    
    return APIResponse(success=True, message="Order packing started.", version="v1")

# API endpoint to put an order on hold
@router.post("/orders/{order_id}/hold", response_model=APIResponse)
async def hold_order(order_id: str, hold_data: HoldOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Marks an order as 'On Hold' with a reason."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    await db_service.hold_order(
        order_id=int(order_id),
        reason=hold_data.reason,
        notes=hold_data.notes,
        skus=hold_data.problem_item_skus
    )
    return APIResponse(success=True, message="Order put on hold.", version="v1")

# API endpoint to fulfill an order
@router.post("/orders/{order_id}/fulfill", response_model=APIResponse)
async def fulfill_order(order_id: str, fulfill_data: FulfillOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """
    Marks an order as 'fulfilled' in Shopify, updates the local database, 
    and triggers customer notification.
    """
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)

    # --- THIS IS THE FIX ---
    # 1. Call the Shopify service to create the fulfillment in Shopify first.
    success, fulfillment_id = await shopify_service.fulfill_order(
        order_id=int(order_id),
        packer_name=fulfill_data.packer_name,
        tracking_number=fulfill_data.tracking_number,
        carrier=fulfill_data.carrier
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to create fulfillment in Shopify. Please check the order status in Shopify admin.")

    # 2. Update our internal database with the new status and the fulfillment ID from Shopify.
    await db_service.complete_order_fulfillment(
        order_id=int(order_id),
        packer_name=fulfill_data.packer_name,
        fulfillment_id=fulfillment_id
    )
    # --- END OF FIX ---

    # 3. Send notification to customer
    order_doc = await db_service.get_order_by_id(int(order_id))
    if order_doc and order_doc.get("phone_numbers"):
        customer_phone = order_doc["phone_numbers"][0]
        customer_name = order_doc.get("raw", {}).get("customer", {}).get("first_name", "there")
        
        notification_message = (
            f"Great news, {customer_name}! 🥳 Your FeelOri order #{order_doc.get('order_number')} has been packed and is on its way!\n\n"
            f"🚚 Tracking Number: {fulfill_data.tracking_number}\n"
            f"📦 Carrier: {fulfill_data.carrier}\n\n"
            "We're so excited for you to receive your items!"
        )
        asyncio.create_task(whatsapp_service.send_message(customer_phone, notification_message))

    return APIResponse(success=True, message="Order fulfilled in Shopify and customer notified.", version="v1")

# --- NEW FUNCTION ---
# API endpoint to move a held order back to the pending queue
@router.post("/orders/{order_id}/requeue", response_model=APIResponse)
async def requeue_order(order_id: str, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from 'On Hold' back to the 'Pending' queue."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    success = await db_service.requeue_held_order(order_id=int(order_id))
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Order not found or was not on hold."
        )
    return APIResponse(success=True, message="Order moved back to pending.", version="v1")
# --- END OF NEW FUNCTION ---

# API endpoint for packing metrics
@router.get("/metrics", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packing_metrics(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides key performance indicators (KPIs) for the packing workflow."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    metrics = await db_service.get_packing_dashboard_metrics()
    return APIResponse(
        success=True,
        message="Packing metrics retrieved successfully.",
        data=metrics
    )