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
    """Marks an order as 'packing_in_progress'."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    await db_service.update_order_packing_status(order_id, "packing_in_progress")
    return APIResponse(success=True, message="Order packing started.", version="v1")

# API endpoint to put an order on hold
@router.post("/orders/{order_id}/hold", response_model=APIResponse)
async def hold_order(order_id: str, hold_data: HoldOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Marks an order as 'on_hold' with a reason."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    # --- THIS IS THE FIX ---
    # Call the correct db_service.hold_order function with the right parameters.
    # We also convert the order_id from a string to an integer here.
    await db_service.hold_order(
        order_id=int(order_id), 
        reason=hold_data.reason, 
        notes=hold_data.notes, 
        skus=hold_data.problem_item_skus
    )
    # --- END OF FIX ---
    
    return APIResponse(success=True, message="Order put on hold.", version="v1")

# API endpoint to fulfill an order
@router.post("/orders/{order_id}/fulfill", response_model=APIResponse)
async def fulfill_order(order_id: str, fulfill_data: FulfillOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Marks an order as 'fulfilled' and triggers customer notification."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    await db_service.update_order_packing_status(order_id, "fulfilled", fulfill_data.model_dump())
    
    # Send notification to customer
    order_doc = await db_service.orders_collection.find_one({"_id": order_id})
    if order_doc and order_doc.get("phone_numbers"):
        customer_phone = order_doc["phone_numbers"][0]
        customer_name = order_doc.get("raw", {}).get("customer", {}).get("first_name", "")
        
        notification_message = (
            f"Great news, {customer_name}! 🥳 Your FeelOri order #{order_doc.get('order_number')} has been packed and is on its way!\n\n"
            f"🚚 Tracking Number: {fulfill_data.tracking_number}\n"
            f"📦 Carrier: {fulfill_data.carrier}\n\n"
            "We're so excited for you to receive your items!"
        )
        asyncio.create_task(whatsapp_service.send_message(customer_phone, notification_message))

    return APIResponse(success=True, message="Order fulfilled and customer notified.", version="v1")

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