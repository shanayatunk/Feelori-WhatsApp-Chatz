# /app/routes/dashboard.py

from fastapi import APIRouter, Depends, HTTPException, Request
from app.models.api import APIResponse, HoldOrderRequest, FulfillOrderRequest
from app.utils.dependencies import verify_jwt_token
# --- Fix Start ---
# Import the db_service INSTANCE from its module, not the module itself
from app.services.db_service import db_service
from app.services import security_service, shopify_service
# --- Fix End ---
from app.services.whatsapp_service import whatsapp_service
from app.utils.rate_limiter import limiter
import asyncio

# This file defines all API routes specifically for the packing dashboard frontend.
# All routes are protected and require JWT authentication.

router = APIRouter(
    prefix="/packing",
    tags=["Packing Dashboard"],
    dependencies=[Depends(verify_jwt_token)]
)


@router.get("/orders", response_model=APIResponse)
async def get_packing_orders(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides the list of all orders for the packing dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    formatted_orders = await db_service.get_all_packing_orders()
    return APIResponse(success=True, message="Orders retrieved", data={"orders": formatted_orders}, version="v1")


@router.post("/orders/{order_id}/start", response_model=APIResponse)
async def start_packing_order(order_id: int, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from Pending to In Progress."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    success = await db_service.update_order_status(order_id, "In Progress")
    if not success:
        raise HTTPException(status_code=404, detail="Order not found or is not in a pending state.")
    return APIResponse(success=True, message="Order moved to In Progress.", version="v1")


@router.post("/orders/{order_id}/requeue", response_model=APIResponse)
async def requeue_packing_order(order_id: int, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order from On Hold back to Pending."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    success = await db_service.requeue_held_order(order_id)
    if not success:
        raise HTTPException(status_code=404, detail="Order not found in 'On Hold' status.")
    return APIResponse(success=True, message="Order moved back to Pending queue.", version="v1")


@router.post("/orders/{order_id}/hold", response_model=APIResponse)
async def hold_packing_order(order_id: int, hold_data: HoldOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Moves an order to On Hold."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    await db_service.hold_order(order_id, hold_data.reason, hold_data.notes, hold_data.problem_item_skus)
    return APIResponse(success=True, message="Order moved to On Hold.", version="v1")


@router.post("/orders/{order_id}/fulfill", response_model=APIResponse)
async def fulfill_packing_order(order_id: int, fulfill_data: FulfillOrderRequest, request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Fulfills the order in Shopify and marks it as complete."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    success, fulfillment_id = await shopify_service.fulfill_order(
        order_id, fulfill_data.tracking_number, fulfill_data.packer_name, fulfill_data.carrier
    )
    if not success:
        raise HTTPException(status_code=502, detail="Failed to fulfill order in Shopify.")

    await db_service.complete_order_fulfillment(order_id, fulfill_data.packer_name, fulfillment_id)

    order_doc = await db_service.get_order_by_id(order_id)
    if order_doc and order_doc.get("phone_numbers"):
        customer_phone = order_doc["phone_numbers"][0]
        customer_name = order_doc.get("raw", {}).get("customer", {}).get("first_name", "")
        
        notification_message = (
            f"Great news, {customer_name}! 笨ｨ Your FeelOri order #{order_doc.get('order_number')} has been packed and is on its way!\n\n"
            f"囹 Tracking Number: {fulfill_data.tracking_number}\n"
            f"召 Carrier: {fulfill_data.carrier}\n\n"
            "We're so excited for you to receive your items!"
        )
        asyncio.create_task(whatsapp_service.send_message(customer_phone, notification_message))

    return APIResponse(success=True, message="Order fulfilled and customer notified.", version="v1")


@router.get("/metrics", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packing_metrics(request: Request, current_user: dict = Depends(verify_jwt_token)):
    """Provides key performance indicators (KPIs) for the packing workflow."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    # This call will now work correctly
    metrics = await db_service.get_packing_dashboard_metrics()
    return APIResponse(
        success=True,
        # --- Fix Start: Add the required 'message' field ---
        message="Packing metrics retrieved successfully.",
        # --- Fix End ---
        data=metrics,
        version="v1"
    )