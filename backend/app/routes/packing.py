# /app/routes/packing.py

import os
import asyncio
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Request, Header
from fastapi.responses import FileResponse
from app.models.api import APIResponse, HoldOrderRequest, FulfillOrderRequest
from app.utils.dependencies import verify_jwt_token
from app.services.db_service import db_service
from app.services import security_service
from app.services.whatsapp_service import whatsapp_service
from app.utils.rate_limiter import limiter
from app.services.shopify_service import shopify_service
from app.config.settings import settings

# This file defines all API routes specifically for the packing team's HTML dashboard.
# All routes are protected and require JWT authentication.

# Allowed businesses for packing operations
ALLOWED_BUSINESSES = {"feelori", "goldencollections", "godjewellery9"}

router = APIRouter(
    prefix="/packing",
    tags=["Packing Dashboard"],
    dependencies=[Depends(verify_jwt_token)]
)

def _validate_business_context(x_business_id: Optional[str], user: dict) -> str:
    """
    Validate business context from header or user object.
    
    Args:
        x_business_id: Business ID from x-business-id header (for shared packers)
        user: Current user dictionary (may contain business_id for single-business admins)
        
    Returns:
        Validated business_id string
        
    Raises:
        HTTPException: 400 if business_id is missing, 403 if not in allowed list
    """
    # Prefer x_business_id header (for shared packers)
    business_id = x_business_id
    
    # Fallback to user.get("business_id") (for single-business admins)
    if not business_id:
        business_id = user.get("business_id")
    
    # Raise 400 if missing
    if not business_id:
        raise HTTPException(
            status_code=400,
            detail="Business context required. Provide 'x-business-id' header or ensure user has business_id."
        )
    
    # Raise 403 if not in ALLOWED_BUSINESSES
    if business_id not in ALLOWED_BUSINESSES:
        raise HTTPException(
            status_code=403,
            detail=f"Business '{business_id}' is not authorized for packing operations."
        )
    
    return business_id

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
async def get_packing_orders(
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Provides the list of all orders for the packing dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    formatted_orders = await db_service.get_all_packing_orders(business_id)
    return APIResponse(success=True, message="Orders retrieved", data={"orders": formatted_orders}, version="v1")

# API endpoint to get global packing configuration
@router.get("/config", response_model=APIResponse)
async def get_packing_config(current_user: dict = Depends(verify_jwt_token)):
    """Provides global packing configuration (packers and carriers)."""
    config = await db_service.get_global_packing_config()
    return APIResponse(success=True, message="Packing config retrieved", data=config, version="v1")

# API endpoint to start packing an order
@router.post("/orders/{order_id}/start", response_model=APIResponse)
async def start_packing(
    order_id: str,
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Marks an order as 'In Progress'."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    
    # --- FIX ---
    # The 'details' argument was missing. We provide an empty dictionary.
    # We also convert order_id to an integer.
    await db_service.update_order_packing_status(int(order_id), business_id, "In Progress", {})
    # --- END OF FIX ---
    
    return APIResponse(success=True, message="Order packing started.", version="v1")

# API endpoint to put an order on hold
@router.post("/orders/{order_id}/hold", response_model=APIResponse)
async def hold_order(
    order_id: str,
    hold_data: HoldOrderRequest,
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Marks an order as 'On Hold' with a reason."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    await db_service.hold_order(
        order_id=int(order_id),
        business_id=business_id,
        reason=hold_data.reason,
        notes=hold_data.notes,
        skus=hold_data.problem_item_skus
    )
    return APIResponse(success=True, message="Order put on hold.", version="v1")

# API endpoint to fulfill an order
@router.post("/orders/{order_id}/fulfill", response_model=APIResponse)
async def fulfill_order(
    order_id: str,
    fulfill_data: FulfillOrderRequest,
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Marks an order as 'Completed' in Shopify and sends a template notification."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    
    success, fulfillment_id, tracking_url = await shopify_service.fulfill_order(
        order_id=int(order_id),
        packer_name=fulfill_data.packer_name,
        tracking_number=fulfill_data.tracking_number,
        carrier=fulfill_data.carrier
    )

    if not success:
        raise HTTPException(status_code=400, detail="Failed to create fulfillment in Shopify.")

    await db_service.complete_order_fulfillment(
        order_id=int(order_id),
        business_id=business_id,
        packer_name=fulfill_data.packer_name,
        fulfillment_id=fulfillment_id
    )
    
    order_doc = await db_service.get_order_by_id(int(order_id))
    if order_doc and order_doc.get("phone_numbers"):
        customer_phone = order_doc["phone_numbers"][0]
        customer_name = (order_doc.get("raw", {}).get("customer", {}) or {}).get("first_name", "there")
        order_number = order_doc.get("order_number")
        
        # --- THIS IS THE FIX ---
        # Add the carrier name to the body_params list
        body_params = [customer_name, order_number, fulfill_data.carrier]
        # --- END OF FIX ---
        
        asyncio.create_task(
            whatsapp_service.send_template_message(
                to=customer_phone,
                template_name="shipping_update_v1",
                body_params=body_params,
                button_url_param=tracking_url
            )
        )

    return APIResponse(success=True, message="Order fulfilled in Shopify and customer notified.", version=settings.api_version)

# --- NEW FUNCTION ---
# API endpoint to move a held order back to the pending queue
@router.post("/orders/{order_id}/requeue", response_model=APIResponse)
async def requeue_order(
    order_id: str,
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Moves an order from 'On Hold' back to the 'Pending' queue."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    success = await db_service.requeue_held_order(order_id=int(order_id), business_id=business_id)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Order not found or was not on hold."
        )
    return APIResponse(success=True, message="Order moved back to pending.", version="v1")
# --- END OF NEW FUNCTION ---

# --- NEW ENDPOINT FOR REACT PERFORMANCE PAGE ---
@router.get("/packer-performance", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packer_performance_metrics(
    request: Request,
    days: int = 7, # Add a query parameter for the number of days
    current_user: dict = Depends(verify_jwt_token)
):
    """Provides advanced packer performance metrics for the React dashboard."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    
    # Performance is global (across all businesses)
    metrics = await db_service.get_packer_performance_metrics(days)
    
    return APIResponse(
        success=True,
        message="Performance metrics retrieved",
        data={"leaderboard": metrics},
        version="v1"
    )
# --- END OF NEW ENDPOINT ---

# API endpoint for packing metrics
@router.get("/metrics", response_model=APIResponse)
@limiter.limit("10/minute")
async def get_packing_metrics(
    request: Request,
    x_business_id: Optional[str] = Header(None),
    current_user: dict = Depends(verify_jwt_token)
):
    """Provides key performance indicators (KPIs) for the packing workflow."""
    security_service.EnhancedSecurityService.validate_admin_session(request, current_user)
    business_id = _validate_business_context(x_business_id, current_user)
    metrics = await db_service.get_packing_dashboard_metrics(business_id)
    return APIResponse(
        success=True,
        message="Metrics retrieved",
        data=metrics,
        version="v1"
    )


