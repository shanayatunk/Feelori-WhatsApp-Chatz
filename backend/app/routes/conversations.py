# /app/routes/conversations.py
from fastapi import APIRouter, Depends, HTTPException, Response, Query
from pydantic import BaseModel
from typing import Optional
import logging

# ✅ 1. Import settings
from app.config.settings import settings
from app.services.db_service import db_service
from app.services.whatsapp_service import whatsapp_service
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse, AssignTicketRequest, SendMessageRequest

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"],
    dependencies=[Depends(verify_jwt_token)]
)

# You can define the response model here for clarity
class TriageTicketModel(BaseModel):
    _id: str
    customer_phone: str
    order_number: str
    issue_type: str
    status: str
    created_at: str
    image_media_id: str | None = None


@router.get("/", response_model=APIResponse)
async def get_pending_triage_tickets(
    status: Optional[str] = Query(None, description="Filter by status (pending, human_needed, resolved)"),
    assigned_to: Optional[str] = Query(None, description="Filter by assigned user ID"),
    business_id: Optional[str] = Query(None, description="Filter by business ID")
):
    """Get conversation tickets with optional filtering by status, assigned_to, and business_id."""
    query = {}
    
    if status:
        query["status"] = status
    if assigned_to:
        query["assigned_to"] = assigned_to
    if business_id:
        query["business_id"] = business_id
    
    # If no filters, default to human_needed (actionable items)
    if not query:
        query["status"] = "human_needed"
    
    tickets_cursor = db_service.db.triage_tickets.find(query).sort("created_at", 1)
    tickets = await tickets_cursor.to_list(length=100)
    for ticket in tickets:
        ticket["_id"] = str(ticket["_id"])
    
    return APIResponse(
        success=True,
        message="Tickets retrieved",
        data={"tickets": tickets},
        version=settings.api_version
    )


@router.get("/stats", response_model=APIResponse)
async def get_conversation_stats():
    """Get conversation statistics (ticket counts by status)."""
    pending_count = await db_service.db.triage_tickets.count_documents({"status": "pending"})
    human_needed_count = await db_service.db.triage_tickets.count_documents({"status": "human_needed"})
    resolved_count = await db_service.db.triage_tickets.count_documents({"status": "resolved"})
    
    return APIResponse(
        success=True,
        message="Conversation stats retrieved",
        data={
            "stats": {
                "open": human_needed_count,  # "Open" usually implies actionable by human
                "pending": pending_count,     # Keep track of bot-active tickets
                "resolved": resolved_count,
                "total": human_needed_count + pending_count + resolved_count
            }
        },
        version=settings.api_version
    )


@router.get("/media/{media_id}")
async def get_media(media_id: str):
    """Retrieves and returns the media file from WhatsApp."""
    try:
        image_bytes, mime_type = await whatsapp_service.get_media_content(media_id)
        if not image_bytes:
            raise HTTPException(status_code=404, detail="Media not found")
        return Response(content=image_bytes, media_type=mime_type)
    except Exception as e:
        logger.error(f"Failed to retrieve media {media_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve media")


@router.put("/{ticket_id}/resolve", response_model=APIResponse)
# ✅ 2. Removed redundant dependency
async def resolve_ticket(ticket_id: str):
    success = await db_service.resolve_triage_ticket(ticket_id)
    if not success:
        raise HTTPException(status_code=404, detail="Ticket not found or already resolved")
    # ✅ 3. Added version to the response
    return APIResponse(
        success=True,
        message="Ticket resolved successfully",
        version=settings.api_version
    )


@router.get("/{ticket_id}/messages", response_model=APIResponse)
async def get_chat_history(
    ticket_id: str,
    limit: int = Query(50, ge=1, le=200, description="Maximum number of messages to return")
):
    """Get chat history for a ticket (retrieves phone number from ticket)."""
    # Retrieve the ticket to get the customer phone number
    ticket = await db_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    customer_phone = ticket.get("customer_phone")
    if not customer_phone:
        raise HTTPException(status_code=400, detail="Ticket missing customer phone number")
    
    messages = await db_service.get_chat_history(customer_phone, limit=limit)
    
    return APIResponse(
        success=True,
        message="Chat history retrieved",
        data={"messages": messages},
        version=settings.api_version
    )


@router.post("/{ticket_id}/assign", response_model=APIResponse)
async def assign_ticket(ticket_id: str, assign_data: AssignTicketRequest):
    """Assign a ticket to a user."""
    success = await db_service.assign_ticket(ticket_id, assign_data.user_id)
    if not success:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    return APIResponse(
        success=True,
        message="Ticket assigned successfully",
        version=settings.api_version
    )


@router.post("/{ticket_id}/send", response_model=APIResponse)
async def send_manual_message(
    ticket_id: str,
    message_data: SendMessageRequest,
    current_user: dict = Depends(verify_jwt_token)
):
    """Send a manual message to a customer (retrieves phone number from ticket)."""
    # Retrieve the ticket to get the customer phone number
    ticket = await db_service.get_ticket_by_id(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    
    customer_phone = ticket.get("customer_phone")
    if not customer_phone:
        raise HTTPException(status_code=400, detail="Ticket missing customer phone number")
    
    user_id = current_user.get("username") or current_user.get("user_id", "unknown")
    
    # Send the message via WhatsApp
    wamid = await whatsapp_service.send_message(
        to_phone=customer_phone,
        message=message_data.message
    )
    
    if not wamid:
        raise HTTPException(status_code=500, detail="Failed to send message")
    
    # Log the manual message
    await db_service.log_manual_message(customer_phone, message_data.message, user_id)
    
    return APIResponse(
        success=True,
        message="Message sent successfully",
        data={"wamid": wamid},
        version=settings.api_version
    )

