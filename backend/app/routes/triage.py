# /app/routes/triage.py
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
import logging

# ✅ 1. Import settings
from app.config.settings import settings
from app.services.db_service import db_service
from app.services.whatsapp_service import whatsapp_service
from app.utils.dependencies import verify_jwt_token
from app.models.api import APIResponse

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/triage",
    tags=["Triage"],
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
# ✅ 2. Removed redundant dependency
async def get_pending_triage_tickets():
    tickets_cursor = db_service.db.triage_tickets.find({"status": "pending"}).sort("created_at", 1)
    tickets = await tickets_cursor.to_list(length=100)
    for ticket in tickets:
        ticket["_id"] = str(ticket["_id"])
    # ✅ 3. Added version to the response
    return APIResponse(
        success=True,
        message="Tickets retrieved",
        data={"tickets": tickets},
        version=settings.api_version
    )


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