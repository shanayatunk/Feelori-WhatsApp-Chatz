# /app/routes/conversations.py

from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel

from app.dependencies.tenant import get_tenant_id
from app.services.db_service import db_service
from app.models.api import APIResponse
from app.config.settings import settings

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


# Request models for conversation actions
class AssignRequest(BaseModel):
    user_id: str


class AIControlRequest(BaseModel):
    enabled: bool


class SendMessageRequest(BaseModel):
    message: str


@router.get("/", response_model=APIResponse)
async def list_conversations(
    status: Optional[str] = Query(None, description="Filter by conversation status"),
    limit: int = Query(20, ge=1, le=100, description="Number of conversations to return"),
    cursor: Optional[str] = Query(None, description="Cursor for pagination (ISO datetime string)"),
    tenant_id: str = Depends(get_tenant_id)
):
    """
    List conversations with cursor-based pagination.
    
    Returns a paginated list of conversations for the authenticated tenant.
    """
    # Build query filter
    query_filter = {"tenant_id": tenant_id}
    
    # Add status filter if provided
    if status:
        query_filter["status"] = status
    
    # Add cursor filter if provided (for pagination)
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            query_filter["updated_at"] = {"$lt": cursor_dt}
        except (ValueError, AttributeError) as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid cursor format. Expected ISO datetime string: {str(e)}"
            )
    
    try:
        # Query conversations sorted by updated_at descending
        cursor_obj = db_service.db.conversations.find(query_filter).sort("updated_at", -1).limit(limit + 1)
        conversations = await cursor_obj.to_list(length=limit + 1)
        
        # Determine if there are more results (next page)
        has_more = len(conversations) > limit
        if has_more:
            conversations = conversations[:limit]
        
        # Build simplified response objects
        data = []
        next_cursor = None
        
        for conv in conversations:
            # Convert ObjectId to string
            conv_id = str(conv.get("_id", ""))
            
            # Extract preview text from last_message
            preview = ""
            last_message = conv.get("last_message")
            if last_message:
                preview = last_message.get("text", "")[:100] if isinstance(last_message, dict) else ""
            
            # Extract phone number
            phone = conv.get("external_user_id", "")
            
            # Extract status
            conv_status = conv.get("status", "open")
            
            # Extract last_message_at timestamp
            last_at = conv.get("last_message_at")
            if last_at:
                if isinstance(last_at, datetime):
                    last_at = last_at.isoformat()
                else:
                    last_at = str(last_at)
            else:
                last_at = None
            
            data.append({
                "id": conv_id,
                "phone": phone,
                "preview": preview,
                "status": conv_status,
                "last_at": last_at,
                "ai_enabled": conv.get("ai_enabled", True),
                "ai_paused_by": conv.get("ai_paused_by"),
                "assigned_to": conv.get("assigned_to")
            })
        
        # Set next cursor if there are more results
        if has_more and conversations:
            last_conv = conversations[-1]
            last_updated = last_conv.get("updated_at")
            if last_updated:
                if isinstance(last_updated, datetime):
                    next_cursor = last_updated.isoformat()
                else:
                    next_cursor = str(last_updated)
        
        # Return dict with "data" (list) and "next_cursor" as specified
        return APIResponse(
            success=True,
            message="Conversations retrieved successfully",
            data={
                "data": data,
                "next_cursor": next_cursor
            },
            version=settings.api_version
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversations: {str(e)}"
        )


@router.get("/stats", response_model=APIResponse)
async def get_conversation_stats(
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Get conversation statistics aggregated by status for the authenticated tenant.
    
    Returns counts for open, resolved, triaged, and total conversations.
    """
    try:
        # Initialize defaults to ensure frontend never crashes
        stats = {"open": 0, "resolved": 0, "triaged": 0, "total": 0}
        
        # Build aggregation pipeline
        pipeline = [
            # Match conversations for this tenant
            {"$match": {"tenant_id": tenant_id}},
            # Group by status and count documents
            {
                "$group": {
                    "_id": "$status",
                    "count": {"$sum": 1}
                }
            }
        ]
        
        # Execute aggregation
        results = await db_service.db.conversations.aggregate(pipeline).to_list(length=None)
        
        # Merge aggregation results into stats dictionary
        for result in results:
            status = result.get("_id", "").lower()
            count = result.get("count", 0)
            
            # Map status values to our stats keys
            if status == "open":
                stats["open"] = count
            elif status == "resolved":
                stats["resolved"] = count
            elif status == "triaged":
                stats["triaged"] = count
            # Handle any other status values (future-proof for new statuses)
        
        # Calculate total dynamically: sum ALL aggregation results
        # This ensures correctness even if new statuses are added
        stats["total"] = sum(r.get("count", 0) for r in results)
        
        return APIResponse(
            success=True,
            message="Conversation statistics retrieved successfully",
            data={"stats": stats},
            version=settings.api_version
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation statistics: {str(e)}"
        )


@router.get("/{conversation_id}", response_model=APIResponse)
async def get_conversation_thread(
    conversation_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Get a specific conversation thread with all its messages.
    
    Returns the conversation details and all associated messages.
    """
    try:
        # Convert conversation_id to ObjectId
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conversation ID format: {conversation_id}"
            )
        
        # Find conversation by _id AND tenant_id (security: tenant isolation)
        conversation = await db_service.db.conversations.find_one({
            "_id": conv_object_id,
            "tenant_id": tenant_id
        })
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        # Convert ObjectId to string for JSON serialization
        conversation["_id"] = str(conversation["_id"])
        
        # Find all messages for this conversation, sorted by _id ascending (chronological)
        # Use ObjectId for referential integrity (matches conversations._id)
        # Support both tenant_id (new) and business_id (legacy) for backward compatibility
        messages_cursor = db_service.db.message_logs.find({
            "conversation_id": conv_object_id,
            "$or": [
                {"tenant_id": tenant_id},
                {"business_id": tenant_id}
            ]
        }).sort("_id", 1)
        
        messages = await messages_cursor.to_list(length=None)
        
        # --- NORMALIZE IN-PLACE ---
        for msg in messages:
            # 1. Convert ObjectIds to strings (Prevent 500 Error)
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])
            if "conversation_id" in msg:
                msg["conversation_id"] = str(msg["conversation_id"])
            
            # 2. Guarantee 'text' exists (Frontend Source of Truth)
            if not msg.get("text") and msg.get("content"):
                msg["text"] = msg["content"]
            
            # 3. Guarantee 'timestamp' exists
            if not msg.get("timestamp") and msg.get("created_at"):
                msg["timestamp"] = msg.get("created_at")

            # 4. Serialize Datetimes safely
            if "timestamp" in msg and isinstance(msg["timestamp"], datetime):
                msg["timestamp"] = msg["timestamp"].isoformat()
            if "created_at" in msg and isinstance(msg["created_at"], datetime):
                msg["created_at"] = msg["created_at"].isoformat()

            # 5. NORMALIZE SENDER (CRITICAL FIX)
            # Map direction/source to 'user', 'bot', or 'agent'
            direction = msg.get("direction")
            source = msg.get("source")
            
            if direction == "inbound":
                msg["sender"] = "user"
            else:
                # Outbound Logic
                if source in ["ai", "bot"]:
                    msg["sender"] = "bot"
                elif source == "agent":
                    msg["sender"] = "agent"
                else:
                    msg["sender"] = "bot" # Default fallback
        
        return APIResponse(
            success=True,
            message="Conversation thread retrieved successfully",
            data={
                "conversation": conversation,
                "messages": messages
            },
            version=settings.api_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve conversation thread: {str(e)}"
        )


@router.post("/{conversation_id}/assign", response_model=APIResponse)
async def assign_conversation(
    conversation_id: str,
    assign_data: AssignRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Assign a conversation to an agent.
    """
    try:
        # Convert conversation_id to ObjectId
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conversation ID format: {conversation_id}"
            )
        
        # Update conversation with tenant_id check (security)
        now = datetime.now(timezone.utc)
        result = await db_service.db.conversations.update_one(
            {
                "_id": conv_object_id,
                "tenant_id": tenant_id  # Security: ensure tenant isolation
            },
            {
                "$set": {
                    "assigned_to": assign_data.user_id,
                    "updated_at": now
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return APIResponse(
            success=True,
            message="Conversation assigned successfully",
            version=settings.api_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to assign conversation: {str(e)}"
        )


@router.put("/{conversation_id}/resolve", response_model=APIResponse)
async def resolve_conversation(
    conversation_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Mark a conversation as resolved.
    """
    try:
        # Convert conversation_id to ObjectId
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conversation ID format: {conversation_id}"
            )
        
        # Update conversation with tenant_id check (security)
        now = datetime.now(timezone.utc)
        result = await db_service.db.conversations.update_one(
            {
                "_id": conv_object_id,
                "tenant_id": tenant_id  # Security: ensure tenant isolation
            },
            {
                "$set": {
                    "status": "resolved",
                    "updated_at": now
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return APIResponse(
            success=True,
            message="Conversation resolved successfully",
            version=settings.api_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to resolve conversation: {str(e)}"
        )


@router.post("/{conversation_id}/ai", response_model=APIResponse)
async def control_ai(
    conversation_id: str,
    ai_data: AIControlRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Pause or resume AI for a conversation.
    """
    try:
        # Convert conversation_id to ObjectId
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conversation ID format: {conversation_id}"
            )
        
        # Build update document based on enabled flag
        now = datetime.now(timezone.utc)
        if ai_data.enabled:
            # Resume AI: enable AI and clear paused_by
            update_doc = {
                "$set": {
                    "ai_enabled": True,
                    "ai_paused_by": None,
                    "updated_at": now
                }
            }
        else:
            # Pause AI: disable AI and set paused_by to "agent"
            update_doc = {
                "$set": {
                    "ai_enabled": False,
                    "ai_paused_by": "agent",
                    "updated_at": now
                }
            }
        
        # Update conversation with tenant_id check (security)
        result = await db_service.db.conversations.update_one(
            {
                "_id": conv_object_id,
                "tenant_id": tenant_id  # Security: ensure tenant isolation
            },
            update_doc
        )
        
        if result.matched_count == 0:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        return APIResponse(
            success=True,
            message=f"AI {'enabled' if ai_data.enabled else 'paused'} successfully",
            version=settings.api_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to control AI: {str(e)}"
        )


@router.post("/{conversation_id}/send", response_model=APIResponse)
async def send_agent_message(
    conversation_id: str,
    message_data: SendMessageRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Send an agent reply message in a conversation.
    """
    try:
        # Convert conversation_id to ObjectId
        try:
            conv_object_id = ObjectId(conversation_id)
        except Exception:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid conversation ID format: {conversation_id}"
            )
        
        # Verify conversation exists and belongs to tenant (security)
        conversation = await db_service.db.conversations.find_one({
            "_id": conv_object_id,
            "tenant_id": tenant_id  # Security: ensure tenant isolation
        })
        
        if not conversation:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found"
            )
        
        now = datetime.now(timezone.utc)
        
        # 1. Insert message into message_logs
        # Use ObjectId for referential integrity (matches conversations._id)
        # Write BOTH tenant_id/business_id and text/content pairs to prevent future drift
        message_doc = {
            "tenant_id": tenant_id,
            "business_id": tenant_id,  # Dual field for backward compatibility
            "conversation_id": conv_object_id,  # Use ObjectId, not string
            "source": "agent",
            "type": "text",
            "text": message_data.message,      # Frontend source of truth
            "content": message_data.message,   # Legacy/backend compatibility
            "created_at": now,
            "timestamp": now                   # Dual field for backward compatibility
        }
        
        message_result = await db_service.db.message_logs.insert_one(message_doc)
        message_id = str(message_result.inserted_id)
        
        # 2. Insert into outbound_messages ledger for async processing
        outbound_doc = {
            "tenant_id": tenant_id,
            "conversation_id": conv_object_id,  # Must be ObjectId for referential integrity
            "channel": "whatsapp",
            "recipient": conversation["external_user_id"],  # Phone number
            "payload": {
                "type": "text",
                "text": message_data.message
            },
            "source": "agent",
            "status": "pending",  # Critical: worker will pick this up
            "attempts": 0,
            "last_error": None,
            "created_at": now,
            "sent_at": None
        }
        
        await db_service.db.outbound_messages.insert_one(outbound_doc)
        
        # 3. Update conversation
        last_message = {
            "type": "text",
            "text": message_data.message
        }
        
        await db_service.db.conversations.update_one(
            {
                "_id": conv_object_id,
                "tenant_id": tenant_id  # Security: ensure tenant isolation
            },
            {
                "$set": {
                    "last_message": last_message,
                    "last_message_at": now,
                    "updated_at": now,
                    "status": "open",  # Re-open if resolved
                    "ai_enabled": False,  # Critical: Disable AI when agent sends message
                    "ai_paused_by": "agent"  # Critical: Track that agent paused AI
                }
            }
        )
        
        return APIResponse(
            success=True,
            message="Message sent successfully",
            data={"message_id": message_id},
            version=settings.api_version
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send message: {str(e)}"
        )
