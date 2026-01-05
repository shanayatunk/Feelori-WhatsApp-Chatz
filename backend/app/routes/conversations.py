# /app/routes/conversations.py
# 
# NEW CODE: Super Admin Implementation
# This file has been rewritten to support Super Admin functionality.
# Admin users can now see and manage conversations from all businesses
# (feelori, goldencollections, godjewellery9) while preserving data integrity.

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import Response
from bson import ObjectId
from datetime import datetime, timezone
from typing import Optional
from pydantic import BaseModel

from app.dependencies.tenant import get_tenant_id
from app.services.db_service import db_service
from app.services.whatsapp_service import whatsapp_service
from app.models.api import APIResponse
from app.config.settings import settings

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)

# --- HELPER: SUPER ADMIN LOGIC ---
def get_tenant_filter(tenant_id: str):
    """
    Returns a MongoDB query filter that allows Admins to access 
    sub-businesses (feelori, goldencollections, etc).
    """
    # 1. Standardize
    tenant_candidates = [tenant_id, tenant_id.lower(), tenant_id.strip()]
    
    # 2. Super Admin List (The "Master Keys")
    if tenant_id.lower() in ["admin", "superadmin", "administrator", "feelori", "master"]:
        tenant_candidates.extend(["feelori", "goldencollections", "godjewellery9"])
    
    # 3. Return Filter
    return {
        "$or": [
            {"tenant_id": {"$in": tenant_candidates}},
            {"business_id": {"$in": tenant_candidates}}
        ]
    }

# Request models
class AssignRequest(BaseModel):
    user_id: str

class AIControlRequest(BaseModel):
    enabled: bool

class SendMessageRequest(BaseModel):
    message: str

@router.get("/", response_model=APIResponse)
async def list_conversations(
    status: Optional[str] = Query(None),
    limit: int = Query(20, ge=1, le=100),
    cursor: Optional[str] = Query(None),
    tenant_id: str = Depends(get_tenant_id)
):
    """List conversations with cursor-based pagination."""
    # 1. Use Super Admin Filter
    query_filter = get_tenant_filter(tenant_id)
    
    if status:
        query_filter["status"] = status
    
    if cursor:
        try:
            cursor_dt = datetime.fromisoformat(cursor.replace("Z", "+00:00"))
            query_filter["updated_at"] = {"$lt": cursor_dt}
        except (ValueError, AttributeError) as e:
            raise HTTPException(status_code=400, detail=f"Invalid cursor: {str(e)}")
    
    try:
        cursor_obj = db_service.db.conversations.find(query_filter).sort("updated_at", -1).limit(limit + 1)
        conversations = await cursor_obj.to_list(length=limit + 1)
        
        has_more = len(conversations) > limit
        if has_more:
            conversations = conversations[:limit]
        
        data = []
        next_cursor = None
        
        for conv in conversations:
            conv_id = str(conv.get("_id", ""))
            preview = ""
            last_message = conv.get("last_message")
            if last_message and isinstance(last_message, dict):
                preview = last_message.get("text", "")[:100]
            
            last_at = conv.get("last_message_at")
            if last_at:
                last_at = last_at.isoformat() if isinstance(last_at, datetime) else str(last_at)
            
            data.append({
                "id": conv_id,
                "phone": conv.get("external_user_id", ""),
                "preview": preview,
                "status": conv.get("status", "open"),
                "last_at": last_at,
                "ai_enabled": conv.get("ai_enabled", True),
                "ai_paused_by": conv.get("ai_paused_by"),
                "assigned_to": conv.get("assigned_to")
            })
        
        if has_more and conversations:
            last_updated = conversations[-1].get("updated_at")
            if last_updated:
                next_cursor = last_updated.isoformat() if isinstance(last_updated, datetime) else str(last_updated)
    
    return APIResponse(
        success=True,
            message="Conversations retrieved successfully",
            data={"data": data, "next_cursor": next_cursor},
        version=settings.api_version
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list: {str(e)}")

@router.get("/stats", response_model=APIResponse)
async def get_conversation_stats(tenant_id: str = Depends(get_tenant_id)):
    """Get conversation statistics aggregated by status."""
    try:
        stats = {"open": 0, "resolved": 0, "triaged": 0, "total": 0}
        
        # 1. Use Super Admin Filter
        pipeline = [
            {"$match": get_tenant_filter(tenant_id)},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}}
        ]
        
        results = await db_service.db.conversations.aggregate(pipeline).to_list(length=None)
        
        for result in results:
            status = str(result.get("_id", "")).lower()
            if status in stats:
                stats[status] = result.get("count", 0)
        
        stats["total"] = sum(r.get("count", 0) for r in results)
    
    return APIResponse(
        success=True,
            message="Conversation statistics retrieved successfully",
            data={"stats": stats},
        version=settings.api_version
    )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{conversation_id}", response_model=APIResponse)
async def get_conversation_thread(conversation_id: str, tenant_id: str = Depends(get_tenant_id)):
    """Get a specific conversation thread with all its messages."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Use Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        conversation = await db_service.db.conversations.find_one(query)
        if not conversation:
            raise HTTPException(status_code=404, detail="Not found")
        
        conversation["_id"] = str(conversation["_id"])
        
        # 2. Fetch Messages with same filter
        msg_query = {"conversation_id": conv_oid}
        msg_query.update(get_tenant_filter(tenant_id))
        
        messages = await db_service.db.message_logs.find(msg_query).sort("_id", 1).to_list(None)
        
        # Normalize
        for msg in messages:
            msg["_id"] = str(msg.get("_id"))
            msg["conversation_id"] = str(msg.get("conversation_id"))
            
            if not msg.get("text") and msg.get("content"):
                msg["text"] = msg["content"]
            
            ts = msg.get("timestamp") or msg.get("created_at")
            if ts and isinstance(ts, datetime):
                msg["timestamp"] = ts.isoformat()
            
            # Map sender
            if msg.get("direction") == "inbound":
                msg["sender"] = "user"
            else:
                src = msg.get("source")
                msg["sender"] = "agent" if src == "agent" else "bot"

        return APIResponse(
            success=True,
            message="Conversation thread retrieved successfully",
            data={"conversation": conversation, "messages": messages},
            version=settings.api_version
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/media/{media_id}")
async def get_media(
    media_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """
    Proxy to fetch media from WhatsApp and return it as a binary stream.
    """
    try:
        # Determine business_id. 
        # If Admin, we default to "feelori". 
        # (Future improvement: lookup message owner in DB to support multiple businesses for Admin)
        business_id = tenant_id.lower()
        if business_id in ["admin", "superadmin", "administrator"]:
            business_id = "feelori"
            
        file_content, mime_type = await whatsapp_service.get_media_content(media_id, business_id)
        
        if not file_content:
            raise HTTPException(status_code=404, detail="Media not found or could not be downloaded")
            
        return Response(content=file_content, media_type=mime_type)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch media: {str(e)}")

@router.post("/{conversation_id}/send", response_model=APIResponse)
async def send_agent_message(
    conversation_id: str,
    message_data: SendMessageRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """Send an agent reply message in a conversation."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Find conversation using Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        conversation = await db_service.db.conversations.find_one(query)
        if not conversation:
            raise HTTPException(status_code=404, detail="Not found")
            
        # 2. Use ACTUAL tenant_id (e.g. 'goldencollections') not 'admin'
        real_tenant_id = conversation.get("tenant_id") or conversation.get("business_id") or tenant_id
        now = datetime.now(timezone.utc)
        
        # Insert Message
        msg_doc = {
            "tenant_id": real_tenant_id,
            "business_id": real_tenant_id,
            "conversation_id": conv_oid,
            "source": "agent",
            "type": "text",
            "text": message_data.message,
            "content": message_data.message,
            "created_at": now,
            "timestamp": now,
            "direction": "outbound"
        }
        res = await db_service.db.message_logs.insert_one(msg_doc)
        
        # Insert Outbound
        await db_service.db.outbound_messages.insert_one({
            "tenant_id": real_tenant_id,
            "conversation_id": conv_oid,
            "channel": "whatsapp",
            "recipient": conversation["external_user_id"],
            "payload": {"type": "text", "text": message_data.message},
            "source": "agent",
            "status": "pending",
            "attempts": 0,
            "last_error": None,
            "created_at": now,
            "sent_at": None
        })
        
        # Update Conversation
        await db_service.db.conversations.update_one(
            {"_id": conv_oid},
            {"$set": {
                "last_message": {"type": "text", "text": message_data.message},
                "last_message_at": now,
                "updated_at": now,
                "status": "open",
                "ai_enabled": False,
                "ai_paused_by": "agent"
            }}
        )
        
    return APIResponse(
        success=True,
            message="Message sent successfully",
            data={"message_id": str(res.inserted_id)},
        version=settings.api_version
    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{conversation_id}/assign", response_model=APIResponse)
async def assign_conversation(
    conversation_id: str,
    assign_data: AssignRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """Assign a conversation to an agent."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Use Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        now = datetime.now(timezone.utc)
        result = await db_service.db.conversations.update_one(
            query,
            {
                "$set": {
                    "assigned_to": assign_data.user_id,
                    "status": "human_needed",
                    "ai_enabled": False,
                    "ai_paused_by": "agent",
                    "updated_at": now
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Not found")
    
    return APIResponse(
        success=True,
            message="Conversation assigned successfully",
        version=settings.api_version
    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/{conversation_id}/resolve", response_model=APIResponse)
async def resolve_conversation(
    conversation_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Mark a conversation as resolved."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Use Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        now = datetime.now(timezone.utc)
        result = await db_service.db.conversations.update_one(
            query,
            {
                "$set": {
                    "status": "resolved",
                    "updated_at": now
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Not found")
    
    return APIResponse(
        success=True,
            message="Conversation resolved successfully",
        version=settings.api_version
    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{conversation_id}/ai", response_model=APIResponse)
async def control_ai(
    conversation_id: str,
    ai_data: AIControlRequest,
    tenant_id: str = Depends(get_tenant_id)
):
    """Pause or resume AI for a conversation."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Use Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        now = datetime.now(timezone.utc)
        if ai_data.enabled:
            update_doc = {
                "$set": {
                    "ai_enabled": True,
                    "ai_paused_by": None,
                    "updated_at": now
                }
            }
        else:
            update_doc = {
                "$set": {
                    "ai_enabled": False,
                    "ai_paused_by": "agent",
                    "updated_at": now
                }
            }
        
        result = await db_service.db.conversations.update_one(query, update_doc)
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Not found")
    
    return APIResponse(
        success=True,
            message=f"AI {'enabled' if ai_data.enabled else 'paused'} successfully",
        version=settings.api_version
    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/{conversation_id}/release", response_model=APIResponse)
async def release_conversation(
    conversation_id: str,
    tenant_id: str = Depends(get_tenant_id)
):
    """Release a conversation back to the AI bot."""
    try:
        conv_oid = ObjectId(conversation_id)
        
        # 1. Use Super Admin Filter
        query = {"_id": conv_oid}
        query.update(get_tenant_filter(tenant_id))
        
        now = datetime.now(timezone.utc)
        result = await db_service.db.conversations.update_one(
            query,
            {
                "$set": {
                    "ai_enabled": True,
                    "ai_paused_by": None,
                    "status": "open",
                    "updated_at": now
                }
            }
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Not found")
    
    return APIResponse(
        success=True,
            message="Conversation released to bot",
        version=settings.api_version
    )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))