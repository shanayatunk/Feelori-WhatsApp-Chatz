# /app/routes/conversations.py

from fastapi import APIRouter, Depends, HTTPException, Query
from bson import ObjectId
from datetime import datetime
from typing import Optional

from app.dependencies.tenant import get_tenant_id
from app.services.db_service import db_service
from app.models.api import APIResponse
from app.config.settings import settings

router = APIRouter(
    prefix="/conversations",
    tags=["Conversations"]
)


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
                "last_at": last_at
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
            # Handle any other status values by adding to total
        
        # Calculate total: sum of open, resolved, and triaged
        stats["total"] = stats["open"] + stats["resolved"] + stats["triaged"]
        
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
        
        # Find all messages for this conversation, sorted by created_at ascending
        messages_cursor = db_service.db.message_logs.find({
            "conversation_id": conversation_id,
            "tenant_id": tenant_id
        }).sort("created_at", 1)
        
        messages = await messages_cursor.to_list(length=None)
        
        # Convert ObjectIds to strings in messages
        for msg in messages:
            if "_id" in msg:
                msg["_id"] = str(msg["_id"])
        
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
