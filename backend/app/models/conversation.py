# /app/models/conversation.py

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict


class MessagePayload(BaseModel):
    """Represents a message in a conversation."""
    type: str = Field(default="text", description="Message type (text, image, video, etc.)")
    text: Optional[str] = Field(default=None, description="Message text content")
    media_url: Optional[str] = Field(default=None, description="URL for media attachments")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class ConversationContext(BaseModel):
    """Contextual information for a conversation."""
    shopify_store: Optional[str] = Field(default=None, description="Shopify store identifier")
    last_product_id: Optional[str] = Field(default=None, description="Last product ID mentioned")
    abandoned_checkout_id: Optional[str] = Field(default=None, description="Abandoned checkout ID if applicable")


class Conversation(BaseModel):
    """Canonical conversation schema for multi-tenant chat management."""
    id: Optional[str] = Field(default=None, alias="_id", description="MongoDB document ID")
    tenant_id: str = Field(..., description="Tenant identifier (required)")
    channel: str = Field(default="whatsapp", description="Communication channel")
    external_user_id: str = Field(..., description="External user identifier (phone number)")
    wa_conversation_id: Optional[str] = Field(default=None, description="WhatsApp conversation ID")
    status: str = Field(default="open", description="Conversation status")
    ai_enabled: bool = Field(default=True, description="Whether AI is enabled for this conversation")
    tags: List[str] = Field(default_factory=list, description="Conversation tags")
    context: ConversationContext = Field(
        default_factory=ConversationContext,
        description="Conversation context information"
    )
    last_message: Optional[MessagePayload] = Field(default=None, description="Last message in the conversation")
    last_message_at: Optional[datetime] = Field(default=None, description="Timestamp of last message")
    created_at: Optional[datetime] = Field(default=None, description="Conversation creation timestamp")
    updated_at: Optional[datetime] = Field(default=None, description="Last update timestamp")
    
    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True
    )

