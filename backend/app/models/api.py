# /app/models/api.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Union
from datetime import datetime
from bson import ObjectId
from enum import Enum

# This file contains Pydantic models that define the structure of data for
# API requests and responses, ensuring type safety and validation.

class LoginRequest(BaseModel):
    password: str = Field(..., min_length=12, max_length=255)

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class BroadcastRequest(BaseModel):
    message: str
    target_type: str = Field(default="all", pattern="^(all|active|recent|custom_group)$")
    target_phones: Optional[List[str]] = None
    image_url: Optional[str] = None
    target_group_id: Optional[str] = None

class APIResponse(BaseModel):
    success: bool
    message: str
    data: Optional[Dict] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    version: str

class HoldOrderRequest(BaseModel):
    reason: str
    problem_item_skus: Optional[List[str]] = None
    notes: Optional[str] = None

class FulfillOrderRequest(BaseModel):
    packer_name: str
    tracking_number: str
    carrier: str

class PackerRequest(BaseModel):
    name: str

class PackingOrder(BaseModel):
    """Model for packing dashboard orders."""
    id: int  # Shopify order ID
    order_id: Optional[str] = None  # Alias for frontend compatibility
    order_number: Union[str, int]
    name: Optional[str] = None  # Critical for "FO1067"
    business_id: Optional[str] = None
    status: str
    created_at: Optional[datetime] = None
    packer_name: Optional[str] = None
    customer: Optional[Dict] = None
    items: Optional[List[Dict]] = None
    notes: Optional[str] = None
    hold_reason: Optional[str] = None
    problem_item_skus: Optional[List[str]] = None
    previously_on_hold_reason: Optional[str] = None
    previously_problem_skus: Optional[List[str]] = None

class Rule(BaseModel):
    name: str
    keywords: List[str]
    phrases: List[str]

class StringResource(BaseModel):
    key: str
    value: str

# --- FIX START: Add the missing models ---

class OrderUpdate(BaseModel):
    status: Optional[str] = None
    packer_name: Optional[str] = None

class OrderHold(BaseModel):
    reason: str
    notes: Optional[str] = None
    problem_item_skus: Optional[List[str]] = None

# --- FIX END ---

class StringUpdateRequest(BaseModel):
    strings: List[StringResource]

class BroadcastGroup(BaseModel):
    name: str = Field(..., min_length=3, max_length=50)
    phone_numbers: List[str]

class BroadcastGroupCreate(BroadcastGroup):
    pass

class BroadcastGroupResponse(BroadcastGroup):
    id: str = Field(..., alias="_id")

    class Config:
        orm_mode = True
        allow_population_by_field_name = True
        json_encoders = {
            ObjectId: str
        }

class TemplateBroadcastRequest(BaseModel):
    """Request model for template-based WhatsApp broadcasts."""
    business_id: str
    recipients: List[str]
    template_name: str
    variables: Dict = Field(default_factory=dict)  # e.g., {"body_params": [...], "header_text_param": "...", "button_url_param": "..."}
    confirmation: Optional[str] = None
    dry_run: bool = True

class ConversationStatus(str, Enum):
    """Status values for conversation tickets."""
    PENDING = "pending"
    HUMAN_NEEDED = "human_needed"
    RESOLVED = "resolved"

class ConversationModel(BaseModel):
    """Model for conversation tickets with assignment and status tracking."""
    _id: str
    customer_phone: str
    order_number: str
    issue_type: str
    status: ConversationStatus = ConversationStatus.PENDING
    assigned_to: Optional[str] = None
    business_id: str = "feelori"
    created_at: datetime
    updated_at: Optional[datetime] = None
    image_media_id: Optional[str] = None

class AssignTicketRequest(BaseModel):
    """Request model for assigning a ticket to a user."""
    user_id: str

class SendMessageRequest(BaseModel):
    """Request model for sending a manual message."""
    message: str