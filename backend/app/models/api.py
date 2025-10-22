# /app/models/api.py

from pydantic import BaseModel, Field, field_serializer
from typing import List, Dict, Optional
from datetime import datetime
from bson import ObjectId

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
    class Config:
        # Pydantic V1 style (still works in V2 for compatibility)
        # Tell Pydantic how to serialize ObjectId types when encountered
        json_encoders = {
            ObjectId: str
        }
        # For Pydantic V2 native style, you might use serialization_alias or custom serializers,
        # but json_encoders is usually sufficient for FastAPI integration.
        # Ensure you have `arbitrary_types_allowed = True` if using complex types directly.
        arbitrary_types_allowed = True

class HoldOrderRequest(BaseModel):
    reason: str
    problem_item_skus: Optional[List[str]] = None
    notes: Optional[str] = None

class FulfillOrderRequest(BaseModel):
    packer_name: str
    tracking_number: str
    carrier: str

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
        orm_mode = True # Deprecated in V2, use from_attributes = True
        from_attributes = True
        allow_population_by_field_name = True
        json_encoders = { # <-- This is correct here too
            ObjectId: str
        }