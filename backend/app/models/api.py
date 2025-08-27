# /app/models/api.py

from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime

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
    target_type: str = Field(default="all", pattern="^(all|active|recent)$")
    target_phones: Optional[List[str]] = None

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