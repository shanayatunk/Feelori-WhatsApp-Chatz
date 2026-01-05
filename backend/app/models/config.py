# /app/models/config.py

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from beanie import Document, Indexed


class RuleConfig(BaseModel):
    """
    Static rule configuration for business message handling.
    Rules are read-only at runtime and define patterns and actions.
    """
    name: str = Field(..., description="Rule name/identifier")
    patterns: List[str] = Field(..., description="List of regex patterns or keywords to match")
    action: str = Field(..., description="Action to take (e.g., 'reply', 'handoff')")
    response_text: Optional[str] = Field(default=None, description="Response text if action is 'reply'")
    priority: int = Field(default=0, description="Rule priority (higher = evaluated first)")


class BusinessConfig(Document):
    """
    MongoDB-backed, multi-tenant business configuration model.
    Stores persona settings and static rules for each business.
    
    This is a STATIC configuration model (read-only at runtime).
    No runtime or conversation state is stored here.
    """
    business_id: Indexed(str, unique=True)
    persona: Dict[str, Any]
    rules: List[RuleConfig] = Field(default_factory=list)
    
    class Settings:
        name = "business_configs"  # MongoDB collection name
        indexes = [
            [("business_id", 1)],  # Unique index on business_id
        ]

