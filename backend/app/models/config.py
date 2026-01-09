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


class KnowledgeEntry(BaseModel):
    """Single knowledge base entry with triggers and response value."""
    value: str = Field(..., description="The exact text to reply with")
    triggers: List[str] = Field(default_factory=list, description="Keywords that trigger this answer")
    enabled: bool = True


class LocationConfig(BaseModel):
    """Simple configuration for store location details."""
    store_address: str = Field(default="", description="Physical address of the store")
    operating_hours: str = Field(default="", description="e.g. Mon-Sat 10am-7pm")
    google_maps_link: str = Field(default="", description="Link to Google Maps")


class KnowledgeBase(BaseModel):
    """Knowledge base for authoritative answers on policies, social media, and custom FAQs."""
    social_media: Dict[str, KnowledgeEntry] = Field(default_factory=dict)
    policies: Dict[str, KnowledgeEntry] = Field(default_factory=dict)
    custom_faqs: Dict[str, KnowledgeEntry] = Field(default_factory=dict)
    
    # Store location configuration
    locations: LocationConfig = Field(default_factory=LocationConfig)


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
    knowledge_base: KnowledgeBase = Field(default_factory=KnowledgeBase)
    
    class Settings:
        name = "business_configs"  # MongoDB collection name
        indexes = [
            [("business_id", 1)],  # Unique index on business_id
        ]

