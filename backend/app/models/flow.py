# /app/models/flow.py

from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class FlowContext(BaseModel):
    """
    Workflow-oriented context model for tracking conversation flow state.
    
    This is a PURE DATA model with no methods, logic, or AI imports.
    Used to track intent, steps, allowed actions, and slots during conversation flows.
    """
    intent: Optional[str] = Field(default=None, description="Current intent identifier")
    step: Optional[str] = Field(default=None, description="Current step in the flow")
    allowed_next_actions: List[str] = Field(default_factory=list, description="List of allowed next actions")
    slots: Dict[str, Any] = Field(default_factory=dict, description="Slot values for the current flow")
    version: int = Field(default=1, description="Flow context version")
    last_updated: datetime = Field(default_factory=datetime.utcnow, description="Timestamp of last update")

