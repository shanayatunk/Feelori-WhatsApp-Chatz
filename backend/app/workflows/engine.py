# /app/workflows/engine.py

"""
Pure workflow execution engine.

This module provides deterministic workflow state management that:
- Validates proposed workflow changes using validator functions
- Enforces strict slot requirements before step advancement
- Merges slot updates safely into flow_context
- Advances steps only when all validations pass
- Updates version and timestamps

All functions are:
- Pure (no side effects)
- Deterministic (same input = same output)
- No database writes
- No AI calls
- No message sending
- No logging
- Pure business logic only
"""

from typing import Dict, Optional, Any, TypedDict
from datetime import datetime
from app.models.flow import FlowContext
from app.models.conversation import Conversation
from app.workflows.validator import (
    validate_intent,
    validate_step,
    validate_allowed_action,
    validate_required_slots
)
from app.workflows.definitions import WORKFLOWS


class EngineResult(TypedDict):
    """Result of workflow engine execution."""
    applied: bool
    reason: Optional[str]
    updated_flow_context: Optional[FlowContext]


def apply_workflow_proposal(
    conversation: Conversation,
    proposed_workflow: Dict[str, Any]
) -> EngineResult:
    """
    Apply a proposed workflow change to a conversation's flow_context.
    
    This function:
    1. Validates the proposed workflow using validator functions
    2. Enforces strict slot requirements (does not advance if slots missing)
    3. Merges slot_updates into flow_context.slots
    4. Advances to next_step only if all validations pass
    5. Increments version and updates last_updated
    
    Args:
        conversation: The Conversation object with current flow_context
        proposed_workflow: Dictionary with intent, step, allowed_next_actions, slot_updates
        
    Returns:
        EngineResult with applied=True if changes were applied, False otherwise
    """
    # Extract proposed values
    proposed_intent = proposed_workflow.get("intent")
    proposed_step = proposed_workflow.get("step")
    proposed_allowed_actions = proposed_workflow.get("allowed_next_actions", [])
    proposed_slot_updates = proposed_workflow.get("slot_updates", {})
    
    # Get or initialize flow_context
    if conversation.flow_context:
        current_flow_context = conversation.flow_context
    else:
        # Initialize new flow_context
        current_flow_context = FlowContext(
            intent=None,
            step=None,
            allowed_next_actions=[],
            slots={},
            version=1,
            last_updated=datetime.utcnow()
        )
    
    # Start with current state
    new_intent = current_flow_context.intent
    new_step = current_flow_context.step
    new_allowed_actions = current_flow_context.allowed_next_actions.copy()
    new_slots = current_flow_context.slots.copy()
    new_version = current_flow_context.version
    new_last_updated = current_flow_context.last_updated
    
    # If intent is proposed, validate and set it
    if proposed_intent is not None:
        intent_validation = validate_intent(proposed_intent)
        if not intent_validation["is_valid"]:
            return {
                "applied": False,
                "reason": intent_validation["message"],
                "updated_flow_context": None
            }
        new_intent = proposed_intent
        # When intent changes, reset step to initial_step
        workflow = WORKFLOWS[new_intent]
        new_step = workflow.get("initial_step")
        # Update allowed_actions from initial step
        if new_step:
            step_def = workflow["steps"][new_step]
            new_allowed_actions = step_def.get("allowed_next_actions", []).copy()
    
    # If step is proposed, validate it
    if proposed_step is not None:
        if new_intent is None:
            return {
                "applied": False,
                "reason": "Cannot set step without intent",
                "updated_flow_context": None
            }
        
        step_validation = validate_step(new_intent, proposed_step)
        if not step_validation["is_valid"]:
            return {
                "applied": False,
                "reason": step_validation["message"],
                "updated_flow_context": None
            }
        new_step = proposed_step
        # Update allowed_actions from new step
        workflow = WORKFLOWS[new_intent]
        step_def = workflow["steps"][new_step]
        new_allowed_actions = step_def.get("allowed_next_actions", []).copy()
    
    # If allowed_next_actions are proposed, validate them
    if proposed_allowed_actions:
        if new_intent is None or new_step is None:
            return {
                "applied": False,
                "reason": "Cannot set allowed_next_actions without intent and step",
                "updated_flow_context": None
            }
        
        # Validate each proposed action
        for action in proposed_allowed_actions:
            action_validation = validate_allowed_action(new_intent, new_step, action)
            if not action_validation["is_valid"]:
                return {
                    "applied": False,
                    "reason": action_validation["message"],
                    "updated_flow_context": None
                }
        new_allowed_actions = proposed_allowed_actions.copy()
    
    # Merge slot_updates into current slots (safe merge)
    if proposed_slot_updates:
        if new_intent is None or new_step is None:
            return {
                "applied": False,
                "reason": "Cannot update slots without intent and step",
                "updated_flow_context": None
            }
        
        # Merge slot updates (proposed values override existing)
        new_slots.update(proposed_slot_updates)
    
    # STRICT ENFORCEMENT: If a step is explicitly proposed, validate required slots
    # Do NOT advance step if required slots are missing
    if proposed_step is not None and new_intent and new_step:
        slots_validation = validate_required_slots(new_intent, new_step, new_slots)
        if not slots_validation["is_valid"]:
            # Cannot advance to proposed step without required slots
            return {
                "applied": False,
                "reason": slots_validation["message"],
                "updated_flow_context": None
            }
    
    # Auto-advance to next_step ONLY if:
    # 1. We're in a valid step
    # 2. All required slots for current step are filled
    # 3. next_step exists and is not None
    # 4. No explicit step was proposed (auto-advance only)
    if (new_intent and new_step and proposed_step is None):
        workflow = WORKFLOWS[new_intent]
        step_def = workflow["steps"][new_step]
        
        # Validate required slots for current step
        slots_validation = validate_required_slots(new_intent, new_step, new_slots)
        if slots_validation["is_valid"]:
            # All required slots are filled, check for next_step
            next_step = step_def.get("next_step")
            if next_step is not None:
                # Safe to advance to next_step
                new_step = next_step
                # Update allowed_actions from new step
                step_def = workflow["steps"][new_step]
                new_allowed_actions = step_def.get("allowed_next_actions", []).copy()
    
    # Create updated flow_context
    updated_flow_context = FlowContext(
        intent=new_intent,
        step=new_step,
        allowed_next_actions=new_allowed_actions,
        slots=new_slots,
        version=new_version + 1,  # Increment version
        last_updated=datetime.utcnow()  # Update timestamp
    )
    
    # Check if anything actually changed
    if (current_flow_context.intent == updated_flow_context.intent and
        current_flow_context.step == updated_flow_context.step and
        current_flow_context.allowed_next_actions == updated_flow_context.allowed_next_actions and
        current_flow_context.slots == updated_flow_context.slots):
        # No actual changes, but version was incremented
        # Return applied=False since no meaningful change occurred
        return {
            "applied": False,
            "reason": "No changes to apply",
            "updated_flow_context": None
        }
    
    return {
        "applied": True,
        "reason": None,
        "updated_flow_context": updated_flow_context
    }

