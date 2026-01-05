# /app/workflows/validator.py

"""
Pure validation functions for workflow definitions.

This module provides deterministic, side-effect-free validation functions
that check workflow structure, steps, actions, and slots against the
WORKFLOWS definitions.

All functions are:
- Pure (no side effects)
- Deterministic (same input = same output)
- Unit-testable (no external dependencies)
- No database access
- No AI calls
- No logging
- No state mutation
"""

from typing import Dict, Optional, Any, TypedDict
from app.workflows.definitions import WORKFLOWS


class ValidationResult(TypedDict):
    """Result of a validation check."""
    is_valid: bool
    error_code: Optional[str]
    message: Optional[str]


def validate_intent(intent: str) -> ValidationResult:
    """
    Validate that an intent exists in the WORKFLOWS definitions.
    
    Args:
        intent: The intent name to validate
        
    Returns:
        ValidationResult with is_valid=True if intent exists, False otherwise
    """
    if not intent:
        return {
            "is_valid": False,
            "error_code": "EMPTY_INTENT",
            "message": "Intent cannot be empty"
        }
    
    if intent not in WORKFLOWS:
        return {
            "is_valid": False,
            "error_code": "UNKNOWN_INTENT",
            "message": f"Intent '{intent}' is not defined in WORKFLOWS"
        }
    
    return {
        "is_valid": True,
        "error_code": None,
        "message": None
    }


def validate_step(intent: str, step: str) -> ValidationResult:
    """
    Validate that a step exists for the given intent.
    
    Args:
        intent: The intent name
        step: The step name to validate
        
    Returns:
        ValidationResult with is_valid=True if step exists for intent, False otherwise
    """
    # First validate the intent
    intent_result = validate_intent(intent)
    if not intent_result["is_valid"]:
        return intent_result
    
    if not step:
        return {
            "is_valid": False,
            "error_code": "EMPTY_STEP",
            "message": "Step cannot be empty"
        }
    
    workflow = WORKFLOWS[intent]
    steps = workflow.get("steps", {})
    
    if step not in steps:
        return {
            "is_valid": False,
            "error_code": "UNKNOWN_STEP",
            "message": f"Step '{step}' is not defined for intent '{intent}'"
        }
    
    return {
        "is_valid": True,
        "error_code": None,
        "message": None
    }


def validate_allowed_action(intent: str, step: str, action: str) -> ValidationResult:
    """
    Validate that an action is allowed in the given step of the intent.
    
    Args:
        intent: The intent name
        step: The step name
        action: The action name to validate
        
    Returns:
        ValidationResult with is_valid=True if action is allowed, False otherwise
    """
    # First validate the step
    step_result = validate_step(intent, step)
    if not step_result["is_valid"]:
        return step_result
    
    if not action:
        return {
            "is_valid": False,
            "error_code": "EMPTY_ACTION",
            "message": "Action cannot be empty"
        }
    
    workflow = WORKFLOWS[intent]
    step_def = workflow["steps"][step]
    allowed_actions = step_def.get("allowed_next_actions", [])
    
    if action not in allowed_actions:
        return {
            "is_valid": False,
            "error_code": "ACTION_NOT_ALLOWED",
            "message": f"Action '{action}' is not allowed in step '{step}' of intent '{intent}'. Allowed actions: {allowed_actions}"
        }
    
    return {
        "is_valid": True,
        "error_code": None,
        "message": None
    }


def validate_required_slots(intent: str, step: str, slots: Dict[str, Any]) -> ValidationResult:
    """
    Validate that all required slots for a step are present and non-empty.
    
    Args:
        intent: The intent name
        step: The step name
        slots: Dictionary of slot values to validate
        
    Returns:
        ValidationResult with is_valid=True if all required slots are present, False otherwise
    """
    # First validate the step
    step_result = validate_step(intent, step)
    if not step_result["is_valid"]:
        return step_result
    
    if slots is None:
        slots = {}
    
    workflow = WORKFLOWS[intent]
    step_def = workflow["steps"][step]
    required_slots = step_def.get("required_slots", [])
    
    # Check if all required slots are present
    missing_slots = []
    for slot_name in required_slots:
        if slot_name not in slots:
            missing_slots.append(slot_name)
        elif slots[slot_name] is None:
            missing_slots.append(slot_name)
        elif isinstance(slots[slot_name], str) and not slots[slot_name].strip():
            missing_slots.append(slot_name)
        elif isinstance(slots[slot_name], list) and len(slots[slot_name]) == 0:
            missing_slots.append(slot_name)
        elif isinstance(slots[slot_name], dict) and len(slots[slot_name]) == 0:
            missing_slots.append(slot_name)
    
    if missing_slots:
        return {
            "is_valid": False,
            "error_code": "MISSING_REQUIRED_SLOTS",
            "message": f"Missing or empty required slots for step '{step}' of intent '{intent}': {', '.join(missing_slots)}"
        }
    
    return {
        "is_valid": True,
        "error_code": None,
        "message": None
    }

