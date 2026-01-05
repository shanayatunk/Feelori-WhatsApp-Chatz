# /app/workflows/definitions.py

"""
Workflow definitions with strict slot enforcement.

This module defines workflow structures as pure data (no logic).
Each workflow specifies:
- initial_step: The starting step name
- steps: A dictionary mapping step names to step definitions

Each step defines:
- required_slots: List of slot names that must be filled before proceeding
- allowed_next_actions: List of action names allowed in this step
- next_step: The next step name (or None if this is a terminal step)
"""

from typing import Dict, Any

# Type definition for a step
StepDefinition = Dict[str, Any]

# Type definition for a workflow
WorkflowDefinition = Dict[str, Any]

WORKFLOWS: Dict[str, WorkflowDefinition] = {
    "return_request": {
        "initial_step": "collect_return_reason",
        "steps": {
            "collect_return_reason": {
                "required_slots": ["order_id", "return_reason"],
                "allowed_next_actions": ["provide_reason", "cancel_return"],
                "next_step": "collect_item_details"
            },
            "collect_item_details": {
                "required_slots": ["order_id", "return_reason", "item_sku", "quantity"],
                "allowed_next_actions": ["provide_item_details", "cancel_return"],
                "next_step": "confirm_return"
            },
            "confirm_return": {
                "required_slots": ["order_id", "return_reason", "item_sku", "quantity", "confirmation"],
                "allowed_next_actions": ["confirm", "cancel_return"],
                "next_step": None  # Terminal step
            }
        }
    },
    
    "order_status": {
        "initial_step": "request_order_id",
        "steps": {
            "request_order_id": {
                "required_slots": ["order_id"],
                "allowed_next_actions": ["provide_order_id", "cancel"],
                "next_step": "check_status"
            },
            "check_status": {
                "required_slots": ["order_id"],
                "allowed_next_actions": ["request_details", "request_tracking", "cancel"],
                "next_step": None  # Terminal step (status is provided)
            }
        }
    },
    
    "handoff": {
        "initial_step": "prepare_handoff",
        "steps": {
            "prepare_handoff": {
                "required_slots": ["handoff_reason", "conversation_summary"],
                "allowed_next_actions": ["provide_reason", "cancel_handoff"],
                "next_step": "confirm_handoff"
            },
            "confirm_handoff": {
                "required_slots": ["handoff_reason", "conversation_summary", "confirmation"],
                "allowed_next_actions": ["confirm", "cancel_handoff"],
                "next_step": None  # Terminal step
            }
        }
    }
}

