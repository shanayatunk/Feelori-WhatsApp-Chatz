# /app/jobs/abandoned_cart_job.py

"""
Abandoned Cart Recovery Eligibility Job (WhatsApp-only).

This job identifies abandoned carts created via WhatsApp product discovery flows.
It specifically targets conversations where products were shown through the
marketing workflow automation (Phase 4.2.C).

This job does NOT overlap with Shopify webhook-based abandoned cart recovery.
It only processes conversations that have:
- flow_context.step == "completed"
- flow_context.metadata.products_sent == True
- flow_context.metadata.abandoned_cart.status == "pending"

This ensures we only recover carts from WhatsApp-native product discovery,
not from web-based Shopify checkout flows.
"""

from datetime import datetime
from typing import List, Dict
import logging
from app.services.db_service import db_service

logger = logging.getLogger(__name__)


async def find_abandoned_cart_candidates(now: datetime) -> List[Dict]:
    """
    Find conversations eligible for abandoned cart recovery nudges.
    
    This function identifies WhatsApp-only abandoned cart candidates by:
    1. Querying conversations with completed marketing workflows
    2. Filtering for pending abandoned cart status
    3. Computing elapsed time since products were first shown
    4. Applying eligibility rules based on elapsed hours and nudge count
    
    Args:
        now: Current datetime for computing elapsed time
    
    Returns:
        List of dictionaries containing eligible conversation data:
        {
            "conversation_id": str,
            "external_user_id": str,
            "business_id": str,
            "abandoned_cart": dict  # Full abandoned_cart metadata
        }
    """
    # Query conversations with strict filters to avoid Shopify overlap
    query = {
        "flow_context.step": "completed",
        "flow_context.metadata.products_sent": True,
        "flow_context.metadata.abandoned_cart.status": "pending"
    }
    
    # Fetch all matching conversations
    conversations = await db_service.db.conversations.find(query).to_list(length=None)
    
    total_scanned = len(conversations)
    eligible_candidates = []
    
    for conversation in conversations:
        try:
            flow_context = conversation.get("flow_context", {})
            metadata = flow_context.get("metadata", {})
            abandoned_cart = metadata.get("abandoned_cart", {})
            
            # Parse first_shown_at timestamp
            first_shown_at_str = abandoned_cart.get("first_shown_at")
            if not first_shown_at_str:
                logger.warning(
                    f"Conversation {conversation.get('_id')} has abandoned_cart "
                    "but missing first_shown_at timestamp. Skipping."
                )
                continue
            
            # Parse ISO string to datetime
            try:
                # Normalize Z suffix to +00:00 for fromisoformat
                normalized_str = first_shown_at_str.replace('Z', '+00:00')
                first_shown_at = datetime.fromisoformat(normalized_str)
                
                # Convert to timezone-naive for comparison
                if first_shown_at.tzinfo is not None:
                    first_shown_at = first_shown_at.replace(tzinfo=None)
                
                # Ensure now is also timezone-naive for comparison
                now_naive = now.replace(tzinfo=None) if now.tzinfo else now
            except (ValueError, AttributeError) as e:
                logger.warning(
                    f"Conversation {conversation.get('_id')} has invalid "
                    f"first_shown_at format: {first_shown_at_str}. Error: {e}. Skipping."
                )
                continue
            
            # Compute elapsed hours
            elapsed = now_naive - first_shown_at
            elapsed_hours = elapsed.total_seconds() / 3600
            
            # Get nudge count
            nudge_count = abandoned_cart.get("nudge_count", 0)
            
            # Eligibility rules (v1)
            is_eligible = False
            
            # Eligible for first nudge if: elapsed >= 24 hours AND nudge_count == 0
            if elapsed_hours >= 24 and nudge_count == 0:
                is_eligible = True
            
            # Eligible for second nudge if: elapsed >= 72 hours AND nudge_count == 1
            elif elapsed_hours >= 72 and nudge_count == 1:
                is_eligible = True
            
            if is_eligible:
                eligible_candidates.append({
                    "conversation_id": str(conversation.get("_id")),
                    "external_user_id": conversation.get("external_user_id"),
                    "business_id": conversation.get("tenant_id") or conversation.get("business_id", "feelori"),
                    "abandoned_cart": abandoned_cart
                })
        
        except Exception as e:
            logger.error(
                f"Error processing conversation {conversation.get('_id')}: {e}",
                exc_info=True
            )
            continue
    
    # Log results
    logger.info(
        f"Abandoned cart scan complete: {total_scanned} conversations scanned, "
        f"{len(eligible_candidates)} eligible candidates found"
    )
    
    return eligible_candidates

