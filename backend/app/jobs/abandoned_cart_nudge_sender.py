# /app/jobs/abandoned_cart_nudge_sender.py

"""
WhatsApp Abandoned Cart Nudge Sender (Phase 4.3.C).

This module orchestrates abandoned cart recovery nudges for WhatsApp-originated carts.
It sends deterministic reminder messages to users who viewed products but haven't
completed a purchase.

This job:
- Fetches eligible candidates from Phase 4.3.B
- Sends WhatsApp messages based on nudge count
- Updates database with optimistic locking
- Tracks nudge history and completion status
"""

from datetime import datetime
import logging
from bson import ObjectId
from app.services.db_service import db_service
from app.services.whatsapp_service import whatsapp_service
from app.services.string_service import string_service
from app.jobs.abandoned_cart_job import find_abandoned_cart_candidates

logger = logging.getLogger(__name__)


async def send_abandoned_cart_nudges(now: datetime):
    """
    Orchestrates abandoned cart recovery nudges for WhatsApp-originated carts.
    
    This function:
    - Fetches eligible candidates (Phase 4.3.B)
    - Sends deterministic WhatsApp messages
    - Updates DB with optimistic locking
    
    Args:
        now: Current datetime for timestamping nudges
    """
    # Fetch eligible candidates
    candidates = await find_abandoned_cart_candidates(now)
    
    if not candidates:
        logger.info("No eligible abandoned cart candidates found.")
        return
    
    logger.info(f"Processing {len(candidates)} abandoned cart nudge candidates")
    
    for candidate in candidates:
        try:
            conversation_id = candidate.get("conversation_id")
            external_user_id = candidate.get("external_user_id")
            business_id = candidate.get("business_id", "feelori")
            abandoned_cart = candidate.get("abandoned_cart", {})
            
            if not conversation_id or not external_user_id:
                logger.warning(f"Skipping candidate with missing conversation_id or external_user_id: {candidate}")
                continue
            
            current_nudge_count = abandoned_cart.get("nudge_count", 0)
            
            # Determine message (deterministic, no AI)
            if current_nudge_count == 0:
                message = string_service.get_formatted_string(
                    "ABANDONED_CART_NUDGE_1",
                    business_id=business_id
                )
                if not message or "ABANDONED_CART_NUDGE_1" in message:
                    message = (
                        "Hey 👋 Just checking in — the pieces you liked are still available.\n"
                        "Would you like me to show them again?"
                    )
            elif current_nudge_count == 1:
                message = string_service.get_formatted_string(
                    "ABANDONED_CART_NUDGE_2",
                    business_id=business_id
                )
                if not message or "ABANDONED_CART_NUDGE_2" in message:
                    message = (
                        "Last reminder 😊 The items you viewed are still in stock.\n"
                        "Reply YES to see them again."
                    )
            else:
                # Safety guard: skip if nudge_count is unexpected
                logger.warning(
                    f"Skipping candidate {conversation_id}: unexpected nudge_count {current_nudge_count}"
                )
                continue
            
            # Send WhatsApp message
            await whatsapp_service.send_message(
                external_user_id,
                message,
                business_id=business_id
            )
            
            # Perform atomic DB update with optimistic locking
            new_nudge_count = current_nudge_count + 1
            new_status = "completed" if new_nudge_count >= 2 else "pending"
            
            # Convert conversation_id string back to ObjectId for MongoDB query
            try:
                conversation_object_id = ObjectId(conversation_id)
            except Exception as e:
                logger.error(f"Invalid conversation_id format: {conversation_id}. Error: {e}")
                continue
            
            # Build update query with optimistic locking
            update_result = await db_service.db.conversations.update_one(
                {
                    "_id": conversation_object_id,
                    "flow_context.metadata.abandoned_cart.nudge_count": current_nudge_count
                },
                {
                    "$set": {
                        "flow_context.metadata.abandoned_cart.nudge_count": new_nudge_count,
                        "flow_context.metadata.abandoned_cart.last_nudge_at": now.isoformat(),
                        "flow_context.metadata.abandoned_cart.status": new_status,
                        "flow_context.last_updated": now.isoformat()
                    }
                }
            )
            
            if update_result.matched_count == 0:
                logger.warning(
                    f"Optimistic lock failed for conversation {conversation_id}: "
                    f"nudge_count may have changed. Message was sent but DB not updated."
                )
            else:
                logger.info(
                    f"Sent abandoned cart nudge #{new_nudge_count} to {external_user_id} "
                    f"(conversation: {conversation_id})"
                )
        
        except Exception as e:
            # Fail safely: log and continue loop
            logger.error(
                f"Error processing abandoned cart nudge for candidate {candidate.get('conversation_id')}: {e}",
                exc_info=True
            )
            continue
    
    logger.info(f"Completed abandoned cart nudge processing for {len(candidates)} candidates")


if __name__ == "__main__":
    import asyncio
    asyncio.run(send_abandoned_cart_nudges(datetime.utcnow()))

