# /app/utils/tasks.py

import logging
from app.services.visual_search_service import visual_matcher
from app.services.shopify_service import shopify_service
# Add the new imports needed for the analytics task
from app.services.db_service import db_service
from app.config import strings
from app.services import whatsapp_service

logger = logging.getLogger(__name__)

async def refresh_visual_search_index():
    """
    A scheduled task to periodically rebuild the visual search index by
    fetching all products from Shopify and re-indexing their images.
    """
    logger.info("--- Starting scheduled visual search index refresh ---")
    try:
        # The logic to index all products is in your VisualProductMatcher.
        # It handles fetching products from Shopify internally.
        await visual_matcher.index_all_products()
        logger.info("--- Successfully completed scheduled visual search index refresh ---")
    except Exception as e:
        # Using exc_info=True will log the full traceback for better debugging.
        logger.error("An error occurred during the scheduled index refresh.", exc_info=True)


# This is the new function you need to add
async def update_escalation_analytics():
    """
    Runs the heavy aggregation for human escalation requests and saves the
    results to a separate collection for fast dashboard lookups.
    """
    logger.info("--- Starting human escalation analytics update ---")
    try:
        pipeline = [
            {"$unwind": "$conversation_history"},
            {"$match": {"conversation_history.response": strings.HUMAN_ESCALATION}},
            {"$sort": {"conversation_history.timestamp": -1}},
            {
                "$group": {
                    "_id": "$_id",
                    "name": {"$first": "$name"},
                    "phone_number": {"$first": "$phone_number"},
                    "latest_escalation_time": {"$first": "$conversation_history.timestamp"}
                }
            },
            {"$sort": {"latest_escalation_time": -1}},
            {
                "$project": {
                    "_id": 1,
                    "name": 1,
                    "phone_number": 1,
                    "timestamp": "$latest_escalation_time"
                }
            },
            # This final stage writes the results to a new collection, overwriting it each time.
            {"$out": "human_escalation_analytics"}
        ]
        # Execute the aggregation. The results are written directly to the new collection by MongoDB.
        await db_service.db.customers.aggregate(pipeline).to_list(length=None)
        logger.info("--- Successfully updated human escalation analytics collection ---")
    except Exception as e:
        logger.error("An error occurred during the escalation analytics update.", exc_info=True)

async def process_abandoned_checkouts():
    """
    Finds abandoned checkouts that are ~1 hour old and sends a reminder.
    """
    logger.info("--- Checking for abandoned checkouts to process ---")
    try:
        checkouts = await db_service.get_pending_abandoned_checkouts()
        if not checkouts:
            logger.info("--- No abandoned checkouts to process at this time. ---")
            return

        for checkout in checkouts:
            phone_number = checkout.get("phone") or (checkout.get("shipping_address") or {}).get("phone")
            if not phone_number:
                await db_service.mark_reminder_as_sent(checkout['id']) # Mark as done to avoid re-checking
                continue

            customer_name = checkout.get("customer", {}).get("first_name", "there")
            message = f"Hi {customer_name}! 👋\nIt looks like you left some beautiful items in your cart. ✨\nComplete your purchase here:\n{checkout['abandoned_checkout_url']}"
            
            await whatsapp_service.send_message(phone_number, message)
            await db_service.mark_reminder_as_sent(checkout['id'])
            logger.info(f"Sent abandoned checkout reminder to {phone_number} for checkout {checkout['id']}")
        
        logger.info(f"--- Finished abandoned checkout task. Processed {len(checkouts)} reminders. ---")
    except Exception as e:
        logger.error("Error processing abandoned checkouts", exc_info=True)