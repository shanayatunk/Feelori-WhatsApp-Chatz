# /app/utils/tasks.py

import logging

from app.services.shopify_service import shopify_service
# Add the new imports needed for the analytics task
from app.services.db_service import db_service
from app.config import strings
from app.services import whatsapp_service
from app.services.whatsapp_service import whatsapp_service

logger = logging.getLogger(__name__)

async def refresh_visual_search_index():
    """
    A scheduled task to periodically rebuild the visual search index by
    fetching all products from Shopify and re-indexing their images.
    """
    # ADD THIS CHECK: First, check if the feature is enabled.
    if not settings.VISUAL_SEARCH_ENABLED:
        logger.info("Visual search is disabled, skipping index refresh.")
        return

    # MOVE THE IMPORT HERE: Only import the service if the feature is enabled.
    from app.services.visual_search_service import visual_matcher
    
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
    Finds abandoned checkouts that are ready for a reminder and sends a WhatsApp message.
    """
    logger.info("Scheduler starting: process_abandoned_checkouts job.")
    
    checkouts = await db_service.get_pending_abandoned_checkouts()
    if not checkouts:
        logger.info("No pending abandoned checkouts to process.")
        return

    logger.info(f"Found {len(checkouts)} abandoned checkouts to process.")
    
    for checkout in checkouts:
        try:
            checkout_id = checkout.get("id")
            customer = checkout.get("customer", {})
            
            phone = customer.get("phone") or (checkout.get("shipping_address") or {}).get("phone")
            if not phone:
                await db_service.mark_reminder_as_sent(checkout_id)
                continue

            customer_name = customer.get("first_name", "there")
            checkout_url = checkout.get("abandoned_checkout_url")
            line_items = checkout.get("line_items", [])
            
            first_item_image_url = None
            if line_items and line_items[0].get("variant", {}).get("image", {}).get("src"):
                 first_item_image_url = line_items[0]["variant"]["image"]["src"]

            # --- THIS IS THE FIX ---
            # Correctly parse the URL to include the unique checkout token
            button_param = ""
            if checkout_url and "checkouts/" in checkout_url:
                button_param = checkout_url.split("checkouts/", 1)[1]
            # --- END OF FIX ---

            await whatsapp_service.send_template_message(
                to=phone,
                template_name="abandoned_cart_reminder_v1",
                header_image_url=first_item_image_url,
                body_params=[customer_name],
                button_url_param=button_param
            )
            
            await db_service.mark_reminder_as_sent(checkout_id)
            logger.info(f"Successfully sent abandoned cart reminder for checkout ID {checkout_id}.")

        except Exception as e:
            logger.error(f"Failed to process abandoned checkout ID {checkout.get('id')}: {e}", exc_info=True)
            
    logger.info("Scheduler finished: process_abandoned_checkouts job.")