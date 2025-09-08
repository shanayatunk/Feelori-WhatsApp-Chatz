# /app/utils/tasks.py

import logging
from app.services.visual_search_service import visual_matcher
from app.services.shopify_service import shopify_service
# Add the new imports needed for the analytics task
from app.services.db_service import db_service
from app.config import strings

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