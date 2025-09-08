# /app/utils/tasks.py

import logging
from app.services.visual_search_service import visual_matcher
from app.services.shopify_service import shopify_service

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