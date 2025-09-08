# /backend/ml_worker.py

import asyncio
import logging
import json
import sys
import redis.asyncio as aioredis
from app.services.visual_search_service import visual_matcher
from app.config.settings import settings

# Configure basic logging for the worker
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s')
logger = logging.getLogger("MLWorker")

# Define the names for our Redis queues
INCOMING_QUEUE = "visual_search:jobs"
RESULT_KEY_PREFIX = "visual_search:results:"
POLLING_TIMEOUT = 1  # Seconds to wait for a new job before checking again

async def main():
    """
    The main event loop for the ML worker.
    Initializes the model and continuously processes jobs from the Redis queue.
    """
    # This is the corrected placement for the feature flag check.
    if not settings.VISUAL_SEARCH_ENABLED:
        logger.warning("Visual search is disabled in settings. ML worker will not start.")
        sys.exit(0) # Exit cleanly

    logger.info("--- Starting ML Worker ---")
    
    # Initialize the visual search model once, a significant memory saving.
    logger.info("Initializing visual search model...")
    await visual_matcher._initialize_vision_model()
    logger.info("Visual search model initialized successfully.")

    # Connect to Redis
    redis = aioredis.from_url(settings.redis_url, decode_responses=True)
    logger.info(f"Connected to Redis. Listening for jobs on '{INCOMING_QUEUE}'...")

    try:
        while True:
            # BRPop is a blocking call that waits for an item on the queue
            # Note: A typo in the original file 'INKOMING_QUEUE' was corrected to 'INCOMING_QUEUE'
            job_data = await redis.brpop(INCOMING_QUEUE, timeout=POLLING_TIMEOUT)

            if not job_data:
                await asyncio.sleep(0.1) # Small sleep to prevent a tight loop if Redis is empty
                continue

            # job_data is a tuple: (queue_name, item_json_string)
            _ , item = job_data
            try:
                job = json.loads(item)
                job_id = job['job_id']
                image_b64 = job['image_b64']
                
                logger.info(f"Processing job {job_id}...")

                # Decode the image and find matches
                import base64
                image_bytes = base64.b64decode(image_b64)
                results = await visual_matcher.find_matching_products(image_bytes)

                # Store the result back in Redis with the unique job_id as the key
                result_key = f"{RESULT_KEY_PREFIX}{job_id}"
                await redis.set(result_key, json.dumps(results), ex=300) # Expire after 5 mins

                logger.info(f"Finished job {job_id}. Results stored in Redis.")
            
            except (json.JSONDecodeError, KeyError) as e:
                logger.error(f"Failed to parse job data: {item}. Error: {e}")
            except Exception as e:
                logger.error(f"An unexpected error occurred while processing job: {e}", exc_info=True)

    except asyncio.CancelledError:
        logger.info("ML Worker shutting down.")
    finally:
        await redis.close()
        logger.info("Redis connection closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("ML Worker stopped by user.")