import asyncio
import logging
from dotenv import load_dotenv
import redis.asyncio as redis

# Load environment variables
load_dotenv()

from app.config.settings import settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

STREAM_NAME = "webhook_messages"

async def clear_redis_stream():
    """
    Connects to Redis and deletes the specified stream key to clear the message queue.
    """
    # Construct the local Redis URL correctly.
    local_redis_url = settings.redis_url.replace("redis:6379", "localhost:6380")
    logger.info(f"Connecting to Redis at: {local_redis_url}")

    try:
        # The from_url method correctly parses the full URL string.
        redis_client = redis.from_url(local_redis_url)
        if await redis_client.ping():
            logger.info("Successfully connected to Redis.")
        else:
            logger.error("Could not ping Redis server.")
            return
    except Exception as e:
        logger.error(f"Failed to connect to Redis: {e}")
        logger.error("Please ensure your docker-compose containers are running.")
        return

    try:
        logger.info(f"Attempting to delete stream key: '{STREAM_NAME}'...")
        result = await redis_client.delete(STREAM_NAME)
        if result > 0:
            logger.info(f"Successfully deleted the stream. The queue has been cleared.")
        else:
            logger.info(f"Stream key '{STREAM_NAME}' did not exist. No action needed.")
    except Exception as e:
        logger.error(f"An error occurred while trying to delete the stream: {e}")
    finally:
        # FIX: Use the modern 'aclose()' for asynchronous connections.
        await redis_client.aclose()


if __name__ == "__main__":
    asyncio.run(clear_redis_stream())
