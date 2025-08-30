# /app/utils/queue.py

import json
import uuid
import asyncio
import logging
import redis as redis_package
from typing import Dict

from app.services.cache_service import cache_service
# We REMOVE the direct import of order_service from the top of the file

# This utility provides a Redis Streams-based message queue for processing
# incoming webhooks asynchronously, improving webhook response time and reliability.

logger = logging.getLogger(__name__)

class RedisMessageQueue:
    def __init__(self, redis_client, stream_name: str = "webhook_messages", max_workers: int = 5):
        self.redis = redis_client
        self.stream_name = stream_name
        self.consumer_group = "webhook_processors"
        self.max_workers = max_workers
        self.workers = []
        self.running = False

    async def initialize(self):
        if not self.redis: return
        try:
            await self.redis.xgroup_create(self.stream_name, self.consumer_group, id="0", mkstream=True)
        except redis_package.exceptions.ResponseError as e:
            if "BUSYGROUP" not in str(e): raise

    async def start_workers(self):
        if not self.redis: return
        await self.initialize()
        self.running = True
        for i in range(self.max_workers):
            self.workers.append(asyncio.create_task(self._worker(f"worker-{i}-{uuid.uuid4().hex[:4]}")))
        logger.info(f"Started {self.max_workers} Redis message queue workers.")

    async def stop_workers(self):
        self.running = False
        for worker in self.workers:
            worker.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)

    async def _worker(self, consumer_name: str):
        while self.running:
            try:
                messages = await self.redis.xreadgroup(self.consumer_group, consumer_name, {self.stream_name: ">"}, count=1, block=1000)
                if not messages: continue
                
                # Note: messages is a list of streams, e.g., [(b'stream_name', [(b'msg_id', {..})])]
                stream_name, stream_messages = messages[0]
                for message_id, fields in stream_messages:
                    try:
                        message_data = json.loads(fields[b'data'].decode())
                        # Call the processing logic
                        await self._process_message_from_queue(message_data)
                        # Acknowledge the message was processed
                        await self.redis.xack(stream_name, self.consumer_group, message_id)
                    except Exception as e:
                        logger.error(f"Error processing message {message_id.decode()}: {e}", exc_info=True)
            except Exception as e:
                if self.running: 
                    logger.error(f"Redis worker '{consumer_name}' error: {e}")
                    await asyncio.sleep(5)

    async def _process_message_from_queue(self, data: Dict):
        """
        Processes a single message consumed from the Redis queue.
        The import is moved here to break the circular dependency.
        """
        # --- THIS IS THE FIX ---
        # By importing here, the order_service module is guaranteed to be fully loaded
        # by the time this worker method is called.
        from app.services.order_service import process_message, get_or_create_customer, db_service
        from app.services.whatsapp_service import whatsapp_service
        from app.utils.metrics import message_counter

        from_number = data["from_number"]

        # Update customer name if provided
        if data.get("profile_name"):
            customer = await get_or_create_customer(from_number)
            if not customer.get("name"):
                await db_service.update_customer_name(from_number, data["profile_name"])
        
        # Get the response from the main message handler
        response_text = await process_message(
            phone_number=from_number,
            message_text=data["message_text"],
            message_type=data["message_type"],
            quoted_wamid=data.get("quoted_wamid")
        )

        # If there's a response to send or log, handle it
        if response_text:
            is_log_only = response_text.startswith('[') and response_text.endswith(']')
            wamid = None

            # Only send a message if it's not a log-only entry
            if not is_log_only:
                wamid = await whatsapp_service.send_message(from_number, response_text)
            
            # Always log the conversation turn
            await db_service.update_conversation_history(from_number, data["message_text"], response_text, wamid)
            
            if wamid:
                message_counter.labels(status="success", message_type=data["message_type"]).inc()
            elif not is_log_only:
                message_counter.labels(status="send_failed", message_type=data["message_type"]).inc()

    async def add_message(self, message_data: Dict):
        """Adds a new message to the Redis stream for a worker to process."""
        if self.redis:
            await self.redis.xadd(self.stream_name, {"data": json.dumps(message_data)})

    async def is_duplicate_message(self, message_id: str, phone_number: str) -> bool:
        """Checks for duplicate message IDs to prevent re-processing."""
        if not self.redis: return False
        # set with nx=True returns True if the key was set, False if it already existed.
        # We want to return True if it's a duplicate, so we invert the result.
        return not await self.redis.set(f"processed:{phone_number}:{message_id}", "1", ex=300, nx=True)

# Globally accessible instance
message_queue = RedisMessageQueue(cache_service.redis)