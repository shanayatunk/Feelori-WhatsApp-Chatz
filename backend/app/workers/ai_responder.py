#!/usr/bin/env python3
"""
AI Responder Worker

Safely generates and enqueues AI responses for conversations where AI is enabled
and the last message was from the user. Uses message_logs to verify message source.
"""

import os
import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional
from pymongo import MongoClient
from pymongo.errors import PyMongoError
from bson import ObjectId

# Import AI service
from app.services.ai_service import ai_service
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("AIResponder")

# Load environment variables
MONGO_URI = os.getenv("MONGO_ATLAS_URI") or os.getenv("MONGODB_URI")


def get_mongo_client() -> MongoClient:
    """Create and return a MongoDB client."""
    if not MONGO_URI:
        raise ValueError("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
    
    return MongoClient(MONGO_URI)


async def generate_ai_response(user_text: str, tenant_id: str) -> str:
    """
    Generate an AI response for the user message.
    
    Args:
        user_text: The user's message text
        tenant_id: Tenant ID to determine business context
        
    Returns:
        Generated AI response text
    """
    try:
        # Use business_id from tenant_id (default to "feelori")
        business_id = tenant_id if tenant_id in ["feelori", "goldencollections"] else "feelori"
        
        # Simple context (can be enhanced later)
        context = {}
        
        # Generate response using AI service
        response = await ai_service.generate_response(
            message=user_text,
            context=context,
            business_id=business_id
        )
        
        return response or f"Echo: {user_text}"  # Fallback to echo if AI fails
        
    except Exception as e:
        logger.error(f"Failed to generate AI response: {e}", exc_info=True)
        # Fallback to simple echo
        return f"Echo: {user_text}"


async def process_conversation_candidate(db, conversation: dict) -> bool:
    """
    Process a single conversation candidate that needs an AI response.
    
    Args:
        db: MongoDB database object
        conversation: Conversation document
        
    Returns:
        True if conversation was processed, False if skipped
    """
    try:
        conv_id = conversation.get("_id")
        conv_id_str = str(conv_id)
        tenant_id = conversation.get("tenant_id", "")
        last_message = conversation.get("last_message", {})
        user_text = last_message.get("text", "")
        
        if not user_text:
            logger.debug(f"Skipping conversation {conv_id_str}: no text in last_message")
            return False
        
        # CRITICAL SAFETY CHECK: Verify the last message in message_logs was from user
        # Find the most recent message for this conversation
        latest_log = db.message_logs.find_one(
            {"conversation_id": conv_id},  # Use ObjectId for referential integrity
            sort=[("created_at", -1)]
        )
        
        if not latest_log:
            logger.debug(f"Skipping conversation {conv_id_str}: no messages in message_logs")
            return False
        
        # Check if the latest message was from user/customer
        # CRITICAL: Only respond if the last message was from the user, not AI or agent
        latest_source = latest_log.get("source", "")
        if latest_source not in ["user", "customer"]:
            # AI or Agent already replied, skip
            logger.debug(f"Skipping conversation {conv_id_str}: last message source is '{latest_source}', not 'user'/'customer'")
            return False
        
        logger.info(f"Processing conversation {conv_id_str} for AI response (user: {latest_log.get('phone', 'unknown')[:4]}...)")
        
        # Generate AI response
        ai_response = await generate_ai_response(user_text, tenant_id)
        
        now = datetime.now(timezone.utc)
        
        # Step A: Insert into message_logs (source="ai")
        message_log_doc = {
            "tenant_id": tenant_id,
            "conversation_id": conv_id,  # Use ObjectId for referential integrity
            "source": "ai",
            "type": "text",
            "text": ai_response,
            "created_at": now
        }
        
        message_log_result = db.message_logs.insert_one(message_log_doc)
        message_log_id = str(message_log_result.inserted_id)
        
        # Step B: Insert into outbound_messages (source="ai", status="pending")
        outbound_doc = {
            "tenant_id": tenant_id,
            "conversation_id": conv_id,  # Use ObjectId for referential integrity
            "channel": "whatsapp",
            "recipient": conversation.get("external_user_id", ""),  # Phone number
            "payload": {
                "type": "text",
                "text": ai_response
            },
            "source": "ai",
            "status": "pending",  # Worker will pick this up
            "attempts": 0,
            "last_error": None,
            "created_at": now,
            "sent_at": None
        }
        
        db.outbound_messages.insert_one(outbound_doc)
        
        # Step C: Update conversations (last_message, last_message_at, updated_at)
        new_last_message = {
            "type": "text",
            "text": ai_response
        }
        
        db.conversations.update_one(
            {"_id": conv_id},
            {
                "$set": {
                    "last_message": new_last_message,
                    "last_message_at": now,
                    "updated_at": now
                }
            }
        )
        
        logger.info(f"AI response generated and enqueued for conversation {conv_id_str}")
        return True
        
    except PyMongoError as e:
        logger.error(f"MongoDB error processing conversation: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing conversation: {e}", exc_info=True)
        return False


async def find_and_process_candidates(db) -> int:
    """
    Find conversation candidates that need AI responses and process them.
    
    Args:
        db: MongoDB database object
        
    Returns:
        Number of conversations processed
    """
    try:
        # Find candidate conversations
        candidates = list(db.conversations.find({
            "ai_enabled": True,
            "ai_paused_by": None,
            "status": "open",
            "last_message.type": "text"  # Ignore images for now
        }))
        
        if not candidates:
            return 0
        
        logger.info(f"Found {len(candidates)} candidate conversations for AI response")
        
        processed_count = 0
        
        # Process each candidate
        for conversation in candidates:
            processed = await process_conversation_candidate(db, conversation)
            if processed:
                processed_count += 1
        
        return processed_count
        
    except PyMongoError as e:
        logger.error(f"MongoDB error finding candidates: {e}", exc_info=True)
        return 0
    except Exception as e:
        logger.error(f"Unexpected error finding candidates: {e}", exc_info=True)
        return 0


async def run_worker_loop():
    """
    Main async worker loop that continuously polls for conversations needing AI responses.
    """
    if not MONGO_URI:
        logger.error("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
        return
    
    logger.info("Starting AI Responder Worker...")
    
    # Connect to MongoDB
    try:
        client = get_mongo_client()
        db = client.get_default_database()
        logger.info(f"Connected to MongoDB database: {db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return
    
    # Main polling loop
    try:
        while True:
            try:
                # Find and process candidates
                processed = await find_and_process_candidates(db)
                
                if processed > 0:
                    logger.info(f"Processed {processed} conversation(s) with AI responses")
                # Always sleep to prevent hammering the DB
                await asyncio.sleep(2)
                
            except KeyboardInterrupt:
                logger.info("Worker stopped by user")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(2)  # Sleep on error to prevent tight error loop
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")


def run_worker():
    """
    Entry point for the worker (synchronous wrapper around async loop).
    """
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    run_worker()

