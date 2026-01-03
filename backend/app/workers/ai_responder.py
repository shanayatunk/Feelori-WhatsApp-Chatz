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
from pymongo import MongoClient

# Import AI service
from app.services.ai_service import ai_service

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
        
        response = await ai_service.generate_response(
            message=user_text,
            business_id=business_id,
            conversation_history=[] # We could pass history here in future
        )
        return response
    except Exception as e:
        logger.error(f"AI Generation failed: {e}")
        return "I apologize, but I'm having trouble processing your request right now."


async def find_and_process_candidates(db):
    """
    Find conversations needing AI reply and process them.
    Returns the number of processed conversations.
    """
    # Criteria:
    # 1. Status is Open
    # 2. AI is Enabled
    # 3. AI is NOT paused by anyone
    # 4. Last message exists and is text
    query = {
        "status": "open",
        "ai_enabled": True,
        "ai_paused_by": None,
        "last_message.type": "text"
    }
    
    # Find candidates
    # Limit to 5 to prevent massive batches blocking the loop
    candidates = list(db.conversations.find(query).limit(5))
    
    processed_count = 0
    
    for conv in candidates:
        conv_id = conv["_id"]
        tenant_id = conv["tenant_id"]
        
        # CRITICAL SAFETY CHECK:
        # Verify the LAST message in the log is actually from the 'user'.
        # The 'conversations' collection summary might be slightly out of sync or
        # we just want to be 100% sure we don't reply to ourselves or an agent.
        last_msg = db.message_logs.find_one(
            {"conversation_id": str(conv_id)}, # message_logs uses string ID in current schema? Checking...
            sort=[("created_at", -1)]
        )
        
        # Note: In Backend Step 2 we enforced ObjectId for conversation_id in message_logs.
        # But let's check both just in case of old data, or trust the new schema.
        if not last_msg:
            # Try with ObjectId just in case
            last_msg = db.message_logs.find_one(
                {"conversation_id": conv_id},
                sort=[("created_at", -1)]
            )
            
        if not last_msg:
            continue # Should not happen if last_message exists, but safe to skip
            
        if last_msg.get("source") != "user":
            continue # The last message was from 'agent' or 'ai'. Do NOT reply.
            
        # Process this conversation
        user_text = last_msg.get("text", "")
        if not user_text:
            continue
            
        logger.info(f"Generating AI reply for conversation {conv_id}")
        
        # Generate Response
        ai_reply_text = await generate_ai_response(user_text, tenant_id)
        
        now = datetime.now(timezone.utc)
        
        # 1. Insert into message_logs
        ai_msg_doc = {
            "tenant_id": tenant_id,
            "conversation_id": conv_id, # Keep ObjectId consistency
            "source": "ai",
            "type": "text",
            "text": ai_reply_text,
            "created_at": now
        }
        db.message_logs.insert_one(ai_msg_doc)
        
        # 2. Insert into outbound_messages (Ledger)
        outbound_doc = {
            "tenant_id": tenant_id,
            "conversation_id": conv_id,
            "channel": "whatsapp",
            "recipient": conv["external_user_id"],
            "payload": {
                "type": "text",
                "text": ai_reply_text
            },
            "source": "ai",
            "status": "pending",
            "attempts": 0,
            "created_at": now,
            "sent_at": None
        }
        db.outbound_messages.insert_one(outbound_doc)
        
        # 3. Update conversation summary
        db.conversations.update_one(
            {"_id": conv_id},
            {
                "$set": {
                    "last_message": {
                        "type": "text",
                        "text": ai_reply_text
                    },
                    "last_message_at": now,
                    "updated_at": now
                    }
            }
        )
        
        processed_count += 1
        
    return processed_count


async def run_worker_loop():
    """Async worker loop."""
    if not MONGO_URI:
        logger.error("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
        return
    
    logger.info("Starting AI Responder Worker...")
    
    # Connect to MongoDB
    client = None
    try:
        client = get_mongo_client()
        db = client.get_default_database()
        logger.info(f"Connected to MongoDB database: {db.name}")
        
        # Main polling loop
        while True:
            try:
                # Find and process candidates
                processed = await find_and_process_candidates(db)
                
                if processed > 0:
                    logger.info(f"Processed {processed} conversation(s) with AI responses")
                
                # Always sleep to prevent hammering the DB
                await asyncio.sleep(2)
                
            except asyncio.CancelledError:
                logger.info("Worker stopped")
                break
            except Exception as e:
                logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
                await asyncio.sleep(2)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


def run_worker():
    """
    Entry point for the worker (synchronous wrapper around async loop).
    """
    asyncio.run(run_worker_loop())


if __name__ == "__main__":
    run_worker()
