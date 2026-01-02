#!/usr/bin/env python3
"""
WhatsApp Sender Worker

Reliably delivers outbound messages from the outbound_messages collection
to the WhatsApp Cloud API. Uses atomic find_one_and_update to prevent
duplicate processing.
"""

import os
import time
import logging
import httpx
from datetime import datetime, timezone
from typing import Optional, Tuple
from pymongo import MongoClient
from pymongo.errors import PyMongoError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
)
logger = logging.getLogger("WhatsAppSender")

# Load environment variables
WHATSAPP_ACCESS_TOKEN = os.getenv("WHATSAPP_ACCESS_TOKEN")
WHATSAPP_PHONE_ID = os.getenv("WHATSAPP_PHONE_ID")
MONGO_URI = os.getenv("MONGO_ATLAS_URI") or os.getenv("MONGODB_URI")

# WhatsApp API configuration
WHATSAPP_API_VERSION = "v21.0"
WHATSAPP_BASE_URL = f"https://graph.facebook.com/{WHATSAPP_API_VERSION}"


def get_mongo_client() -> MongoClient:
    """Create and return a MongoDB client."""
    if not MONGO_URI:
        raise ValueError("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
    
    return MongoClient(MONGO_URI)


def send_whatsapp_message(phone_id: str, access_token: str, recipient: str, message_text: str) -> Tuple[bool, Optional[str]]:
    """
    Send a WhatsApp message via the Meta Cloud API.
    
    Args:
        phone_id: WhatsApp Phone Number ID
        access_token: WhatsApp Access Token
        recipient: Recipient phone number
        message_text: Message text content
        
    Returns:
        Tuple of (success: bool, error_message: Optional[str])
    """
    url = f"{WHATSAPP_BASE_URL}/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "messaging_product": "whatsapp",
        "to": recipient,
        "type": "text",
        "text": {
            "body": message_text
        }
    }
    
    try:
        with httpx.Client(timeout=15.0) as client:
            response = client.post(url, json=payload, headers=headers)
            
            if response.status_code == 200:
                logger.info(f"Message sent successfully to {recipient[:4]}...")
                return True, None
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                logger.error(f"Failed to send message to {recipient[:4]}...: {error_msg}")
                return False, error_msg
                
    except httpx.RequestError as e:
        error_msg = f"Request error: {str(e)}"
        logger.error(f"Network error sending to {recipient[:4]}...: {error_msg}")
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error: {str(e)}"
        logger.error(f"Unexpected error sending to {recipient[:4]}...: {error_msg}")
        return False, error_msg


def process_pending_message(db, phone_id: str, access_token: str) -> bool:
    """
    Process a single pending message from the outbound_messages collection.
    
    Args:
        db: MongoDB database object
        phone_id: WhatsApp Phone Number ID
        access_token: WhatsApp Access Token
        
    Returns:
        True if a message was processed, False if no message was found
    """
    try:
        # Atomically claim a pending message
        message = db.outbound_messages.find_one_and_update(
            {
                "status": "pending",
                "attempts": {"$lt": 5}
            },
            {
                "$set": {"status": "processing"},
                "$inc": {"attempts": 1}
            },
            sort=[("created_at", 1)]  # Process oldest first
        )
        
        if not message:
            return False
        
        message_id = str(message.get("_id", ""))
        recipient = message.get("recipient", "")
        payload = message.get("payload", {})
        message_text = payload.get("text", "")
        tenant_id = message.get("tenant_id", "")
        
        # Note: find_one_and_update returns document BEFORE update, so attempts is the old value
        # After the update, attempts will be incremented by 1
        old_attempts = message.get("attempts", 0)
        current_attempt = old_attempts + 1
        
        logger.info(f"Processing message {message_id} to {recipient[:4]}... (attempt {current_attempt})")
        
        # Send the message
        success, error_msg = send_whatsapp_message(phone_id, access_token, recipient, message_text)
        
        now = datetime.now(timezone.utc)
        
        if success:
            # Update document on success
            db.outbound_messages.update_one(
                {"_id": message["_id"]},
                {
                    "$set": {
                        "status": "sent",
                        "sent_at": now
                    }
                }
            )
            logger.info(f"Message {message_id} marked as sent")
        else:
            # Update document on failure - set back to pending for retry
            # Note: If attempts >= 5, the initial query won't pick it up (effectively dead)
            db.outbound_messages.update_one(
                {"_id": message["_id"]},
                {
                    "$set": {
                        "status": "pending",
                        "last_error": error_msg
                    }
                }
            )
            logger.warning(f"Message {message_id} failed, will retry (attempts: {current_attempt})")
        
        return True
        
    except PyMongoError as e:
        logger.error(f"MongoDB error processing message: {e}", exc_info=True)
        return False
    except Exception as e:
        logger.error(f"Unexpected error processing message: {e}", exc_info=True)
        return False


def run_worker():
    """
    Main worker loop that continuously polls for pending messages and sends them.
    """
    # Validate environment variables
    if not WHATSAPP_ACCESS_TOKEN:
        logger.error("WHATSAPP_ACCESS_TOKEN environment variable is required")
        return
    
    if not WHATSAPP_PHONE_ID:
        logger.error("WHATSAPP_PHONE_ID environment variable is required")
        return
    
    if not MONGO_URI:
        logger.error("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
        return
    
    logger.info("Starting WhatsApp Sender Worker...")
    logger.info(f"Phone ID: {WHATSAPP_PHONE_ID[:4]}...")
    logger.info(f"API Version: {WHATSAPP_API_VERSION}")
    
    # Connect to MongoDB
    try:
        client = get_mongo_client()
        db = client.get_default_database()
        logger.info(f"Connected to MongoDB database: {db.name}")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        return
    
    # Main polling loop
    while True:
        try:
            # Try to process a pending message
            processed = process_pending_message(db, WHATSAPP_PHONE_ID, WHATSAPP_ACCESS_TOKEN)
            
            if not processed:
                # No message found, sleep before next poll
                time.sleep(1)
            # If processed, immediately try next message (no sleep for throughput)
            
        except KeyboardInterrupt:
            logger.info("Worker stopped by user")
            break
        except Exception as e:
            logger.error(f"Unexpected error in worker loop: {e}", exc_info=True)
            time.sleep(1)  # Sleep on error to prevent tight error loop
    finally:
        if 'client' in locals():
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    run_worker()

