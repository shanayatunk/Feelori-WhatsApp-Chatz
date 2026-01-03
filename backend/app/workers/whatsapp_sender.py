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
        phone_id: The WhatsApp Phone ID
        access_token: The System User Access Token
        recipient: The recipient's phone number
        message_text: The text body to send
        
    Returns:
        Tuple[success (bool), error_message (Optional[str])]
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
        "text": {"body": message_text}
    }
    
    try:
        # Use httpx for sync HTTP requests (robust standard lib alternative)
        with httpx.Client(timeout=10.0) as client:
            response = client.post(url, headers=headers, json=payload)
            
        if response.status_code in (200, 201):
            return True, None
        else:
            # Capture detailed error
            error_data = response.json()
            return False, f"API Error {response.status_code}: {error_data}"
            
    except Exception as e:
        return False, f"Network Exception: {str(e)}"


def process_pending_message(db, phone_id: str, access_token: str) -> bool:
    """
    Fetch and process one pending message.
    Returns True if a message was processed, False otherwise.
    """
    # 1. Atomically fetch and lock a pending message
    # We look for status="pending" AND attempts < 5 (Retry Limit)
    message = db.outbound_messages.find_one_and_update(
        {
            "status": "pending",
            "attempts": {"$lt": 5}
        },
        {
            "$set": {
                "status": "processing",
                "last_attempt_at": datetime.now(timezone.utc)
            },
            "$inc": {"attempts": 1}
        },
        return_document=True
    )
    
    if not message:
        return False

    # 2. Extract Data
    recipient = message.get("recipient")
    payload = message.get("payload", {})
    text_body = payload.get("text", "")
    
    if not recipient or not text_body:
        # Invalid data, mark as failed immediately
        db.outbound_messages.update_one(
            {"_id": message["_id"]},
            {
                "$set": {
                    "status": "failed",
                    "last_error": "Missing recipient or text",
                    "updated_at": datetime.now(timezone.utc)
                }
            }
        )
        return True

    logger.info(f"Sending message to {recipient} (Attempt {message.get('attempts')})")

    # 3. Send to WhatsApp
    success, error = send_whatsapp_message(phone_id, access_token, recipient, text_body)

    # 4. Update Status based on result
    now = datetime.now(timezone.utc)
    if success:
        db.outbound_messages.update_one(
            {"_id": message["_id"]},
            {
                "$set": {
                    "status": "sent",
                    "sent_at": now,
                    "updated_at": now
                }
            }
        )
        logger.info(f"✓ Message sent to {recipient}")
    else:
        db.outbound_messages.update_one(
            {"_id": message["_id"]},
            {
                "$set": {
                    "status": "pending", # Retry logic: set back to pending
                    "last_error": error,
                    "updated_at": now
                }
            }
        )
        logger.warning(f"✗ Failed to send to {recipient}: {error}")
        
    return True


def run_worker():
    """Main worker loop."""
    if not WHATSAPP_ACCESS_TOKEN or not WHATSAPP_PHONE_ID:
        logger.error("WHATSAPP_ACCESS_TOKEN and WHATSAPP_PHONE_ID environment variables are required")
        return
    
    if not MONGO_URI:
        logger.error("MONGO_ATLAS_URI or MONGODB_URI environment variable is required")
        return
    
    logger.info("Starting WhatsApp Sender Worker...")
    logger.info(f"Phone ID: {WHATSAPP_PHONE_ID[:4]}...")
    logger.info(f"API Version: {WHATSAPP_API_VERSION}")
    
    # Connect to MongoDB
    client = None
    try:
        client = get_mongo_client()
        db = client.get_default_database()
        logger.info(f"Connected to MongoDB database: {db.name}")
        
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
                
    except Exception as e:
         logger.error(f"Failed to connect to MongoDB or fatal error: {e}")
    finally:
        # Cleanup runs correctly here because it follows the try block
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    run_worker()
