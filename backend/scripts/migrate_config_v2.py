#!/usr/bin/env python3
"""
Migration script to migrate hardcoded persona and rules configuration
from app/config/persona.py and app/config/rules.py into MongoDB.

This script:
1. Reads persona data from app/config/persona.py
2. Reads rules data from app/config/rules.py
3. Transforms them into BusinessConfig + RuleConfig structure
4. Inserts a document for business_id = "feelori"
5. Is idempotent (safe to run multiple times)

Usage:
    python scripts/migrate_config_v2.py
"""

import asyncio
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings
from app.config.persona import FEELORI_SYSTEM_PROMPT
from app.config.rules import INTENT_RULES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def transform_persona() -> Dict[str, Any]:
    """
    Transform FEELORI_SYSTEM_PROMPT into persona dict structure.
    """
    return {
        "name": "FeelOri Assistant",
        "tone": "friendly",
        "language": "en",
        "prompt": FEELORI_SYSTEM_PROMPT
    }


def transform_rules() -> List[Dict[str, Any]]:
    """
    Transform INTENT_RULES into RuleConfig structure.
    Each rule tuple is: (single_word_tokens_set, multi_word_phrases_list, intent_name)
    """
    rules = []
    
    for idx, (single_words, phrases, intent_name) in enumerate(INTENT_RULES):
        # Combine single words and phrases into patterns list
        patterns = list(single_words) + phrases
        
        # Determine action based on intent name
        if "escalation" in intent_name or "support" in intent_name:
            action = "handoff"
            response_text = None
        else:
            action = "reply"
            response_text = None  # Rules don't have hardcoded responses
        
        # Priority: higher index = lower priority (processed first = higher priority)
        # Reverse the index so first rules have higher priority
        priority = len(INTENT_RULES) - idx
        
        rule = {
            "name": intent_name,
            "patterns": patterns,
            "action": action,
            "response_text": response_text,
            "priority": priority
        }
        rules.append(rule)
    
    return rules


async def migrate_config():
    """
    Main migration function.
    """
    client = None
    try:
        # Connect to MongoDB
        logger.info("Connecting to MongoDB...")
        client = AsyncIOMotorClient(
            settings.mongo_atlas_uri,
            maxPoolSize=settings.max_pool_size,
            minPoolSize=settings.min_pool_size,
            tls=settings.mongo_ssl,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000
        )
        
        # Get database
        db = client.get_default_database()
        logger.info(f"Connected to database: {db.name}")
        
        # Test connection
        await client.admin.command('ping')
        logger.info("MongoDB connection verified")
        
        # Transform data
        logger.info("Transforming persona and rules data...")
        persona = transform_persona()
        rules = transform_rules()
        
        logger.info(f"Transformed {len(rules)} rules from INTENT_RULES")
        
        # Create BusinessConfig document
        business_id = "feelori"
        config_doc = {
            "business_id": business_id,
            "persona": persona,
            "rules": rules
        }
        
        # Check if document already exists (idempotency)
        collection = db.business_configs
        existing = await collection.find_one({"business_id": business_id})
        
        if existing:
            logger.info(f"BusinessConfig for '{business_id}' already exists. Skipping insertion.")
            logger.info("To update, delete the existing document first.")
            return
        
        # Insert document
        logger.info(f"Inserting BusinessConfig for business_id: {business_id}")
        result = await collection.insert_one(config_doc)
        logger.info(f"✓ Successfully inserted BusinessConfig with _id: {result.inserted_id}")
        
        # Verify insertion
        verify = await collection.find_one({"business_id": business_id})
        if verify:
            logger.info("✓ Verification successful: Document exists in database")
            logger.info(f"  - Persona name: {verify['persona']['name']}")
            logger.info(f"  - Number of rules: {len(verify['rules'])}")
        else:
            logger.error("✗ Verification failed: Document not found after insertion")
            sys.exit(1)
        
        logger.info("✓ Migration completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during migration: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    asyncio.run(migrate_config())

