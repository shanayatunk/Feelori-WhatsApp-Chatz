#!/usr/bin/env python3
"""
One-time database setup script for SaaS multi-tenancy indexes.

Creates indexes on:
- conversations collection: unique index on (tenant_id, external_user_id)
- conversations collection: sorting index on (tenant_id, status, updated_at)
- message_logs collection: index on (conversation_id, created_at)

Usage:
    python scripts/create_saas_indexes.py
"""

import asyncio
import logging
import sys
from pathlib import Path

# Add parent directory to path to import app modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from motor.motor_asyncio import AsyncIOMotorClient
from app.config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_saas_indexes():
    """
    Create indexes for SaaS multi-tenancy collections.
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
        
        # 1. Create unique index on conversations: (tenant_id, external_user_id)
        logger.info("Creating unique index on conversations: (tenant_id, external_user_id)")
        try:
            await db.conversations.create_index(
                [("tenant_id", 1), ("external_user_id", 1)],
                unique=True,
                name="tenant_user_unique"
            )
            logger.info("✓ Unique index created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
        
        # 2. Create sorting index on conversations: (tenant_id, status, updated_at)
        logger.info("Creating sorting index on conversations: (tenant_id, status, updated_at)")
        try:
            await db.conversations.create_index(
                [("tenant_id", 1), ("status", 1), ("updated_at", -1)],
                name="tenant_status_updated"
            )
            logger.info("✓ Sorting index created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
        
        # 3. Create index on message_logs: (conversation_id, created_at)
        logger.info("Creating index on message_logs: (conversation_id, created_at)")
        try:
            await db.message_logs.create_index(
                [("conversation_id", 1), ("created_at", 1)],
                name="conversation_created"
            )
            logger.info("✓ Message index created successfully")
        except Exception as e:
            logger.warning(f"Index creation warning (may already exist): {e}")
        
        logger.info("✓ All indexes created successfully!")
        
        # List created indexes for verification
        logger.info("\nVerifying indexes...")
        conversations_indexes = await db.conversations.list_indexes().to_list(length=None)
        logger.info(f"Conversations indexes: {[idx['name'] for idx in conversations_indexes]}")
        
        message_logs_indexes = await db.message_logs.list_indexes().to_list(length=None)
        logger.info(f"Message logs indexes: {[idx['name'] for idx in message_logs_indexes]}")
        
    except Exception as e:
        logger.error(f"Error creating indexes: {e}", exc_info=True)
        sys.exit(1)
    finally:
        if client:
            client.close()
            logger.info("MongoDB connection closed")


if __name__ == "__main__":
    asyncio.run(create_saas_indexes())

