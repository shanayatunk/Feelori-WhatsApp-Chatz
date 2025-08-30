import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# FIX: Explicitly load the .env file from the current directory.
# This ensures that when the script is run directly, it finds the necessary variables
# before the settings module is imported and validated.
load_dotenv()

# Load application settings and rules AFTER loading the environment
from app.config.settings import settings
from app.config.rules import INTENT_RULES

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def migrate_rules_to_db():
    """
    A one-time script to migrate intent rules from app/config/rules.py
    into the MongoDB database.
    """
    # When running this script locally, we connect to 'localhost' instead of 'mongo'.
    local_mongo_uri = settings.mongo_atlas_uri.replace("mongo:27017", "localhost:27017")
    logger.info(f"Connecting to MongoDB at: {local_mongo_uri}")
    
    try:
        client = AsyncIOMotorClient(local_mongo_uri)
        db = client.get_default_database()
        rules_collection = db.rules
        # Check connection
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Please ensure your docker-compose containers are running.")
        return

    logger.info(f"Found {len(INTENT_RULES)} rules to migrate in rules.py.")
    migrated_count = 0

    for single_words, phrases, intent_name in INTENT_RULES:
        rule_doc = {
            "name": intent_name,
            "keywords": list(single_words),
            "phrases": list(phrases)
        }
        
        # Use update_one with upsert=True to avoid creating duplicates
        result = await rules_collection.update_one(
            {"name": intent_name},
            {"$set": rule_doc},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"  -> Migrated new rule: '{intent_name}'")
            migrated_count += 1
        elif result.modified_count > 0:
            logger.info(f"  -> Updated existing rule: '{intent_name}'")
        else:
            logger.info(f"  -> Rule '{intent_name}' already up to date.")

    logger.info(f"\nMigration complete. Added {migrated_count} new rules to the database.")
    client.close()


if __name__ == "__main__":
    asyncio.run(migrate_rules_to_db())
