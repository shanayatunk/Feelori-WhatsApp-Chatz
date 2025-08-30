import asyncio
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

# Load application settings and the strings module
from app.config.settings import settings
from app.config import strings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def migrate_strings_to_db():
    """
    A one-time script to migrate string constants from app/config/strings.py
    into the MongoDB database.
    """
    local_mongo_uri = settings.mongo_atlas_uri.replace("mongo:27017", "localhost:27017")
    logger.info(f"Connecting to MongoDB at: {local_mongo_uri}")
    
    try:
        client = AsyncIOMotorClient(local_mongo_uri)
        db = client.get_default_database()
        strings_collection = db.strings
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB.")
    except Exception as e:
        logger.error(f"Failed to connect to MongoDB: {e}")
        logger.error("Please ensure your docker-compose containers are running.")
        return

    # Find all uppercase variables in the strings module
    string_keys = [key for key in dir(strings) if key.isupper()]
    logger.info(f"Found {len(string_keys)} string resources to migrate.")
    migrated_count = 0

    for key in string_keys:
        value = getattr(strings, key)
        string_doc = {
            "key": key,
            "value": value
        }
        
        # Use update_one with upsert=True to avoid creating duplicates
        result = await strings_collection.update_one(
            {"key": key},
            {"$set": string_doc},
            upsert=True
        )
        
        if result.upserted_id:
            logger.info(f"  -> Migrated new string: '{key}'")
            migrated_count += 1
        elif result.modified_count > 0:
            logger.info(f"  -> Updated existing string: '{key}'")
        else:
            logger.info(f"  -> String '{key}' already up to date.")

    logger.info(f"\nMigration complete. Added {migrated_count} new strings to the database.")
    client.close()


if __name__ == "__main__":
    asyncio.run(migrate_strings_to_db())