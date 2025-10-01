import asyncio
import logging

# --- THIS BLOCK IS THE FIX ---
# It explicitly loads the .env.local file for local script execution
from dotenv import load_dotenv
load_dotenv(dotenv_path='backend/.env.local')
# --- END OF FIX ---

from app.server import services, VisualProductMatcher

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def main():
    """Initializes services and runs the one-time product indexing process."""
    logger.info("--- Initializing services to build visual search index ---")
    
    # Define services and matcher here to ensure cleanup happens
    builder_services = None
    
    try:
        # Initialize services using the loaded .env.local config
        await services.initialize()
        builder_services = services
        
        logger.info("\n--- Starting to build the visual product index ---")
        
        # Correctly point to the database file in the backend folder
        matcher = VisualProductMatcher(db_path="backend/product_embeddings.db")
        await matcher.index_all_products()
        
        logger.info("\n--- Indexing complete. You can now start the main server. ---")

    except Exception:
        logger.error("An error occurred during the indexing process.", exc_info=True)
    finally:
        if builder_services:
            await builder_services.cleanup()

if __name__ == "__main__":
    asyncio.run(main())