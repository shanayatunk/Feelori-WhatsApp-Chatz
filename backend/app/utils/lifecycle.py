# /app/utils/lifecycle.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.utils.logging import setup_logging
from app.utils.alerting import alerting_service
from app.utils.queue import message_queue
from app.services.db_service import db_service
from app.services import security_service
from app.services.string_service import string_service
from app.services.rule_service import rule_service
from app.config.settings import settings

# This file manages the application's lifespan, handling startup tasks like
# initializing services and shutdown tasks like cleaning up connections.

logger = logging.getLogger(__name__)
# Note: The global ADMIN_PASSWORD_HASH variable has been removed.

def setup_sentry():
    # Sentry initialization logic can be added here if needed
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    setup_logging()
    setup_sentry()
    
    logger.info("Application starting up...")
    
    # --- The password hashing block has been completely removed ---

    await db_service.create_indexes()
    await string_service.load_strings()
    await rule_service.load_rules()
    
    await message_queue.start_workers()
    
    logger.info("Application startup complete. Ready to accept requests.")
    
    yield  # Application is now running
    
    logger.info("Application shutting down...")
        
    await message_queue.stop_workers()
    await alerting_service.cleanup()
    if db_service.client:
        db_service.client.close()