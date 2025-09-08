# /app/utils/lifecycle.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from app.utils.logging import setup_logging
from app.utils.alerting import alerting_service
from app.utils.queue import message_queue
from app.services.db_service import db_service
from app.services import security_service
from app.services.string_service import string_service
from app.services.rule_service import rule_service
from app.config.settings import settings
from app.utils.tasks import refresh_visual_search_index

# This file manages the application's lifespan, handling startup tasks like
# initializing services and shutdown tasks like cleaning up connections.

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler(timezone="Asia/Kolkata") # Set the timezone for the scheduler

# This global variable will hold the hashed password.
ADMIN_PASSWORD_HASH: str | None = None

def setup_sentry():
    # Sentry initialization logic can be added here if needed
    pass

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup and shutdown events."""
    global ADMIN_PASSWORD_HASH

    setup_logging()
    setup_sentry()
    
    logger.info("Application starting up...")
    
    # Hash the admin password securely on startup
    try:
        if settings.environment == "test":
            test_password = "a_secure_test_password_123"
            ADMIN_PASSWORD_HASH = security_service.EnhancedSecurityService.hash_password(test_password)
            logger.info("Using test password hash")
        else:
            ADMIN_PASSWORD_HASH = security_service.EnhancedSecurityService.hash_password(settings.admin_password)
            logger.info("Admin password hash created successfully")
        
        logger.info(f"ADMIN_PASSWORD_HASH type: {type(ADMIN_PASSWORD_HASH)}")
        logger.info(f"ADMIN_PASSWORD_HASH length: {len(ADMIN_PASSWORD_HASH) if ADMIN_PASSWORD_HASH else 'None'}")
        
    except Exception as e:
        logger.error(f"Failed to create password hash: {e}")
        logger.warning("Password hashing failed - authentication will use direct comparison (not secure for production)")
        ADMIN_PASSWORD_HASH = None

    await db_service.create_indexes()
    await string_service.load_strings()
    await rule_service.load_rules()
    
    await message_queue.start_workers()
    
    # --- Corrected Scheduler Logic ---
    # The scheduler will only start if the visual search feature is enabled in the settings.
    if settings.VISUAL_SEARCH_ENABLED:
        scheduler.add_job(
            refresh_visual_search_index, 
            'cron', 
            hour=3, 
            minute=0, 
            id="daily_rebuild_index_job"
        )
        scheduler.start()
        logger.info("Scheduler started. Visual search index will refresh daily at 3:00 AM IST.")
    else:
        logger.info("Visual search is disabled. Scheduler will not be started.")
    # --- End of Correction ---

    logger.info("Application startup complete. Ready to accept requests.")
    
    yield  # Application is now running
    
    logger.info("Application shutting down...")
    if scheduler.running:
        scheduler.shutdown()
        logger.info("Scheduler shut down.")
        
    await message_queue.stop_workers()
    await alerting_service.cleanup()
    if db_service.client:
        db_service.client.close()