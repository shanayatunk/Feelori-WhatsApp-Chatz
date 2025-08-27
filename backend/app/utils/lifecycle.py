# /app/utils/lifecycle.py

import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI

from app.utils.logging import setup_logging
from app.utils.alerting import alerting_service
from app.utils.queue import message_queue
from app.services.db_service import db_service
from app.services.order_service import scheduler
from app.services import security_service
from app.config.settings import settings

# This file manages the application's lifespan, handling startup tasks like
# initializing services and shutdown tasks like cleaning up connections.

logger = logging.getLogger(__name__)

# This global variable will hold the hashed password and be imported by the auth route.
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
        
        # Debug logging to verify hash creation
        logger.info(f"ADMIN_PASSWORD_HASH type: {type(ADMIN_PASSWORD_HASH)}")
        logger.info(f"ADMIN_PASSWORD_HASH length: {len(ADMIN_PASSWORD_HASH) if ADMIN_PASSWORD_HASH else 'None'}")
        
    except Exception as e:
        logger.error(f"Failed to create password hash: {e}")
        logger.warning("Password hashing failed - authentication will use direct comparison (not secure for production)")
        ADMIN_PASSWORD_HASH = None

    await db_service.create_indexes()
    
    await message_queue.start_workers()
    scheduler.start()
    
    logger.info("Application startup complete. Ready to accept requests.")
    
    yield  # Application is running
    
    logger.info("Application shutting down...")
    scheduler.shutdown()
    await message_queue.stop_workers()
    await alerting_service.cleanup()
    if db_service.client:
        db_service.client.close()
    
    logger.info("Application shutdown complete.")