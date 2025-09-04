# backend/app/utils/logging.py

import logging
import sys
import structlog
from app.config.settings import settings

# This utility sets up structured logging (JSON format in production)
# for consistent and machine-readable logs across the application.

def setup_logging():
    """
    Configures structured logging using structlog, properly integrated
    with Python's standard logging to work with Gunicorn/Uvicorn.
    """
    # Define shared processors for structlog
    shared_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]

    # Determine the final renderer based on the environment
    if settings.environment == "development":
        final_processor = structlog.dev.ConsoleRenderer()
    else:
        # Use JSONRenderer for production/staging environments
        final_processor = structlog.processors.JSONRenderer()

    # Configure structlog
    structlog.configure(
        processors=shared_processors + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure the standard logging formatter
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=final_processor,
        foreign_pre_chain=shared_processors,
    )

    # Configure the root logger
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    root_logger.addHandler(handler)
    root_logger.setLevel(logging.INFO) # Set the default log level

    # Silence overly verbose libraries if necessary
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)