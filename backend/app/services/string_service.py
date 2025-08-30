import logging
from typing import Dict
from app.services.db_service import db_service
from app.config import strings as default_strings

logger = logging.getLogger(__name__)

class StringService:
    def __init__(self):
        self._strings_cache: Dict[str, str] = {}
        logger.info("StringService initialized.")

    async def load_strings(self):
        """Loads all strings from the database into the in-memory cache."""
        logger.info("Loading strings from database into cache...")
        try:
            db_strings = await db_service.get_all_strings()
            self._strings_cache = {s['key']: s['value'] for s in db_strings}
            logger.info(f"Successfully loaded {len(self._strings_cache)} strings into cache.")
        except Exception as e:
            logger.error(f"Failed to load strings from database: {e}", exc_info=True)
            # Fallback to default strings from the file if DB load fails
            self._load_defaults()

    def get_string(self, key: str, default: str = "") -> str:
        """Gets a string from the cache, falling back to a default value."""
        return self._strings_cache.get(key, default)

    def _load_defaults(self):
        """Loads default strings from the strings.py file as a fallback."""
        logger.warning("Falling back to default strings from config file.")
        for key in dir(default_strings):
            if key.isupper():
                self._strings_cache[key] = getattr(default_strings, key)

# Globally accessible instance
string_service = StringService()