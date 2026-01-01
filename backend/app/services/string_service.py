import logging
from typing import Dict, List
from app.services.db_service import db_service
from app.config import persona

logger = logging.getLogger(__name__)

class StringService:
    def __init__(self):
        self._strings_cache: Dict[str, str] = {}
        logger.info("StringService initialized.")

    async def load_strings(self):
        """Loads all strings from the database into the in-memory cache using defaults + overrides strategy."""
        logger.info("Loading strings from database into cache...")
        
        # First, initialize with defaults
        self._strings_cache = {
            "FEELORI_SYSTEM_PROMPT": persona.FEELORI_SYSTEM_PROMPT,
            "GOLDEN_SYSTEM_PROMPT": persona.GOLDEN_SYSTEM_PROMPT
        }
        
        # Second, fetch all strings from DB and update the cache (DB values overwrite defaults)
        try:
            db_strings = await db_service.get_all_strings()
            for s in db_strings:
                self._strings_cache[s['key']] = s['value']
            logger.info(f"Successfully loaded {len(self._strings_cache)} strings into cache (defaults + DB overrides).")
        except Exception as e:
            logger.error(f"Failed to load strings from database: {e}", exc_info=True)
            logger.warning("Using default strings only.")

    def get_string(self, key: str, default: str = "") -> str:
        """Gets a string from the cache, falling back to a default value."""
        return self._strings_cache.get(key, default)

    def get_all_strings(self) -> List[Dict[str, str]]:
        """Returns the full list of cached items so the Admin UI sees both defaults and edits."""
        return [{"key": key, "value": value} for key, value in self._strings_cache.items()]

# Globally accessible instance
string_service = StringService()