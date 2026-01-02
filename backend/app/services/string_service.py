import logging
from typing import Dict, List
from app.services.db_service import db_service
from app.config import persona
from app.config.settings import get_business_config

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

    def get_formatted_string(self, key: str, business_id: str, **kwargs) -> str:
        """
        Gets a string template and formats it with business configuration and optional variables.
        
        Args:
            key: The string key to retrieve from the cache
            business_id: The business identifier (e.g., 'feelori', 'goldencollections')
            **kwargs: Optional override variables for formatting (e.g., order_number, customer_name)
            
        Returns:
            Formatted string with business config variables and kwargs substituted.
            Falls back to raw string if formatting fails.
        """
        # 1. Fetch the raw string template
        raw_string = self.get_string(key)
        
        # 2. Fetch the business configuration
        business_config = get_business_config(business_id)
        
        # 3. Create context dictionary with business config fields
        context = {
            "business_name": business_config.business_name,
            "support_email": business_config.support_email,
            "support_phone": business_config.support_phone,
            "website_url": business_config.website_url,
            "business_address": business_config.business_address
        }
        
        # 4. Update context with any kwargs (allows one-off overrides)
        context.update(kwargs)
        
        # 5. Safe formatting with try/except
        try:
            return raw_string.format(**context)
        except KeyError as e:
            logger.warning(f"[TEMPLATE] Missing placeholder {e} in string {key} | business={business_id}")
            return raw_string  # Fallback: Return raw string, do not crash
        except Exception as e:
            logger.error(f"Formatting error for {key}: {e}")
            return raw_string

# Globally accessible instance
string_service = StringService()