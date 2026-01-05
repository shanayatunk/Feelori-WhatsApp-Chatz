import logging
from typing import Dict, List
from app.services.db_service import db_service
from app.config.settings import get_business_config

logger = logging.getLogger(__name__)


async def _get_persona_prompt(business_id: str) -> str:
    """
    Helper to get persona prompt from MongoDB BusinessConfig.
    Falls back to fetching from database if not in request.state.
    """
    try:
        config = await db_service.db.business_configs.find_one({"business_id": business_id})
        if config and "persona" in config:
            return config["persona"].get("prompt", "")
    except Exception as e:
        logger.warning(f"Failed to fetch persona from BusinessConfig for {business_id}: {e}")
    
    # Fallback: Return empty string (caller should handle)
    return ""


class StringService:
    def __init__(self):
        self._strings_cache: Dict[str, str] = {}
        logger.info("StringService initialized.")

    async def load_strings(self):
        """Loads all strings from the database into the in-memory cache using defaults + overrides strategy."""
        logger.info("Loading strings from database into cache...")
        
        # First, initialize with defaults from BusinessConfig
        self._strings_cache = {}
        try:
            # Fetch persona prompts from BusinessConfig for feelori and goldencollections
            feelori_config = await db_service.db.business_configs.find_one({"business_id": "feelori"})
            if feelori_config and "persona" in feelori_config:
                self._strings_cache["FEELORI_SYSTEM_PROMPT"] = feelori_config["persona"].get("prompt", "")
            
            golden_config = await db_service.db.business_configs.find_one({"business_id": "goldencollections"})
            if golden_config and "persona" in golden_config:
                self._strings_cache["GOLDEN_SYSTEM_PROMPT"] = golden_config["persona"].get("prompt", "")
        except Exception as e:
            logger.warning(f"Failed to load persona from BusinessConfig: {e}")
        
        # Ensure we have at least empty strings if configs don't exist
        if "FEELORI_SYSTEM_PROMPT" not in self._strings_cache:
            self._strings_cache["FEELORI_SYSTEM_PROMPT"] = ""
        if "GOLDEN_SYSTEM_PROMPT" not in self._strings_cache:
            self._strings_cache["GOLDEN_SYSTEM_PROMPT"] = ""
        
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