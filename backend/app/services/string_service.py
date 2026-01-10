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
        Gets a string template and replaces {{Variables}} with business config values.
        """
        # 1. Fetch the raw string template (e.g., "Call us at {{Support_Phone}}")
        raw_string = self.get_string(key)
        if not raw_string:
            return ""
        
        # 2. Fetch the business configuration
        config = get_business_config(business_id)
        
        # 3. Define the variable map (Ingredients)
        # We use .get() or getattr() to be safe if a field is missing
        replacements = {
            "{{Store_Name}}": config.business_name,
            "{{Store_Address}}": config.business_address,
            "{{Support_Phone}}": config.support_phone,
            "{{Wholesale_Phone}}": getattr(config, "wholesale_phone", config.support_phone),
            "{{Support_Email}}": config.support_email,
            "{{Store_Hours}}": getattr(config, "store_hours", "11 AM - 8 PM"),
            "{{Website_URL}}": config.website_url,
            "{{Shipping_Policy_URL}}": getattr(config, "shipping_policy_url", config.website_url),
            "{{Google_Review_Link}}": getattr(config, "google_review_url", config.website_url),
            "{{Social_Media_Links}}": getattr(config, "social_media_links", "")
        }

        # 4. Also include any dynamic kwargs passed in code (e.g., order_number)
        # Convert kwargs keys to {{Key}} format if they aren't already
        for k, v in kwargs.items():
            replacements[f"{{{{{k}}}}}"] = str(v)

        # 5. Perform the replacement
        formatted_text = raw_string
        for placeholder, value in replacements.items():
            if value is not None:
                formatted_text = formatted_text.replace(placeholder, str(value))
        
        return formatted_text

# Globally accessible instance
string_service = StringService()