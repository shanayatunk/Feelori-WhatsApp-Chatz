import logging
from typing import List, Dict, Any
from app.services.db_service import db_service

logger = logging.getLogger(__name__)

class RuleService:
    def __init__(self):
        self._rules_cache: List[Dict[str, Any]] = []
        logger.info("RuleService initialized.")

    async def load_rules(self):
        """Loads all rules from the database into the in-memory cache."""
        logger.info("Loading rules from database into cache...")
        try:
            self._rules_cache = await db_service.get_all_rules()
            logger.info(f"Successfully loaded {len(self._rules_cache)} rules into cache.")
        except Exception as e:
            logger.error(f"Failed to load rules from database: {e}", exc_info=True)

    def get_rules(self) -> List[Dict[str, Any]]:
        """Gets all rules from the cache."""
        return self._rules_cache

# Globally accessible instance
rule_service = RuleService()