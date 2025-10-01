# /app/utils/alerting.py

import httpx
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from app.config.settings import settings

# This utility provides a service for sending critical alerts to an external
# webhook, ensuring that system failures are immediately reported.

logger = logging.getLogger(__name__)

class AlertingService:
    def __init__(self, webhook_url: Optional[str]):
        self.webhook_url = webhook_url
        self.client = httpx.AsyncClient(timeout=5.0) if webhook_url else None

    async def send_critical_alert(self, error: str, context: Dict[str, Any]):
        if not self.client: 
            return
        try:
            alert_data = {
                "severity": "critical", "service": "feelori-whatsapp-assistant",
                "error": error, "context": context, "timestamp": datetime.utcnow().isoformat(),
                "environment": settings.environment
            }
            await self.client.post(self.webhook_url, json=alert_data)
        except Exception as e:
            logger.error(f"Failed to send critical alert: {e}")

    async def cleanup(self):
        if self.client: 
            await self.client.aclose()

# Globally accessible instance
alerting_service = AlertingService(settings.alerting_webhook_url)