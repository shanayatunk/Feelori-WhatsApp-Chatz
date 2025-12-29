# /app/services/broadcast_service.py

import asyncio
import httpx
import logging
import re
from typing import List, Dict
from datetime import datetime
import pytz

from app.config.settings import settings, BusinessConfig
from app.services.whatsapp_service import whatsapp_service
from app.services.db_service import db_service

logger = logging.getLogger(__name__)

# Template allowlist for security
ALLOWED_TEMPLATES = {
    "hello_world",
    "order_update",
    "order_confirmation_v2",
    "shipping_update_v1"
}


class BroadcastService:
    """
    Service for sending WhatsApp template broadcasts with business isolation and safety guardrails.
    """
    
    def __init__(self):
        self.base_url = "https://graph.facebook.com/v18.0"
        self.http_client = httpx.AsyncClient(timeout=15.0)
    
    def _find_business_config(self, target_business_id: str) -> BusinessConfig:
        """
        Find business configuration from BUSINESS_REGISTRY.
        
        Args:
            target_business_id: The business ID to find
            
        Returns:
            BusinessConfig instance
            
        Raises:
            ValueError: If business ID is not found in registry
        """
        for config in settings.BUSINESS_REGISTRY.values():
            if config.business_id == target_business_id:
                return config
        
        raise ValueError(f"Business ID {target_business_id} not configured")
    
    def _validate_template(self, template_name: str) -> None:
        """
        Validate template name against allowlist.
        
        Args:
            template_name: Template name to validate
            
        Raises:
            ValueError: If template is not in allowlist
        """
        if template_name not in ALLOWED_TEMPLATES:
            raise ValueError(f"Template '{template_name}' is not in the allowlist. Allowed templates: {ALLOWED_TEMPLATES}")
    
    def _format_phone(self, phone: str) -> str:
        """Format phone number for WhatsApp API."""
        clean_phone = re.sub(r"[^\d+]", "", phone)
        if not clean_phone.startswith("+"):
            clean_phone = "+" + clean_phone.lstrip("+")
        return clean_phone
    
    def is_quiet_hours(self) -> bool:
        """
        Check if current time is within quiet hours (8:00 PM to 9:00 AM IST).
        
        Returns:
            True if within quiet hours, False otherwise
        """
        ist = pytz.timezone("Asia/Kolkata")
        now_ist = datetime.now(ist)
        current_hour = now_ist.hour
        
        # Quiet hours: 8:00 PM (20:00) to 9:00 AM (09:00)
        # This means: hour >= 20 OR hour < 9
        return current_hour >= 20 or current_hour < 9
    
    async def send_broadcast(
        self,
        target_business_id: str,
        template_name: str,
        recipients: List[str],
        variables: Dict,
        dry_run: bool = True
    ) -> Dict:
        """
        Send a WhatsApp template broadcast to multiple recipients.
        
        Args:
            target_business_id: Business ID (e.g., "feelori")
            template_name: WhatsApp template name
            recipients: List of phone numbers
            variables: Dictionary of template variables (e.g., {"body_params": [...], "header_text_param": "...", "button_url_param": "..."})
            dry_run: If True, log but don't send actual messages
            
        Returns:
            Dictionary with status and sent_count
            
        Raises:
            ValueError: If quiet hours check fails
        """
        # Check 1: Quiet Hours Guardrail
        if self.is_quiet_hours():
            error_msg = "Cannot send broadcast during quiet hours (8:00 PM - 9:00 AM IST). Please try again later."
            logger.warning(error_msg)
            if not dry_run:
                raise ValueError(error_msg)
            # In dry run, just log warning but continue
        
        # Guardrail 1: Business Isolation
        business_config = self._find_business_config(target_business_id)
        
        # Guardrail 2: Template Validation
        self._validate_template(template_name)
        
        # Extract variables
        body_params = variables.get("body_params", [])
        header_text_param = variables.get("header_text_param")
        header_image_url = variables.get("header_image_url")
        button_url_param = variables.get("button_url_param")
        
        sent_count = 0
        failed_count = 0
        
        for recipient in recipients:
            try:
                formatted_phone = self._format_phone(recipient)
                
                if dry_run:
                    logger.info(
                        "Dry run broadcast",
                        extra={
                            "business": target_business_id,
                            "recipient": formatted_phone[:4] + "...",
                            "template": template_name
                        }
                    )
                    sent_count += 1
                    await asyncio.sleep(0.5)  # Still throttle in dry run
                    continue
                
                # Use whatsapp_service.send_template_message for each user
                wamid = await whatsapp_service.send_template_message(
                    to=formatted_phone,
                    template_name=template_name,
                    body_params=body_params,
                    header_image_url=header_image_url,
                    header_text_param=header_text_param,
                    button_url_param=button_url_param,
                    source="broadcast"
                )
                
                if wamid:
                    # Message is already logged by send_whatsapp_request with source="broadcast"
                    logger.info(f"Broadcast sent to {formatted_phone[:4]}... (wamid: {wamid})")
                    sent_count += 1
                else:
                    logger.error(f"Broadcast failed to {formatted_phone[:4]}...: send_template_message returned None")
                    failed_count += 1
                
                # Guardrail 3: Throttling
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error sending broadcast to {recipient[:4] if recipient else 'unknown'}...: {e}", exc_info=True)
                failed_count += 1
        
        return {
            "status": "success",
            "sent_count": sent_count,
            "failed_count": failed_count,
            "dry_run": dry_run
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Global instance
broadcast_service = BroadcastService()

