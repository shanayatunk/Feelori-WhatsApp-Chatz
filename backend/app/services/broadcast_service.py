# /app/services/broadcast_service.py

import asyncio
import httpx
import logging
import re
from typing import List, Dict

from app.config.settings import settings, BusinessConfig

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
        """
        # Guardrail 1: Business Isolation
        business_config = self._find_business_config(target_business_id)
        access_token = settings.get_business_token(business_config)
        phone_id = business_config.phone_number_id
        
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
                
                # Build template payload
                components = []
                
                # Header component
                if header_image_url and header_text_param:
                    logger.error(f"Template '{template_name}' cannot have both image and text header")
                    failed_count += 1
                    continue
                
                if header_image_url:
                    components.append({
                        "type": "header",
                        "parameters": [{"type": "image", "image": {"link": header_image_url}}]
                    })
                elif header_text_param:
                    components.append({
                        "type": "header",
                        "parameters": [{"type": "text", "text": str(header_text_param)}]
                    })
                
                # Body component
                if body_params:
                    components.append({
                        "type": "body",
                        "parameters": [{"type": "text", "text": str(p)} for p in body_params]
                    })
                
                # Button component
                if button_url_param:
                    components.append({
                        "type": "button",
                        "sub_type": "url",
                        "index": "0",
                        "parameters": [{"type": "text", "text": button_url_param}]
                    })
                
                payload = {
                    "messaging_product": "whatsapp",
                    "to": formatted_phone,
                    "type": "template",
                    "template": {
                        "name": template_name,
                        "language": {"code": "en"},
                        "components": components
                    }
                }
                
                # Send HTTP request
                url = f"{self.base_url}/{phone_id}/messages"
                headers = {
                    "Authorization": f"Bearer {access_token}",
                    "Content-Type": "application/json"
                }
                
                response = await self.http_client.post(url, json=payload, headers=headers)
                
                if response.status_code == 200:
                    response_data = response.json()
                    message_id = response_data.get("messages", [{}])[0].get("id")
                    logger.info(f"Broadcast sent to {formatted_phone[:4]}... (wamid: {message_id})")
                    sent_count += 1
                else:
                    error_data = response.json()
                    error_message = (error_data.get("error") or {}).get("message", "Unknown error")
                    logger.error(f"Broadcast failed to {formatted_phone[:4]}...: {response.status_code} - {error_message}")
                    failed_count += 1
                
                # Guardrail 2: Throttling
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

