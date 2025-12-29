# /app/services/broadcast_service.py

import asyncio
import httpx
import logging
import re
from typing import List, Dict, Optional
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config.settings import settings, BusinessConfig
from app.services.whatsapp_service import whatsapp_service
from app.services.db_service import db_service

logger = logging.getLogger(__name__)

# Template allowlist for security
ALLOWED_TEMPLATES = {
    "hello_world",
    "order_update",
    "order_confirmation_v2",
    "shipping_update_v1",
    "new_arrival_showcase",
    "video_collection_launch",
    "festival_sale_alert",
    "gentle_greeting_v1"
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
        # Get current time in IST
        ist = ZoneInfo("Asia/Kolkata")
        now = datetime.now(ist)
        
        # Define quiet hours (20:00 to 09:00)
        current_hour = now.hour
        if current_hour >= 20 or current_hour < 9:
            return True
        return False
    
    async def send_broadcast(
        self,
        target_business_id: str,
        template_name: str,
        recipients: List[str],
        variables: Dict,
        dry_run: bool = True,
        job_id: Optional[str] = None
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
        # Use business_config for logging and validation
        logger.info(f"Broadcast initiated for business: {business_config.business_id} ({business_config.business_name})")
        
        # Guardrail 2: Template Validation
        self._validate_template(template_name)
        
        # Extract variables
        raw_body_params = variables.get("body_params", [])
        header_text_param = variables.get("header_text_param")
        header_image_url = variables.get("header_image_url")
        header_video_url = variables.get("header_video_url")
        button_url_suffix = variables.get("button_url_suffix")
        button_url_param = variables.get("button_url_param")  # Keep for backward compatibility
        
        sent_count = 0
        failed_count = 0
        skipped_count = 0
        
        for recipient in recipients:
            try:
                formatted_phone = self._format_phone(recipient)
                
                # Check opt-out status for this recipient
                customer = await db_service.get_customer(formatted_phone)
                if customer and customer.get("opted_out"):
                    logger.debug(f"Skipping opted-out user {formatted_phone[:4]}...")
                    skipped_count += 1
                    continue
                
                # Smart Name Injection: Replace {{name}} with customer's first_name
                user_body_params = []
                for param in raw_body_params:
                    if param == "{{name}}":
                        # Replace with customer's first_name or default to "there"
                        customer_name = customer.get("first_name", "there") if customer else "there"
                        user_body_params.append(customer_name)
                    else:
                        # Keep param as is
                        user_body_params.append(param)
                
                if dry_run:
                    logger.info(
                        "Dry run broadcast",
                        extra={
                            "business": business_config.business_id,
                            "business_name": business_config.business_name,
                            "recipient": formatted_phone[:4] + "...",
                            "template": template_name,
                            "personalized": "{{name}}" in raw_body_params
                        }
                    )
                    sent_count += 1
                    await asyncio.sleep(0.5)  # Still throttle in dry run
                    continue
                
                # Use button_url_suffix if provided, otherwise fall back to button_url_param
                final_button_param = button_url_suffix if button_url_suffix else button_url_param
                
                # Use whatsapp_service.send_template_message for each user
                wamid = await whatsapp_service.send_template_message(
                    to=formatted_phone,
                    template_name=template_name,
                    body_params=user_body_params,
                    header_image_url=header_image_url,
                    header_text_param=header_text_param,
                    header_video_url=header_video_url,
                    button_url_param=final_button_param,
                    source="broadcast"
                )
                
                if wamid:
                    # Link message to job immediately so we don't miss fast delivery webhooks
                    if job_id:
                        await db_service.link_message_to_job(wamid, job_id)
                    
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
            "skipped_count": skipped_count,
            "dry_run": dry_run
        }
    
    async def execute_job(self, job_id: str, **kwargs):
        """
        Wrapper to run a broadcast and update the job status in DB.
        
        Args:
            job_id: Broadcast job ID
            **kwargs: Arguments to pass to send_broadcast (target_business_id, template_name, recipients, variables, dry_run)
        """
        try:
            # 1. Mark as Processing
            await db_service.update_broadcast_job(job_id, {
                "status": "processing",
                "started_at": datetime.utcnow()
            })
            
            # 2. Run the actual broadcast
            # Pass all kwargs (template_name, recipients, etc.) to send_broadcast
            # Include job_id so messages can be linked to this job
            result = await self.send_broadcast(**kwargs, job_id=job_id)
            
            # 3. Mark as Completed
            await db_service.update_broadcast_job(job_id, {
                "status": "completed",
                "stats.sent": result.get("sent_count", 0),
                "stats.failed": result.get("failed_count", 0),
                "completed_at": datetime.utcnow()
            })
            
        except Exception as e:
            logger.error(f"Broadcast Job {job_id} failed: {e}", exc_info=True)
            # Mark as Failed if it crashes
            await db_service.update_broadcast_job(job_id, {
                "status": "failed",
                "error": str(e)
            })
    
    async def close(self):
        """Close HTTP client."""
        await self.http_client.aclose()


# Global instance
broadcast_service = BroadcastService()

