# /app/services/broadcast_service.py

import httpx
import logging
import re
from typing import List, Dict, Optional, Any
from datetime import datetime
from zoneinfo import ZoneInfo

from app.config.settings import settings, WhatsAppBusinessConfig
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
    
    def _find_business_config(self, target_business_id: str) -> WhatsAppBusinessConfig:
        """
        Find business configuration from WHATSAPP_BUSINESS_REGISTRY.
        
        Args:
            target_business_id: The business ID to find
            
        Returns:
            WhatsAppBusinessConfig instance
            
        Raises:
            ValueError: If business ID is not found in registry
        """
        for config in settings.WHATSAPP_BUSINESS_REGISTRY.values():
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
        variables: Dict[str, Any],
        dry_run: bool = False,
        job_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Send a template message to a list of recipients.
        """
        # 1. Initialize counters (Fixes F821 Undefined name)
        success_count = 0
        failed_count = 0
        
        # Extract variables
        header_image_url = variables.get("header_image_url")
        header_video_url = variables.get("header_video_url")
        button_url_suffix = variables.get("button_url_suffix")
        raw_body_params = variables.get("body_params", [])
        
        logger.info(f"Starting broadcast: {template_name} to {len(recipients)} recipients. Job: {job_id}")

        # Phase 7A: Auto-append UTM tracking to dynamic button links
        if button_url_suffix:
            # Determine if we need '?' or '&'
            separator = "&" if "?" in button_url_suffix else "?"
            
            # Clean the template name for URL usage
            safe_campaign_name = template_name.replace(" ", "_").replace("-", "_").lower()
            
            # Construct parameters
            utm_params = f"utm_source=whatsapp&utm_medium=broadcast&utm_campaign={safe_campaign_name}"
            
            # Add Job ID for precise ROI tracking if available
            if job_id:
                utm_params += f"&utm_id={job_id}"
                
            # Append to the suffix
            button_url_suffix = f"{button_url_suffix}{separator}{utm_params}"
            
            logger.info(f"Attached UTM tracking: {button_url_suffix}")

        for recipient in recipients:
            try:
                # Format phone
                if not recipient.startswith("+"):
                    formatted_phone = f"+{recipient.strip()}"
                else:
                    formatted_phone = recipient.strip()

                # Smart Name Injection Logic
                customer = await db_service.get_customer(formatted_phone)
                customer_name = customer.get("first_name", "there") if customer else "there"

                user_body_params = []
                for param in raw_body_params:
                    if param == "{{name}}":
                        user_body_params.append(customer_name)
                    else:
                        user_body_params.append(param)

                if dry_run:
                    logger.info(f"[DRY RUN] Would send {template_name} to {formatted_phone}")
                    success_count += 1
                    continue

                # Send to WhatsApp
                wamid = await whatsapp_service.send_template_message(
                    to=formatted_phone,
                    template_name=template_name,
                    body_params=user_body_params,
                    header_image_url=header_image_url,
                    header_video_url=header_video_url,
                    button_url_param=button_url_suffix,
                    source="broadcast"
                )

                if wamid:
                    # 1. Try to Link (Update) first. 
                    # This handles the case where whatsapp_service ALREADY created the log.
                    if job_id:
                        await db_service.link_message_to_job(wamid, job_id)
                    
                    # 2. Check if we need to Log (Insert).
                    # We only log if we didn't just update it (or just rely on whatsapp_service).
                    # Actually, let's keep it simple: Just Link.
                    
                    # If whatsapp_service logged it, this link works.
                    # If whatsapp_service FAILED to log, this might miss, but that's a bigger issue.
                    
                    logger.info(f"Broadcast sent to {formatted_phone[:4]}... (wamid: {wamid}) linked to Job {job_id}")
                    success_count += 1
                else:
                    logger.error(f"Broadcast failed to {formatted_phone[:4]}...: send_template_message returned None")
                    failed_count += 1

            except Exception as e:
                logger.error(f"Error sending broadcast to {recipient}: {e}")
                failed_count += 1

        return {
            "sent_count": success_count,
            "failed_count": failed_count,
            "total_processed": len(recipients)
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

