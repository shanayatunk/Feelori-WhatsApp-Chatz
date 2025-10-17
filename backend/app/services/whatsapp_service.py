# /app/services/whatsapp_service.py

import httpx
import logging
import re
import asyncio
import tenacity
import json
from typing import Optional, Dict, Tuple

from app.config.settings import settings
from app.utils.circuit_breaker import RedisCircuitBreaker
from app.services.cache_service import cache_service
from app.models.domain import Product
from app.utils.alerting import alerting_service

logger = logging.getLogger(__name__)

class WhatsAppService:
    def __init__(self, access_token: str, phone_id: str, business_account_id: Optional[str]):
        self.access_token = access_token
        self.phone_id = phone_id
        self.business_account_id = business_account_id
        self.http_client = httpx.AsyncClient(timeout=15.0)
        self.base_url = "https://graph.facebook.com/v18.0"
        self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "whatsapp")
        self._catalog_id_cache: Optional[str] = None

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    async def resilient_api_call(self, func, *args, **kwargs):
        return await self.circuit_breaker.call(func, *args, **kwargs)

    async def send_whatsapp_request(self, payload: dict, metadata: dict | None = None) -> Optional[str]:
        """Generic method to send a request to the WhatsApp messages API."""
        try:
            to_phone = payload.get("to")
            if not to_phone:
                logger.error(f"send_whatsapp_request_invalid_phone: {to_phone}")
                return None

            url = f"{self.base_url}/{self.phone_id}/messages"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)

            if response.status_code == 200:
                response_data = response.json()
                message_id = response_data.get("messages", [{}])[0].get("id")
                logger.info(f"WhatsApp message sent to {to_phone}, wamid: {message_id}")

                from app.services.db_service import db_service
                from datetime import datetime
                
                content_payload = {}
                if payload.get("type") == "text":
                    content_payload = payload.get("text", {})
                elif payload.get("type") == "template":
                    content_payload = payload.get("template", {})
                
                log_data = {
                    "wamid": message_id,
                    "phone": to_phone,
                    "direction": "outbound",
                    "message_type": payload.get("type"),
                    "content": json.dumps(content_payload),
                    "status": "sent",
                    "timestamp": datetime.utcnow(),
                    "metadata": metadata or {}
                }
                await db_service.log_message(log_data)
                
                return message_id
            else:
                error_data = response.json()
                error_message = (error_data.get("error") or {}).get("message", "Unknown error")
                logger.error(f"whatsapp_send_failed to {to_phone}: {response.status_code} - {error_message}")
                if response.status_code == 401:
                    await alerting_service.send_critical_alert("WhatsApp authentication failed", {"error": "Invalid access token"})
                return None
        except Exception as e:
            to_phone = payload.get("to", "unknown")
            logger.error(f"whatsapp_send_error to {to_phone}: {e}", exc_info=True)
            await alerting_service.send_critical_alert("WhatsApp send message unexpected error", {"phone": to_phone, "error": str(e)})
            return None


    # --- THIS FUNCTION IS NOW CORRECTED ---
    async def send_message(self, to_phone: str, message: str, image_url: Optional[str] = None) -> Optional[str]:
        """
        Sends a message. If image_url is provided, sends an image with the message as a caption.
        Otherwise, sends a simple text message.
        """
        clean_phone = re.sub(r"[^\d+]", "", to_phone)
        if not clean_phone.startswith("+"):
            clean_phone = "+" + clean_phone.lstrip("+")
            
        payload = {
            "messaging_product": "whatsapp",
            "recipient_type": "individual",
            "to": clean_phone,
        }

        if image_url:
            payload["type"] = "image"
            payload["image"] = {"link": image_url, "caption": message[:1024]}
        else:
            payload["type"] = "text"
            payload["text"] = {"body": message[:4096]}
        
        return await self.send_whatsapp_request(payload)

    # --- THIS FUNCTION IS NOW CORRECTED ---
async def send_template_message(
        self,
        to: str,
        template_name: str,
        body_params: list,
        header_image_url: Optional[str] = None,
        header_text_param: Optional[str] = None, # <-- ADD THIS NEW PARAMETER
        button_url_param: Optional[str] = None
    ) -> Optional[str]:
        """Sends a pre-approved WhatsApp message template with optional header and button parameters."""
        clean_phone = re.sub(r"[^\d+]", "", to)
        if not clean_phone.startswith("+"):
            clean_phone = "+" + clean_phone.lstrip("+")

        components = []

        # --- REVISED LOGIC TO HANDLE BOTH HEADER TYPES ---
        if header_image_url and header_text_param:
            logger.error(f"Template message for '{template_name}' cannot have both an image and text header.")
            return None

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
        # --- END OF REVISED LOGIC ---

        # Add body component
        if body_params:
            components.append({
                "type": "body",
                "parameters": [{"type": "text", "text": str(p)} for p in body_params]
            })

        # Add button component
        if button_url_param:
            components.append({
                "type": "button",
                "sub_type": "url",
                "index": "0",
                "parameters": [{"type": "text", "text": button_url_param}]
            })

        payload = {
            "messaging_product": "whatsapp",
            "to": clean_phone,
            "type": "template",
            "template": {
                "name": template_name,
                "language": {"code": "en"},
                "components": components
            }
        }
        return await self.send_whatsapp_request(payload)

    async def get_catalog_id(self) -> Optional[str]:
        """Fetches and caches the WhatsApp Business catalog ID, prioritizing env settings."""
        if settings.whatsapp_catalog_id:
            logger.info(f"Using WhatsApp Catalog ID from settings: {settings.whatsapp_catalog_id}")
            return settings.whatsapp_catalog_id

        if self._catalog_id_cache:
            return self._catalog_id_cache
        
        if not self.business_account_id:
            logger.error("whatsapp_business_account_id_not_set")
            return None

        url = f"{self.base_url}/{self.business_account_id}/catalogs"
        params = {"access_token": self.access_token}
        try:
            resp = await self.http_client.get(url, params=params)
            resp.raise_for_status()
            catalogs = resp.json().get("data", [])
            if catalogs:
                self._catalog_id_cache = catalogs[0].get("id")
                logger.info(f"WhatsApp catalog ID fetched: {self._catalog_id_cache}")
                return self._catalog_id_cache
            logger.warning("whatsapp_no_catalog_found")
            return None
        except Exception as e:
            logger.error(f"whatsapp_get_catalog_id_error: {e}")
            return None

    async def send_multi_product_message(self, to: str, header_text: str, body_text: str, footer_text: str, catalog_id: Optional[str], section_title: str, product_items: list, fallback_products: list):
        """Sends a WhatsApp Multi-Product Message with a fallback to individual cards."""
        if catalog_id and product_items:
            try:
                payload = {
                    "messaging_product": "whatsapp", "to": to, "type": "interactive",
                    "interactive": {
                        "type": "product_list",
                        "header": {"type": "text", "text": header_text},
                        "body": {"text": body_text},
                        "footer": {"text": footer_text},
                        "action": {"catalog_id": catalog_id, "sections": [{"title": section_title, "product_items": product_items}]}
                    }
                }
                url = f"{self.base_url}/{self.phone_id}/messages"
                headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
                response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
                if response.status_code == 200:
                    logger.info(f"WhatsApp multi-product message sent to {to}")
                    return
                logger.error(f"WhatsApp multi-product send failed: {response.status_code} - {response.text}")
            except Exception as e:
                logger.error(f"WhatsApp multi-product send error: {e}")

        logger.info(f"Using fallback message for {to}")
        for product in fallback_products:
            if not product.image_url: 
                continue
            try:
                payload = {
                    "messaging_product": "whatsapp", "to": to, "type": "interactive",
                    "interactive": {
                        "type": "button",
                        "header": {"type": "image", "image": {"link": product.image_url}},
                        "body": {"text": f"{product.title}\nTap below to view details."},
                        "footer": {"text": footer_text},
                        "action": {"buttons": [{"type": "reply", "reply": {"id": f"product_{product.id}", "title": "View Details"}}]}
                    }
                }
                url = f"{self.base_url}/{self.phone_id}/messages"
                headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
                response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)
                if response.status_code != 200:
                     logger.error(f"WhatsApp fallback send failed for {product.id}: {response.text}")
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"WhatsApp fallback template error: {e}")

    async def send_product_detail_with_buttons(self, to_phone: str, product: Product):
        """Sends a product detail message with interactive reply buttons."""
        try:
            short_desc = (product.description[:120] + '...') if len(product.description) > 120 else product.description
            body_text = f"*{product.title}*\n\n💰 ₹{product.price:,.2f}\n\n✨ {short_desc}"
            payload = {
                "messaging_product": "whatsapp", "to": to_phone, "type": "interactive",
                "interactive": {
                    "type": "button", "body": {"text": body_text},
                    "action": { "buttons": [
                            {"type": "reply", "reply": {"id": f"buy_{product.id}", "title": "🛒 Buy Now"}},
                            {"type": "reply", "reply": {"id": f"more_{product.id}", "title": "📖 More Info"}},
                            {"type": "reply", "reply": {"id": f"similar_{product.id}", "title": "🔍 Similar Items"}}
                        ]}
                }
            }
            if product.image_url:
                payload["interactive"]["header"] = {"type": "image", "image": {"link": product.image_url}}

            url = f"{self.base_url}/{self.phone_id}/messages"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)

            if response.status_code == 200:
                logger.info(f"Interactive product detail sent to {to_phone} for product {product.id}")
                return True
            logger.error(f"Interactive product detail failed: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"Interactive product detail error for {to_phone}: {e}")
            return False

    async def send_quick_replies(self, to_phone: str, message: str, options: Dict[str, str]):
        """Sends a message with up to 3 quick reply buttons."""
        try:
            buttons = [{"type": "reply", "reply": {"id": option_id, "title": title[:20]}} for option_id, title in list(options.items())[:3]]
            payload = {
                "messaging_product": "whatsapp", "to": to_phone, "type": "interactive",
                "interactive": {"type": "button", "body": {"text": message}, "action": {"buttons": buttons}}
            }
            url = f"{self.base_url}/{self.phone_id}/messages"
            headers = {"Authorization": f"Bearer {self.access_token}", "Content-Type": "application/json"}
            response = await self.resilient_api_call(self.http_client.post, url, json=payload, headers=headers)

            if response.status_code == 200:
                logger.info(f"Quick replies sent to {to_phone}")
                return True
            logger.error(f"Quick replies failed: {response.status_code} - {response.text}")
            return False
        except Exception as e:
            logger.error(f"Send quick replies error for {to_phone}: {e}")
            return False

    async def get_media_url(self, media_id: str) -> Optional[str]:
        """Fetches a temporary URL for a media object from WhatsApp."""
        try:
            url = f"{self.base_url}/{media_id}"
            headers = {"Authorization": f"Bearer {self.access_token}"}
            resp = await self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            return resp.json().get("url")
        except Exception as e:
            logger.error(f"get_media_url_failed for {media_id}: {e}")
            return None

    async def get_media_content(self, media_id: str) -> Tuple[Optional[bytes], Optional[str]]:
        """Downloads media bytes and returns (bytes, mime_type)."""
        try:
            url = await self.get_media_url(media_id)
            if not url: 
                return None, None
            
            headers = {"Authorization": f"Bearer {self.access_token}"}
            resp = await self.http_client.get(url, headers=headers)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type")
            return resp.content, content_type
        except Exception as e:
            logger.error(f"download_media_failed for {media_id}: {e}")
            return None, None

# Globally accessible instance
whatsapp_service = WhatsAppService(
    settings.whatsapp_access_token,
    settings.whatsapp_phone_id,
    settings.whatsapp_business_account_id
)