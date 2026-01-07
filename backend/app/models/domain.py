# /app/models/domain.py

import re
import html
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime

logger = logging.getLogger(__name__)

class Product(BaseModel):
    id: str
    title: str
    description: str
    price: float
    variant_id: str
    first_variant_id: Optional[str] = None
    sku: Optional[str] = None
    currency: str = "INR"
    image_url: Optional[str] = None
    availability: str = "in_stock"
    tags: List[str] = []
    handle: str = ""

    @classmethod
    def from_shopify_api(cls, product_data: Dict[str, Any]) -> Optional["Product"]:
        """
        A factory method to create a Product instance from a raw Shopify API dictionary.
        This acts as a translator, handling the complex structure of the API response.
        """
        try:
            raw_description = product_data.get("body_html", "") or ""
            clean_description = html.unescape(re.sub("<[^<]+?>", "", raw_description)).strip()

            variants = product_data.get("variants", [])
            if not variants:
                logger.warning(f"Product ID {product_data.get('id')} has no variants. Skipping.")
                return None

            first_variant = variants[0]
            
            images = product_data.get("images", [])
            image_url = images[0].get("src") if images else None
            
            inventory = first_variant.get("inventory_quantity")
            availability = "in_stock" if inventory and inventory > 0 else "out_of_stock"

            tags_str = product_data.get("tags", "")
            tags_list = [tag.strip() for tag in tags_str.split(",")] if tags_str else []

            # Extract variant ID and strip GID prefix for WhatsApp catalog compatibility
            variant_id_raw = str(first_variant.get("id", ""))
            first_variant_id = None
            if variant_id_raw:
                # Strip GID prefix: gid://shopify/ProductVariant/45068921667773 -> 45068921667773
                if 'gid://' in variant_id_raw:
                    first_variant_id = variant_id_raw.rstrip('/').split('/')[-1]
                else:
                    first_variant_id = variant_id_raw

            return cls(
                id=str(product_data.get("id")),
                title=product_data.get("title", "No Title"),
                handle=product_data.get("handle", ""),
                description=clean_description,
                price=float(first_variant.get("price", 0.0)),
                variant_id=variant_id_raw,
                first_variant_id=first_variant_id,
                sku=first_variant.get("sku"), # Added SKU
                image_url=image_url,
                availability=availability,
                tags=tags_list,
            )
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Could not parse product with ID {product_data.get('id')}: {e}")
            return None

class Customer(BaseModel):
    id: Optional[str] = None
    phone: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_seen: datetime = Field(default_factory=datetime.utcnow)

class Conversation(BaseModel):
    phone: str
    history: List[Dict[str, str]] = []
    last_updated: datetime = Field(default_factory=datetime.utcnow)
    mode: Literal["bot", "human"] = "bot"
    locked_by: Optional[str] = None  # Stores the user_id who took over
    last_mode_change_at: Optional[datetime] = None

class Order(BaseModel):
    id: Optional[str] = None
    customer_phone: str
    products: List[Dict] # e.g., [{"variant_id": "...", "quantity": 1}]
    status: str = "pending"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    shopify_order_id: Optional[str] = None

class MessageLog(BaseModel):
    wamid: str
    phone: str
    direction: str # "inbound" or "outbound"
    message_type: str
    content: str
    status: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
