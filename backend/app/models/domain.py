# /app/models/domain.py

import re
import html
import logging
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

# This file defines the core Pydantic models used throughout the application's
# business logic. These models ensure data consistency and provide validation.

logger = logging.getLogger(__name__)

class Product(BaseModel):
    id: str
    title: str
    handle: str
    description: Optional[str] = None
    price: float
    variant_id: str
    image_url: Optional[str] = None
    availability: str
    tags: List[str] = []

    @classmethod
    def from_shopify_api(cls, product_data: Dict[str, Any]) -> Optional["Product"]:
        """
        A factory method to create a Product instance from a raw Shopify API dictionary.
        This acts as a translator, handling the complex structure of the API response.
        """
        try:
            # Shopify's description is in 'body_html', so we clean it by removing HTML tags.
            raw_description = product_data.get("body_html", "") or ""
            clean_description = html.unescape(re.sub("<[^<]+?>", "", raw_description)).strip()

            # Price and variant info are in a nested list.
            variants = product_data.get("variants", [])
            if not variants:
                logger.warning(f"Product ID {product_data.get('id')} has no variants. Skipping.")
                return None # Cannot create a product without a variant

            first_variant = variants[0]
            
            # Image URL is also nested.
            images = product_data.get("images", [])
            image_url = images[0].get("src") if images else None
            
            # Determine availability from inventory quantity.
            inventory = first_variant.get("inventory_quantity")
            availability = "in_stock" if inventory and inventory > 0 else "out_of_stock"

            # Tags are a single string in the REST API, so we split it.
            tags_str = product_data.get("tags", "")
            tags_list = [tag.strip() for tag in tags_str.split(",")] if tags_str else []

            return cls(
                id=str(product_data.get("id")),
                title=product_data.get("title", "No Title"),
                handle=product_data.get("handle", ""),
                description=clean_description,
                price=float(first_variant.get("price", 0.0)),
                variant_id=str(first_variant.get("id")),
                image_url=image_url,
                availability=availability,
                tags=tags_list,
            )
        except (KeyError, IndexError, TypeError) as e:
            # If the product data is malformed for any reason, log it and return None
            # so it doesn't crash the entire indexing process.
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
