# /app/models/domain.py

from pydantic import BaseModel
from typing import List, Optional

# This file contains Pydantic models that represent core business concepts
# or "domain" objects, like a Product.

class Product(BaseModel):
    id: str
    title: str
    description: str
    price: float
    variant_id: str
    sku: Optional[str] = None
    currency: str = "INR"
    image_url: Optional[str] = None
    availability: str = "in_stock"
    tags: List[str] = []
    handle: str = ""