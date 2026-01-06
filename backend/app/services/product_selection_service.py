# /app/services/product_selection_service.py

"""
Product Selection Service for Phase 4.2.B Marketing Workflow.

This module provides product fetching and selection logic for the marketing workflow.
It handles query building, Shopify product retrieval, filtering, ranking, and
normalization to WhatsApp-safe product structures.

This is Phase 4.2.B - selection only. No messaging or workflow state mutation.
"""

from typing import List, Dict
from app.services.product_query_adapter import build_product_query
from app.services.shopify_service import shopify_service
from app.models.domain import Product


def _extract_numeric_id(product_id: str) -> str:
    """Extract numeric ID from Shopify GID (e.g., 'gid://shopify/Product/123' -> '123')."""
    if 'gid://' in product_id:
        return product_id.rstrip('/').split('/')[-1]
    return product_id


async def fetch_and_select_products(
    category: str,
    price_range: str,
    limit: int = 5,
    business_id: str = "feelori"
) -> List[Dict]:
    """
    Fetch and select products based on category and price range.
    
    Args:
        category: Product category (e.g., "earrings", "necklaces", "bangles")
        price_range: Price range identifier (e.g., "under_2000", "2000_5000", "above_5000")
        limit: Maximum number of products to return (default: 5)
        business_id: Business identifier for multi-tenant support (default: "feelori")
    
    Returns:
        List of normalized product dictionaries with WhatsApp-safe structure:
        {
            "product_id": str,      # Numeric ID string
            "title": str,
            "price": float,
            "currency": str,
            "image_url": str,
            "product_url": str
        }
    
    Raises:
        ValueError: If zero products are found after filtering.
        Exception: If Shopify returns an error.
    """
    # Build query using adapter
    query_dict = build_product_query(category, price_range)
    
    # Convert query dict to Shopify service format
    # product_type becomes text query
    text_query = query_dict.get("product_type", "")
    
    # Convert price_min/price_max to filters format
    filters = {}
    if "price_min" in query_dict:
        filters["price"] = {"greaterThan": query_dict["price_min"]}
    if "price_max" in query_dict:
        if "price" in filters:
            filters["price"]["lessThan"] = query_dict["price_max"]
        else:
            filters["price"] = {"lessThan": query_dict["price_max"]}
    
    # Fetch products from Shopify
    try:
        products, _ = await shopify_service.get_products(
            query=text_query,
            filters=filters if filters else None,
            limit=limit * 3,  # Fetch more to account for filtering
            business_id=business_id
        )
    except Exception as e:
        raise Exception(f"Shopify product fetch failed: {str(e)}") from e
    
    if not products:
        raise ValueError(f"No products found for category '{category}' and price range '{price_range}'")
    
    # Filter products
    filtered_products = []
    for product in products:
        # Filter out out-of-stock products
        if product.availability != "in_stock":
            continue
        
        # Filter out products without images
        if not product.image_url:
            continue
        
        filtered_products.append(product)
    
    if not filtered_products:
        raise ValueError(
            f"No in-stock products with images found for category '{category}' "
            f"and price range '{price_range}'"
        )
    
    # Rank products deterministically
    # Priority: 1. In stock (already filtered), 2. Has images (already filtered),
    # 3. Lower price first, 4. Keep original order as tiebreaker
    sorted_products = sorted(
        filtered_products,
        key=lambda p: p.price
    )
    
    # Limit results
    selected_products = sorted_products[:limit]
    
    # Normalize to WhatsApp-safe structure
    normalized_products = []
    for product in selected_products:
        # Get product URL
        product_url = shopify_service.get_product_page_url(product.handle, business_id=business_id)
        
        normalized_products.append({
            "product_id": _extract_numeric_id(product.id),
            "title": product.title,
            "price": product.price,
            "currency": product.currency,
            "image_url": product.image_url,
            "product_url": product_url
        })
    
    return normalized_products

