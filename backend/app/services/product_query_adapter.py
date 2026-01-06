# /app/services/product_query_adapter.py

"""
Product Query Adapter for Phase 4.2 Marketing Workflow.

This module provides a pure function to convert marketing workflow slots
(category and price_range) into Shopify-compatible product search queries.

This adapter serves as a translation layer between the marketing workflow
state and the commerce search API, ensuring clean separation of concerns.
"""


def build_product_query(category: str, price_range: str) -> dict:
    """
    Build a Shopify-compatible product query from marketing workflow slots.
    
    Args:
        category: Product category (e.g., "earrings", "necklaces", "bangles")
        price_range: Price range identifier (e.g., "under_2000", "2000_5000", "above_5000")
    
    Returns:
        Dictionary with product_type, price_min, price_max, and in_stock_only keys
        suitable for Shopify product search.
    
    Raises:
        ValueError: If category or price_range is not recognized.
    """
    # Category mapping (normalize to title case)
    category_map = {
        "earrings": "Earrings",
        "necklaces": "Necklaces",
        "bangles": "Bangles",
    }
    
    normalized_category = category.lower().strip()
    if normalized_category not in category_map:
        raise ValueError(f"Unknown category: {category}. Expected one of: {list(category_map.keys())}")
    
    product_type = category_map[normalized_category]
    
    # Price range mapping
    price_range_map = {
        "under_2000": {"price_max": 2000},
        "2000_5000": {"price_min": 2000, "price_max": 5000},
        "above_5000": {"price_min": 5000},
    }
    
    normalized_price_range = price_range.lower().strip()
    if normalized_price_range not in price_range_map:
        raise ValueError(
            f"Unknown price_range: {price_range}. "
            f"Expected one of: {list(price_range_map.keys())}"
        )
    
    price_params = price_range_map[normalized_price_range]
    
    # Build the query dict
    query = {
        "product_type": product_type,
        "in_stock_only": True,
    }
    
    # Add price parameters
    query.update(price_params)
    
    return query

