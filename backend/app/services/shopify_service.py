# /app/services/shopify_service.py

import re
import html
import json
import httpx
import logging
import tenacity
from typing import Optional, List, Dict, Tuple

from app.config.settings import settings
from app.models.domain import Product
from app.utils.circuit_breaker import RedisCircuitBreaker
from app.services.cache_service import cache_service
from app.services.security_service import EnhancedSecurityService

logger = logging.getLogger(__name__)

class ShopifyService:
    def __init__(self, store_url: str, access_token: str, storefront_token: Optional[str]):
        self.store_url = store_url.replace('https://', '').replace('http://', '')
        self.access_token = access_token
        self.storefront_token = storefront_token
        self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "shopify")
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, read=30.0, connect=5.0)
        )

    @tenacity.retry(
        retry=tenacity.retry_if_exception_type((httpx.RequestError, httpx.TimeoutException)),
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=2, min=1, max=10),
        before_sleep=tenacity.before_sleep_log(logger, logging.WARNING)
    )
    async def resilient_api_call(self, func, *args, **kwargs):
        return await func(*args, **kwargs)

    # --- Product Lookups ---
    
    async def get_product_image_url(self, product_id: int) -> Optional[str]:
        """Gets a product's primary image URL using the REST API."""
        try:
            url = f"https://{self.store_url}/admin/api/2025-07/products/{product_id}/images.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            images = resp.json().get("images", [])
            return images[0].get("src") if images else None
        except Exception as e:
            logger.error(f"get_product_image_url_error for product_id {product_id}: {e}")
            return None

    async def get_inventory_for_variant(self, variant_id: int) -> Optional[int]:
        """Gets the inventory quantity for a single variant ID."""
        try:
            url = f"https://{self.store_url}/admin/api/2025-07/variants/{variant_id}.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            variant_data = resp.json().get("variant", {})
            return variant_data.get("inventory_quantity")
        except Exception as e:
            logger.error(f"get_inventory_for_variant_error for variant_id {variant_id}: {e}")
            return None

    async def get_product_by_handle(self, handle: str) -> Optional[Product]:
        """Gets a single product by its handle (URL slug)."""
        products, _ = await self.get_products(f'handle:"{handle}"', limit=1)
        return products[0] if products else None

    async def get_products(self, query: str, limit: int = 25, sort_key: str = "RELEVANCE", filters: Optional[Dict] = None) -> Tuple[List[Product], int]:
        """Executes a product search via Storefront API."""
        try:
            edges = await self._shopify_search(query, limit, sort_key, filters)
            if not edges:
                return [], 0

            products = []
            for edge in edges:
                node = edge.get("node", {})
                variants_edge = node.get("variants", {}).get("edges", [])
                if not variants_edge:
                    continue

                variant_node = variants_edge[0].get("node", {})
                price_info = variant_node.get("priceV2", {})

                # Use `quantityAvailable` for Storefront API results
                inventory = variant_node.get("quantityAvailable")
                availability = "in_stock" if inventory is not None and inventory > 0 else "out_of_stock"

                products.append(Product(
                    id=node.get("id"),
                    title=node.get("title"),
                    description=node.get("description", "No description available."),
                    price=float(price_info.get("amount", 0.0)),
                    variant_id=variant_node.get("id"),
                    sku=variant_node.get("sku"),
                    currency=price_info.get("currencyCode", "INR"),
                    image_url=node.get("featuredImage", {}).get("url"),
                    handle=node.get("handle", ""),
                    tags=node.get("tags", []),
                    availability=availability
                ))

            unfiltered_count = len(products)
            if filters and "price" in filters and products:
                price_condition = filters["price"]
                if "lessThan" in price_condition:
                    return [p for p in products if p.price < price_condition["lessThan"]], unfiltered_count
                if "greaterThan" in price_condition:
                    return [p for p in products if p.price > price_condition["greaterThan"]], unfiltered_count

            return products, unfiltered_count
        except Exception as e:
            logger.error(f"shopify_get_products_error: {e}", exc_info=True)
            return [], 0


    async def get_product_by_id(self, product_id: str) -> Optional[Product]:
        """Gets a single product by its GraphQL GID using the Admin API."""
        gql_query = """
        query($id: ID!) {
          node(id: $id) { ... on Product { ...productFields } }
        }
        fragment productFields on Product { 
            id title handle bodyHtml productType tags 
            variants(first: 1){edges{node{id price inventoryQuantity}}} 
            images(first: 1){edges{node{originalSrc}}}
        }
        """
        try:
            data = await self._execute_gql_query(gql_query, {"id": product_id})
            if not data.get("node"): return None
            products = self._parse_products([{"node": data.get("node")}])
            return products[0] if products else None
        except Exception as e:
            logger.error(f"shopify_product_by_id_error for {product_id}: {e}")
            return None

    async def get_product_variants(self, product_id: str) -> List[Dict]:
        """Gets all variants for a given product ID using the Admin API."""
        gql_query = """
        query($id: ID!) {
          node(id: $id) {
            ... on Product {
              variants(first: 10) { edges { node { id title price inventoryQuantity } } }
            }
          }
        }
        """
        try:
            data = await self._execute_gql_query(gql_query, {"id": product_id})
            variant_edges = data.get("node", {}).get("variants", {}).get("edges", [])
            return [edge["node"] for edge in variant_edges]
        except Exception as e:
            logger.error(f"get_product_variants_error for {product_id}: {e}")
            return []

    async def get_all_products(self) -> List[Product]:
        """Fetches all published products from Shopify, handling pagination."""
        products = []
        url = f"https://{self.store_url}/admin/api/2024-01/products.json?status=active&limit=250"
        
        logger.info("Starting to fetch all products from Shopify...")
        
        while url:
            try:
                headers = {"X-Shopify-Access-Token": self.access_token}
                response = await self.http_client.get(url, headers=headers)
                response.raise_for_status()
                
                data = response.json()
                for product_data in data.get("products", []):
                    product = Product.from_shopify_api(product_data)
                    if product:
                        products.append(product)
                
                link_header = response.headers.get("link")
                if link_header:
                    links = link_header.split(", ")
                    next_link = next((link for link in links if 'rel="next"' in link), None)
                    url = next_link[next_link.find("<")+1:next_link.find(">")] if next_link else None
                else:
                    url = None

            except Exception as e:
                logger.error(f"An unexpected error occurred fetching Shopify products: {e}", exc_info=True)
                break
        
        logger.info(f"Successfully fetched a total of {len(products)} products.")
        return products

    # --- Cart and Checkout ---
    
    async def create_cart(self) -> Optional[str]:
        gql_mutation = "mutation { cartCreate { cart { id checkoutUrl } userErrors { field message } } }"
        data = await self._execute_storefront_gql_query(gql_mutation)
        return (data.get("cartCreate") or {}).get("cart", {}).get("id")

    async def add_item_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1) -> bool:
        gql_mutation = '''
        mutation($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) { cart { id } userErrors { field message } }
        }'''
        variables = {"cartId": cart_id, "lines": [{"merchandiseId": variant_id, "quantity": quantity}]}
        data = await self._execute_storefront_gql_query(gql_mutation, variables)
        return not ((data.get("cartLinesAdd") or {}).get("userErrors"))

    async def get_checkout_url(self, cart_id: str) -> Optional[str]:
        gql_query = "query($cartId: ID!) { cart(id: $cartId) { checkoutUrl } }"
        data = await self._execute_storefront_gql_query(gql_query, {"cartId": cart_id})
        return (data.get("cart") or {}).get("checkoutUrl")

    # --- Order Management ---
    
# In /app/services/shopify_service.py

    async def search_orders_by_phone(self, phone_number: str, max_fetch: int = 50) -> List[Dict]:
        """Search for recent orders using a customer's phone number via the REST API.
        Ensures strict filtering to prevent leaking other customers' orders.
        """
        # Using a sanitized phone number for the cache key is more consistent.
        sanitized_user_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        cache_key = f"shopify:orders_by_phone:{sanitized_user_phone}"
        cached = await cache_service.get(cache_key)
        if cached:
            return json.loads(cached)

        rest_url = f"https://{self.store_url}/admin/api/2025-07/orders.json"
        headers = {"X-Shopify-Access-Token": self.access_token}

        # Fetch the 50 most recent orders regardless of phone number
        params = {
            "status": "any",
            "limit": max_fetch,
            "order": "processed_at desc"  # Fetch the most recent first
        }

        try:
            resp = await self.resilient_api_call(
                self.http_client.get, rest_url, params=params, headers=headers
            )
            resp.raise_for_status()
            orders = resp.json().get("orders", []) or []

            # --- DEBUG LOGGING ---
            logger.info(f"DEBUG: Input phone_number: '{phone_number}'")
            logger.info(f"DEBUG: Sanitized user phone: '{sanitized_user_phone}'")
            logger.info(f"DEBUG: Total orders fetched: {len(orders)}")

            for i, o in enumerate(orders):
                order_name = o.get("name", "Unknown")
                raw_phone = o.get("phone", "")
                shipping_phone = (o.get("shipping_address") or {}).get("phone", "")
                billing_phone = (o.get("billing_address") or {}).get("phone", "")
                
                logger.info(f"DEBUG Order {i+1} ({order_name}):")
                logger.info(f"  - Raw phone: '{raw_phone}'")
                logger.info(f"  - Shipping phone: '{shipping_phone}'") 
                logger.info(f"  - Billing phone: '{billing_phone}'")
                
                for field_name, phone_value in [("phone", raw_phone), ("shipping", shipping_phone), ("billing", billing_phone)]:
                    if phone_value:
                        sanitized = EnhancedSecurityService.sanitize_phone_number(phone_value)
                        matches = sanitized.endswith(sanitized_user_phone)
                        logger.info(f"  - {field_name}: '{phone_value}' -> '{sanitized}' -> matches: {matches}")

            # --- Security filtering logic ---
            filtered_orders = []
            for o in orders:
                order_phone = (
                    o.get("phone")
                    or (o.get("shipping_address") or {}).get("phone")
                    or (o.get("billing_address") or {}).get("phone")
                    or ""
                )
                sanitized_order_phone = EnhancedSecurityService.sanitize_phone_number(order_phone)
                if sanitized_order_phone and sanitized_order_phone.endswith(sanitized_user_phone):
                    filtered_orders.appen_

    

    async def get_order_by_name(self, order_name: str) -> Optional[Dict]:
        """
        Gets a single order by its name (e.g., "#1037").
        The name format should include the '#'.
        """
        # Ensure correct format
        if not order_name.startswith('#'):
            order_name = f"#{order_name}"

        cache_key = f"shopify:order_by_name:{order_name}"
        cached = await cache_service.get(cache_key)
        if cached:
            return json.loads(cached)

        rest_url = f"https://{self.store_url}/admin/api/2025-07/orders.json"
        headers = {"X-Shopify-Access-Token": self.access_token}

        params = {
            "name": order_name,
            "status": "any"
        }

        try:
            resp = await self.resilient_api_call(
                self.http_client.get, rest_url, params=params, headers=headers
            )
            resp.raise_for_status()
            orders = resp.json().get("orders", [])
            
            if orders:
                order = orders[0]
                await cache_service.set(
                    cache_key, json.dumps(order, default=str), ttl=300
                )  # cache for 5 minutes
                logger.info(f"Found order {order_name} via API.")
                return order

            logger.warning(f"Could not find order with name {order_name}.")
            return None
        except Exception as e:
            logger.error(f"Shopify get_order_by_name error for {order_name}: {e}", exc_info=True)
            return None

    async def fulfill_order(self, order_id: int, tracking_number: str, packer_name: str, carrier: str = "India Post") -> Tuple[bool, Optional[int]]:
        """Fulfills an order using the Fulfillment Orders API."""
        try:
            fo_url = f"https://{self.store_url}/admin/api/2025-07/orders/{order_id}/fulfillment_orders.json"
            headers = {"X-Shopify-Access-Token": self.access_token}
            fo_resp = await self.resilient_api_call(self.http_client.get, fo_url, headers=headers)
            fo_resp.raise_for_status()
            fulfillment_orders = fo_resp.json().get("fulfillment_orders", [])
            open_fo = next((fo for fo in fulfillment_orders if fo.get("status") == "open"), None)
            if not open_fo: return False, None

            line_items = [{"id": item["id"], "quantity": item["fulfillable_quantity"]} for item in open_fo.get("line_items", []) if item.get("fulfillable_quantity", 0) > 0]
            payload = {
                "fulfillment": {
                    "line_items_by_fulfillment_order": [{"fulfillment_order_id": open_fo["id"], "fulfillment_order_line_items": line_items}],
                    "tracking_info": {"number": tracking_number, "company": carrier},
                    "notify_customer": True
                }
            }
            fulfillment_url = f"https://{self.store_url}/admin/api/2025-07/fulfillments.json"
            resp = await self.resilient_api_call(self.http_client.post, fulfillment_url, json=payload, headers=headers)
            
            if resp.status_code == 201:
                fulfillment_id = resp.json().get("fulfillment", {}).get("id")
                return True, fulfillment_id
            return False, None
        except Exception as e:
            logger.error(f"Shopify fulfill_order exception for {order_id}: {e}", exc_info=True)
            return False, None
            
    # --- URL Generation Helpers ---

    def get_add_to_cart_url(self, variant_gid: str) -> str:
        numeric_variant_id = variant_gid.split('/')[-1]
        return f"https://{self.store_url}/cart/{numeric_variant_id}:1"

    def get_product_page_url(self, handle: str) -> str:
        return f"https://{self.store_url}/products/{handle}"


    # --- Private Helper Methods ---

    async def _shopify_search(self, query: str, limit: int, sort_key: str, filters: Optional[Dict]) -> List[Dict]:
        """Executes a GraphQL query against the Shopify Storefront API."""
        if not self.storefront_token: return []
        graphql_payload = {
            "query": """
            query ($query: String!, $limit: Int!, $sortKey: ProductSortKeys!) {
              products(first: $limit, query: $query, sortKey: $sortKey) {
                edges { node { id title handle description tags featuredImage { url }
                    variants(first: 1) { edges { node { id sku priceV2 { amount currencyCode } quantityAvailable } } }
                }}
              }
            }""", "variables": {"query": query, "limit": limit, "sortKey": sort_key}
        }
        url = f"https://{self.store_url}/api/2025-07/graphql.json"
        headers = {"Content-Type": "application/json", "X-Shopify-Storefront-Access-Token": self.storefront_token}
        try:
            resp = await self.http_client.post(url, headers=headers, json=graphql_payload)
            resp.raise_for_status()
            response_data = resp.json()
            return response_data.get("data", {}).get("products", {}).get("edges", [])
        except Exception as e:
            logger.error(f"Shopify _shopify_search error: {e}")
            return []

    async def _execute_gql_query(self, query: str, variables: Dict) -> Dict:
        """Executes a GraphQL query against the Shopify Admin API."""
        url = f"https://{self.store_url}/admin/api/2025-07/graphql.json"
        headers = {"X-Shopify-Access-Token": self.access_token, "Content-Type": "application/json"}
        resp = await self.resilient_api_call(self.http_client.post, url, json={"query": query, "variables": variables}, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def _execute_storefront_gql_query(self, query: str, variables: Optional[Dict] = None) -> Dict:
        """Executes a GraphQL query against the Shopify Storefront API."""
        if not self.storefront_token: return {}
        url = f"https://{self.store_url}/api/2025-07/graphql.json"
        headers = {"X-Shopify-Storefront-Access-Token": self.storefront_token, "Content-Type": "application/json"}
        resp = await self.resilient_api_call(self.http_client.post, url, json={"query": query, "variables": variables or {}}, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", {})

    def _parse_products(self, product_edges: List[Dict]) -> List[Product]:
        """Parses GraphQL product edges from the Admin API into Pydantic Product models."""
        products = []
        for edge in product_edges:
            node = edge.get("node", {})
            if not node: continue
            variant_edge = node.get("variants", {}).get("edges", [])
            if not variant_edge: continue
            
            image_edge = node.get("images", {}).get("edges", [])
            # Use `inventoryQuantity` for Admin API results
            inventory = variant_edge[0]["node"].get("inventoryQuantity")
            clean_description = html.unescape(re.sub("<[^<]+?>", "", node.get("bodyHtml", "")))

            products.append(Product(
                id=node.get("id"), title=node.get("title"), description=clean_description,
                price=float(variant_edge[0]["node"].get("price", 0.0)),
                handle=node.get("handle"), variant_id=variant_edge[0]["node"].get("id"),
                image_url=(image_edge[0]["node"]["originalSrc"] if image_edge else None),
                availability="in_stock" if inventory and inventory > 0 else "out_of_stock",
                tags=node.get("tags", []),
            ))
        return products

# Globally accessible instance
shopify_service = ShopifyService(
    settings.shopify_store_url,
    settings.shopify_access_token,
    settings.shopify_storefront_access_token
)