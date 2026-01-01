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


def _mask_phone_for_log(phone: str) -> str:
    """Masks phone number for logging, showing only last 4 digits."""
    sanitized = EnhancedSecurityService.sanitize_phone_number(phone)
    return f"***{sanitized[-4:]}" if sanitized else "N/A"

class ShopifyService:
    def __init__(self):
        self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "shopify")
        self.http_client = httpx.AsyncClient(
            timeout=httpx.Timeout(10.0, read=30.0, connect=5.0)
        )

    def _get_credentials(self, business_id: str) -> Tuple[str, str, Optional[str]]:
        """
        Returns (store_url, access_token, storefront_access_token) based on business_id.
        
        Args:
            business_id: Business identifier (e.g., 'feelori', 'goldencollections')
            
        Returns:
            Tuple of (store_url, access_token, storefront_access_token)
        """
        normalized_id = (business_id or "feelori").strip().lower()
        
        if normalized_id == "goldencollections" and settings.golden_shopify_store_url:
            store_url = settings.golden_shopify_store_url.replace('https://', '').replace('http://', '')
            return (
                store_url,
                settings.golden_shopify_access_token or "",
                settings.golden_shopify_storefront_access_token
            )
        else:
            # Default to Feelori
            store_url = settings.shopify_store_url.replace('https://', '').replace('http://', '')
            return (
                store_url,
                settings.shopify_access_token,
                settings.shopify_storefront_access_token
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
    
    async def get_product_image_url(self, product_id: int, business_id: str = "feelori") -> Optional[str]:
        """Gets a product's primary image URL using the REST API."""
        try:
            store_url, access_token, _ = self._get_credentials(business_id)
            url = f"https://{store_url}/admin/api/2025-07/products/{product_id}/images.json"
            headers = {"X-Shopify-Access-Token": access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            images = resp.json().get("images", [])
            return images[0].get("src") if images else None
        except Exception as e:
            logger.error(f"get_product_image_url_error for product_id {product_id}: {e}")
            return None

    async def get_inventory_for_variant(self, variant_id: int, business_id: str = "feelori") -> Optional[int]:
        """Gets the inventory quantity for a single variant ID."""
        try:
            store_url, access_token, _ = self._get_credentials(business_id)
            url = f"https://{store_url}/admin/api/2025-07/variants/{variant_id}.json"
            headers = {"X-Shopify-Access-Token": access_token}
            resp = await self.resilient_api_call(self.http_client.get, url, headers=headers)
            resp.raise_for_status()
            variant_data = resp.json().get("variant", {})
            return variant_data.get("inventory_quantity")
        except Exception as e:
            logger.error(f"get_inventory_for_variant_error for variant_id {variant_id}: {e}")
            return None

    async def get_product_by_handle(self, handle: str, business_id: str = "feelori") -> Optional[Product]:
        """Gets a single product by its handle (URL slug)."""
        products, _ = await self.get_products(f'handle:"{handle}"', limit=1, business_id=business_id)
        return products[0] if products else None

    async def get_products(self, query: str, limit: int = 25, sort_key: str = "RELEVANCE", filters: Optional[Dict] = None, business_id: str = "feelori") -> Tuple[List[Product], int]:
        """Executes a product search via the Admin GraphQL API with smart query logic."""
        store_url, access_token, _ = self._get_credentials(business_id)
        cache_key = f"shopify_search:{business_id}:{query}:{limit}:{sort_key}"
        cached_products = await cache_service.get(cache_key)
        if cached_products:
            try:
                products_data = json.loads(cached_products)
                products = [Product(**p) for p in products_data]
                return products, len(products)
            except (json.JSONDecodeError, TypeError):
                logger.warning(f"Corrupted product search cache for query: {query}")

        try:
            # Smart Query Logic: Handle singular/plural and wildcards
            keywords = query.split(' AND ') if ' AND ' in query else [query]
            
            query_parts = []
            for keyword in keywords:
                keyword = keyword.strip()
                if not keyword:
                    continue
                
                # Strip trailing 's' to get singular form
                singular = keyword.rstrip('s') if keyword.endswith('s') and len(keyword) > 1 else keyword
                
                # Build search with wildcards for both singular and plural
                part = f'(title:{singular}* OR title:{keyword}* OR product_type:{singular}* OR product_type:{keyword}* OR tag:{singular} OR tag:{keyword})'
                query_parts.append(part)
            
            if not query_parts:
                return [], 0
            
            search_query = " AND ".join(query_parts)
            
            # GraphQL Query
            gql_query = """
            query($query: String!, $limit: Int!) {
              products(first: $limit, query: $query, sortKey: RELEVANCE) {
                edges {
                  node {
                    id
                    title
                    handle
                    descriptionHtml
                    tags
                    productType
                    variants(first: 1) {
                      edges {
                        node {
                          id
                          price
                          inventoryQuantity
                        }
                      }
                    }
                    images(first: 1) {
                      edges {
                        node {
                          originalSrc
                        }
                      }
                    }
                  }
                }
              }
            }
            """
            
            # Execute GraphQL query
            data = await self._execute_gql_query(
                gql_query,
                variables={"query": search_query, "limit": limit},
                store_url=store_url,
                access_token=access_token
            )
            
            # Parse response
            product_edges = data.get("products", {}).get("edges", [])
            products = self._parse_products(product_edges)

            # Cache results
            await cache_service.set(cache_key, json.dumps([p.dict() for p in products]), ttl=600)

            return products, len(products)

        except Exception as e:
            logger.error(f"shopify_get_products_error using GraphQL API: {e}", exc_info=True)
            return [], 0

    async def get_product_by_id(self, product_id: str, business_id: str = "feelori") -> Optional[Product]:
        """Gets a single product by its GraphQL GID using the Admin API."""
        store_url, access_token, _ = self._get_credentials(business_id)
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
            data = await self._execute_gql_query(gql_query, {"id": product_id}, store_url, access_token)
            if not data.get("node"): 
                return None
            products = self._parse_products([{"node": data.get("node")}])
            return products[0] if products else None
        except Exception as e:
            logger.error(f"shopify_product_by_id_error for {product_id}: {e}")
            return None

    async def get_product_variants(self, product_id: str, business_id: str = "feelori") -> List[Dict]:
        """Gets all variants for a given product ID using the Admin API."""
        store_url, access_token, _ = self._get_credentials(business_id)
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
            data = await self._execute_gql_query(gql_query, {"id": product_id}, store_url, access_token)
            variant_edges = data.get("node", {}).get("variants", {}).get("edges", [])
            return [edge["node"] for edge in variant_edges]
        except Exception as e:
            logger.error(f"get_product_variants_error for {product_id}: {e}")
            return []

    async def get_all_products(self, business_id: str = "feelori") -> List[Product]:
        """Fetches all published products from Shopify, handling pagination."""
        store_url, access_token, _ = self._get_credentials(business_id)
        products = []
        url = f"https://{store_url}/admin/api/2025-07/products.json?status=active&limit=250"
        
        logger.info("Starting to fetch all products from Shopify...")
        
        while url:
            try:
                headers = {"X-Shopify-Access-Token": access_token}
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
    
    async def create_cart(self, business_id: str = "feelori") -> Optional[str]:
        store_url, _, storefront_token = self._get_credentials(business_id)
        gql_mutation = "mutation { cartCreate { cart { id checkoutUrl } userErrors { field message } } }"
        data = await self._execute_storefront_gql_query(gql_mutation, store_url, storefront_token, variables=None)
        return (data.get("cartCreate") or {}).get("cart", {}).get("id")

    async def add_item_to_cart(self, cart_id: str, variant_id: str, quantity: int = 1, business_id: str = "feelori") -> bool:
        store_url, _, storefront_token = self._get_credentials(business_id)
        gql_mutation = '''
        mutation($cartId: ID!, $lines: [CartLineInput!]!) {
          cartLinesAdd(cartId: $cartId, lines: $lines) { cart { id } userErrors { field message } }
        }'''
        variables = {"cartId": cart_id, "lines": [{"merchandiseId": variant_id, "quantity": quantity}]}
        data = await self._execute_storefront_gql_query(gql_mutation, store_url, storefront_token, variables)
        return not ((data.get("cartLinesAdd") or {}).get("userErrors"))

    async def get_checkout_url(self, cart_id: str, business_id: str = "feelori") -> Optional[str]:
        store_url, _, storefront_token = self._get_credentials(business_id)
        gql_query = "query($cartId: ID!) { cart(id: $cartId) { checkoutUrl } }"
        data = await self._execute_storefront_gql_query(gql_query, store_url, storefront_token, {"cartId": cart_id})
        return (data.get("cart") or {}).get("checkoutUrl")

    # --- Order Management ---
    
# In /app/services/shopify_service.py

    async def search_orders_by_phone(self, phone_number: str, max_fetch: int = 50, business_id: str = "feelori") -> List[Dict]:
        """Search for recent orders using a customer's phone number via the REST API.
        Ensures strict filtering to prevent leaking other customers' orders.
        """
        store_url, access_token, _ = self._get_credentials(business_id)
        sanitized_user_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        cache_key = f"shopify:orders_by_phone:{business_id}:{sanitized_user_phone}"
        cached = await cache_service.get(cache_key)
        if cached:
            return json.loads(cached)

        rest_url = f"https://{store_url}/admin/api/2025-07/orders.json"
        headers = {"X-Shopify-Access-Token": access_token}

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

            # --- Enhanced Diagnostic Logging (DEBUG only, masked) ---
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"--- RAW SHOPIFY ORDER DATA CHECK (First 5 of {len(orders)} orders) ---")
                for i, order in enumerate(orders[:5]):  # Log first 5 to avoid noise
                    order_name = order.get("name", "N/A")
                    customer_info = order.get("customer", {}) or {}
                    customer_name = f"{customer_info.get('first_name', '')} {customer_info.get('last_name', '')}".strip()
                    
                    def _mask(p): 
                        s = EnhancedSecurityService.sanitize_phone_number(p or "")
                        return f"***{s[-4:]}" if s else "N/A"

                    logger.debug(f"Order #{i+1}: {order_name} for Customer: '{customer_name}'")
                    logger.debug(f"  -> Top-level 'phone': {_mask(order.get('phone'))}")
                    
                    shipping_address = order.get("shipping_address", {}) or {}
                    logger.debug(f"  -> shipping_address.phone: {_mask(shipping_address.get('phone'))}")
                    
                    billing_address = order.get("billing_address", {}) or {}
                    logger.debug(f"  -> billing_address.phone: {_mask(billing_address.get('phone'))}")
                logger.debug("--- END OF RAW SHOPIFY ORDER DATA CHECK ---")
            # --- End of Logging ---

            # --- Security filtering logic ---
            # FIX: Use a more robust comparison of the last 10 digits
            user_phone_suffix = sanitized_user_phone[-10:]
            filtered_orders = []

            for o in orders:
                order_phone = (
                    o.get("phone")
                    or (o.get("shipping_address") or {}).get("phone")
                    or (o.get("billing_address") or {}).get("phone")
                    or ""
                )
                sanitized_order_phone = EnhancedSecurityService.sanitize_phone_number(order_phone)

                if sanitized_order_phone and sanitized_order_phone.endswith(user_phone_suffix):
                    filtered_orders.append(o)

            await cache_service.set(
                cache_key, json.dumps(filtered_orders, default=str), ttl=120
            )
            logger.info(f"Shopify orders found for {_mask_phone_for_log(phone_number)}: {len(filtered_orders)}")
            return filtered_orders

        except Exception as e:
            logger.error(f"Shopify search_orders_by_phone error: {e}", exc_info=True)
            return []


    

    async def get_order_by_name(self, order_name: str, business_id: str = "feelori") -> Optional[Dict]:
        """
        Gets a single order by its name (e.g., "#1037").
        The name format should include the '#'.
        """
        store_url, access_token, _ = self._get_credentials(business_id)
        # Ensure correct format
        if not order_name.startswith('#'):
            order_name = f"#{order_name}"

        cache_key = f"shopify:order_by_name:{business_id}:{order_name}"
        cached = await cache_service.get(cache_key)
        if cached:
            return json.loads(cached)

        rest_url = f"https://{store_url}/admin/api/2025-07/orders.json"
        headers = {"X-Shopify-Access-Token": access_token}

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

    async def fulfill_order(self, order_id: int, tracking_number: str, packer_name: str, carrier: str = "India Post", business_id: str = "feelori") -> Tuple[bool, Optional[int], Optional[str]]:
        """Fulfills an order using the Fulfillment Orders API and returns the tracking URL."""
        try:
            store_url, access_token, _ = self._get_credentials(business_id)
            fo_url = f"https://{store_url}/admin/api/2025-07/orders/{order_id}/fulfillment_orders.json"
            headers = {"X-Shopify-Access-Token": access_token}
            fo_resp = await self.resilient_api_call(self.http_client.get, fo_url, headers=headers)
            fo_resp.raise_for_status()
            fulfillment_orders = fo_resp.json().get("fulfillment_orders", [])
            open_fo = next((fo for fo in fulfillment_orders if fo.get("status") == "open"), None)
            if not open_fo: 
                return False, None, None

            line_items = [{"id": item["id"], "quantity": item["fulfillable_quantity"]} for item in open_fo.get("line_items", []) if item.get("fulfillable_quantity", 0) > 0]
            payload = {
                "fulfillment": {
                    "line_items_by_fulfillment_order": [{"fulfillment_order_id": open_fo["id"], "fulfillment_order_line_items": line_items}],
                    "tracking_info": {"number": tracking_number, "company": carrier},
                    "notify_customer": True
                }
            }
            fulfillment_url = f"https://{store_url}/admin/api/2025-07/fulfillments.json"
            resp = await self.resilient_api_call(self.http_client.post, fulfillment_url, json=payload, headers=headers)
            
            if resp.status_code == 201:
                fulfillment = resp.json().get("fulfillment", {})
                fulfillment_id = fulfillment.get("id")
                tracking_url = (fulfillment.get("tracking_urls") or [""])[0]
                return True, fulfillment_id, tracking_url
            
            # --- THIS IS THE FIX ---
            # Explicitly return the failure tuple if the status code is not 201.
            logger.error(f"Shopify fulfillment failed for order {order_id} with status {resp.status_code}: {resp.text}")
            return False, None, None
            # --- END OF FIX ---

        except Exception as e:
            logger.error(f"Shopify fulfill_order exception for {order_id}: {e}", exc_info=True)
            return False, None, None

            
    # --- URL Generation Helpers ---

    def get_add_to_cart_url(self, variant_gid: str, business_id: str = "feelori") -> str:
        store_url, _, _ = self._get_credentials(business_id)
        numeric_variant_id = variant_gid.split('/')[-1]
        return f"https://{store_url}/cart/{numeric_variant_id}:1"

    def get_product_page_url(self, handle: str, business_id: str = "feelori") -> str:
        store_url, _, _ = self._get_credentials(business_id)
        return f"https://{store_url}/products/{handle}"


    # --- Private Helper Methods ---

    async def _shopify_search(self, query: str, limit: int, sort_key: str, filters: Optional[Dict], business_id: str = "feelori") -> List[Dict]:
        """Executes a GraphQL query against the Shopify Storefront API."""
        store_url, _, storefront_token = self._get_credentials(business_id)
        if not storefront_token: 
            return []
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
        url = f"https://{store_url}/api/2025-07/graphql.json"
        headers = {"Content-Type": "application/json", "X-Shopify-Storefront-Access-Token": storefront_token}
        try:
            resp = await self.http_client.post(url, headers=headers, json=graphql_payload)
            resp.raise_for_status()
            response_data = resp.json()
            return response_data.get("data", {}).get("products", {}).get("edges", [])
        except Exception as e:
            logger.error(f"Shopify _shopify_search error: {e}")
            return []

    async def _execute_gql_query(self, query: str, variables: Dict, store_url: str, access_token: str) -> Dict:
        """Executes a GraphQL query against the Shopify Admin API."""
        url = f"https://{store_url}/admin/api/2025-07/graphql.json"
        headers = {"X-Shopify-Access-Token": access_token, "Content-Type": "application/json"}
        resp = await self.resilient_api_call(self.http_client.post, url, json={"query": query, "variables": variables}, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", {})

    async def _execute_storefront_gql_query(self, query: str, store_url: str, storefront_token: str, variables: Optional[Dict] = None) -> Dict:
        """Executes a GraphQL query against the Shopify Storefront API."""
        if not storefront_token: 
            return {}
        url = f"https://{store_url}/api/2025-07/graphql.json"
        headers = {"X-Shopify-Storefront-Access-Token": storefront_token, "Content-Type": "application/json"}
        resp = await self.resilient_api_call(self.http_client.post, url, json={"query": query, "variables": variables or {}}, headers=headers)
        resp.raise_for_status()
        return resp.json().get("data", {})

    def _parse_products(self, product_edges: List[Dict]) -> List[Product]:
        """Parses GraphQL product edges from the Admin API into Pydantic Product models."""
        products = []
        for edge in product_edges:
            node = edge.get("node", {})
            if not node: 
                continue
            variant_edge = node.get("variants", {}).get("edges", [])
            if not variant_edge: 
                continue
            
            image_edge = node.get("images", {}).get("edges", [])
            # Use `inventoryQuantity` for Admin API results
            inventory = variant_edge[0]["node"].get("inventoryQuantity")
            # Handle both descriptionHtml (GraphQL) and bodyHtml (REST) for compatibility
            description_html = node.get("descriptionHtml") or node.get("bodyHtml", "")
            clean_description = html.unescape(re.sub("<[^<]+?>", "", description_html))

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
shopify_service = ShopifyService()