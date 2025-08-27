import asyncio
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# --- Basic Setup ---
load_dotenv()
# --- Configuration (Copied from server.py) ---

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

@dataclass
class SearchConfig:
    """Minimal config for testing"""
    STOP_WORDS: Set[str] = field(default_factory=lambda: {
        "a", "about", "an", "any", "are", "authentic", "buy", "can", "do",
        "does", "find", "for", "get", "genuine", "give", "have", "help",
        "how", "i", "im", "is", "looking", "material", "me", "need",
        "please", "quality", "real", "send", "show", "some", "tell",
        "to", "want", "what", "when", "where", "which", "why", "you"
    })
    MIN_WORD_LENGTH: int = 2
    CATEGORY_EXCLUSIONS: Dict[str, List[str]] = field(default_factory=dict)

class QueryBuilder:
    """Simplified QueryBuilder for testing"""
    def __init__(self, config: SearchConfig):
        self.config = config
        self.plural_mappings = {
            "necklaces": "necklace", "earrings": "earring", "bangles": "bangle",
            "bracelets": "bracelet", "rings": "ring", "rubies": "ruby"
        }

    def _extract_keywords(self, message: str) -> List[str]:
        words = [
            word.lower().strip() for word in re.findall(r'\b\w+\b', message.lower())
            if len(word) >= self.config.MIN_WORD_LENGTH and
               word.lower() not in self.config.STOP_WORDS
        ]
        return self._deduplicate_keywords(self._normalize_keywords(words))

    def _normalize_keywords(self, keywords: List[str]) -> List[str]:
        return [self.plural_mappings.get(keyword, keyword) for keyword in keywords]

    def _deduplicate_keywords(self, words: List[str]) -> List[str]:
        return list(dict.fromkeys(words))

    def build_query(self, message: str) -> str:
        keywords = self._extract_keywords(message)
        if not keywords:
            return ""
        field_clauses = [
            f"(title:{kw}* OR tag:{kw} OR product_type:{kw})"
            for kw in keywords
        ]
        # Using OR for broader, more reliable test results
        return " OR ".join(field_clauses)

class ShopifyService:
    """Isolated Shopify Service for direct testing"""
    def __init__(self, store_url: str, storefront_token: str):
        self.store_url = store_url.replace('https://', '').replace('http://', '')
        self.storefront_token = storefront_token
        self.http_client = httpx.AsyncClient(timeout=10.0)

    async def get_products(self, query: str, limit: int = 5) -> List[dict]:
        graphql_query = """
        query ($query: String!, $limit: Int!) {
          products(first: $limit, query: $query, sortKey: RELEVANCE) {
            edges {
              node {
                id
                title
                handle
                description
                tags
                variants(first: 1) {
                  edges { node { priceV2 { amount currencyCode } } }
                }
              }
            }
          }
        }
        """
        variables = {"query": query, "limit": limit}
        url = f"https://{self.store_url}/api/2023-07/graphql.json"
        headers = {
            "Content-Type": "application/json",
            "X-Shopify-Storefront-Access-Token": self.storefront_token
        }
        
        print("---" * 20)
        print(f"🚀 SENDING SHOPIFY QUERY:\n{query}")
        print("---" * 20)
        
        try:
            resp = await self.http_client.post(url, headers=headers, json={"query": graphql_query, "variables": variables})
            resp.raise_for_status()
            data = resp.json()
            return data.get("data", {}).get("products", {}).get("edges", [])
        except httpx.HTTPStatusError as e:
            print(f"❌ HTTP ERROR: {e.response.status_code}")
            print(f"📄 RESPONSE: {e.response.text}")
            return []
        except Exception as e:
            print(f"❌ GENERAL ERROR: {str(e)}")
            return []

async def main():
    """Main function to run the test"""
    store_url = os.getenv("SHOPIFY_STORE_URL")
    storefront_token = os.getenv("SHOPIFY_STOREFRONT_ACCESS_TOKEN")

    if not store_url or not storefront_token:
        print("❌ ERROR: Please ensure SHOPIFY_STORE_URL and SHOPIFY_STOREFRONT_ACCESS_TOKEN are in your .env file.")
        return

    shopify_service = ShopifyService(store_url, storefront_token)
    query_builder = QueryBuilder(SearchConfig())
    
    while True:
        # Get search term from user
        search_term = input("\n👉 Enter search term (or 'exit' to quit): ")
        if search_term.lower() == 'exit':
            break

        # Build the query
        final_query = query_builder.build_query(search_term)
        
        if not final_query:
            print("No valid keywords extracted. Please try another term.")
            continue

        # Get products
        products = await shopify_service.get_products(final_query)

        print("\n" + "---" * 20)
        if products:
            print(f"✅ FOUND {len(products)} PRODUCTS:")
            for i, product_edge in enumerate(products):
                product = product_edge.get('node', {})
                print(f"  {i+1}. {product.get('title')}")
                print(f"     Tags: {product.get('tags')}")
        else:
            print("❌ NO PRODUCTS FOUND FOR THIS QUERY.")
        print("---" * 20)


if __name__ == "__main__":
    asyncio.run(main())