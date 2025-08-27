# /app/services/order_service.py

import re
import json
import uuid
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from contextlib import asynccontextmanager
from functools import wraps
from dataclasses import dataclass, field
from enum import Enum
import traceback

from app.config.settings import settings
from app.config import strings
from app.config.rules import JewelryRulesEngine
from app.models.domain import Product
from app.services.security_service import EnhancedSecurityService
from app.services.ai_service import ai_service
from app.services.shopify_service import shopify_service
from app.services.whatsapp_service import whatsapp_service
from app.services.db_service import db_service
from app.services.cache_service import cache_service
from app.utils.queue import message_queue
from app.utils.metrics import message_counter, active_customers_gauge, response_time_histogram
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)
scheduler = AsyncIOScheduler()
rules_engine = JewelryRulesEngine()

# --- Performance and reliability decorators ---

def with_metrics(func_name: str):
    """Decorator to track function performance metrics"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            with response_time_histogram.labels(endpoint=func_name).time():
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def with_error_handling(fallback_message: str = "Something went wrong. Our team has been notified."):
    """Decorator for consistent error handling"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in {func.__name__}: {e}", exc_info=True)

                # Extract phone number from ConversationContext or args
                phone_number = None
                if args and isinstance(args[0], ConversationContext):
                    phone_number = args[0].phone_number
                elif args and isinstance(args[0], dict) and 'phone_number' in args[0]:
                    phone_number = args[0]['phone_number']
                elif 'phone_number' in kwargs:
                    phone_number = kwargs['phone_number']

                if phone_number:
                    await whatsapp_service.send_message(phone_number, fallback_message)
                return fallback_message
        return wrapper
    return decorator

# Enhanced configuration constants
class CacheConfig:
    CUSTOMER_TTL = 1800  # 30 minutes
    PRODUCT_STATE_TTL = 900  # 15 minutes
    SEARCH_STATE_TTL = 1800  # 30 minutes
    CONVERSATION_CONTEXT_TTL = 3600  # 1 hour

class MessageLimits:
    MAX_RESPONSE_LENGTH = 4096
    MAX_PRODUCTS_PER_CARD = 5
    MAX_VARIANTS_DISPLAY = 5
    MAX_ORDERS_DISPLAY = 3
    MAX_CONVERSATION_HISTORY = 10

class ErrorMessages:
    PRODUCT_NOT_FOUND = "Sorry, that product is no longer available."
    SEARCH_ERROR = "I'm having trouble searching right now. Please try again in a moment."
    AI_ERROR = "I'm having trouble understanding that right now. Could you rephrase?"
    GENERAL_ERROR = "Something went wrong. Our team has been notified."

class JourneyStage(Enum):
    NEW = "new"
    BROWSING = "browsing"
    CONSIDERING = "considering"
    PURCHASING = "purchasing"
    LOYAL = "loyal"

# Enhanced Context Management
@dataclass
class ConversationContext:
    """Consolidated context object for handler functions"""
    phone_number: str
    customer: Dict
    analysis: Dict
    original_message: str
    message_type: str = "text"
    quoted_wamid: Optional[str] = None
    profile_name: Optional[str] = None

    # Computed properties
    @property
    def intent(self) -> str:
        return self.analysis.get("primary_intent", "unknown")

    @property
    def confidence(self) -> float:
        return self.analysis.get("confidence", 0.0)

    @property
    def entities(self) -> Dict:
        return self.analysis.get("entities", {})

    @property
    def user_context(self) -> Dict:
        return self.analysis.get("user_context", {})

    @property
    def context_data(self) -> Dict:
        return self.analysis.get("context", {})

    @property
    def customer_name(self) -> str:
        return self.customer.get("name", "").strip()

    @property
    def is_new_customer(self) -> bool:
        return len(self.customer.get("conversation_history", [])) < 3

    @property
    def favorite_categories(self) -> List[str]:
        return self.customer.get("favorite_categories", [])

# Enhanced Router with State Machine Pattern
class MessageRouter:
    """Enhanced message router with better organization and state management"""

    def __init__(self):
        self.high_confidence_handlers = {
            "human_escalation": (self._handle_human_escalation, 0.7),
            "urgent_request": (self._handle_urgent_request, 0.7),
            "purchase_intent": (self._handle_purchase_intent, 0.6),
            "latest_arrivals_inquiry": (self._handle_latest_arrivals, 0.8),
            "bestseller_inquiry": (self._handle_bestsellers, 0.8),
        }

        self.medium_confidence_handlers = {
            "complaint_inquiry": (self._handle_complaint_inquiry, 0.6), # Add this line
            "reseller_inquiry": (self._handle_reseller_inquiry, 0.7),      # Add this
            "bulk_order_inquiry": (self._handle_bulk_order_inquiry, 0.7),
            "order_inquiry": (self._handle_order_inquiry, 0.5),
            "support": (self._handle_support_request, 0.5),
            "shipping_inquiry": (self._handle_shipping_inquiry, 0.5),
            "price_inquiry": (self._handle_price_inquiry, 0.4),
            "size_inquiry": (self._handle_size_inquiry, 0.4),
            "product_search": (self._handle_product_search, 0.4),
            "contact_inquiry": (self._handle_contact_inquiry, 0.4),
            "discount_inquiry": (self._handle_discount_inquiry, 0.4),
            "stock_inquiry": (self._handle_stock_inquiry, 0.4),
            "more_results": (self._handle_more_results, 0.3),
        }

        self.interactive_handlers = {
            "purchase_action": self._handle_interactive_purchase,
            "show_more_action": self._handle_interactive_more_info,
            "similar_products_action": self._handle_interactive_similar,
            "product_option_action": self._handle_interactive_option,
            "product_detail_view": self._handle_product_detail,
        }

        self.context_handlers = {
            "casual_greeting": self._handle_casual_greeting,
            "business_greeting": self._handle_business_greeting,
            "gratitude": self._handle_thank_you,
            "affirmative_response": self._handle_affirmative_response,
            "negative_response": self._handle_negative_response,
        }

    async def route_message(self, ctx: ConversationContext) -> Optional[str]:
        """Main routing logic with corrected priority and smarter contextual replies."""

        # Handle special message types first
        if ctx.message_type == "image":
            return await self._handle_visual_search(ctx)

        # Try high confidence handlers (like latest arrivals, bestsellers)
        result = await self._try_handlers(self.high_confidence_handlers, ctx)
        if result:
            return result

        # Try interactive handlers (product_detail_view, buy_now, etc.)
        if ctx.intent in self.interactive_handlers:
            handler = self.interactive_handlers[ctx.intent]
            return await handler(ctx)

        # Handle context-dependent responses (generic replies)
        if ctx.quoted_wamid:
            # First, check if the reply is to a single product detail view
            last_single_product_raw = await cache_service.get(f"state:last_single_product:{ctx.phone_number}")
            if last_single_product_raw:
                return await self._handle_contextual_product_question(ctx, last_single_product_raw)

            # Next, check if the reply is to a list of products
            last_product_list_raw = await cache_service.get(f"state:last_product_list:{ctx.phone_number}")
            if last_product_list_raw:
                return await self._handle_contextual_list_question(ctx)

        # Try medium confidence handlers
        result = await self._try_handlers(self.medium_confidence_handlers, ctx)
        if result:
            return result

        # If any product-related entities are found, force a product search
        if any(ctx.entities.get(category) for category in ["products", "materials", "gemstones", "styles"]):
            logger.info(f"Entities found. Forcing call to _handle_product_search for message: '{ctx.original_message}'")
            return await self._handle_product_search(ctx)

        # Handle conversational context
        conversational_context = ctx.analysis.get("conversational_context")
        if conversational_context and conversational_context in self.context_handlers:
            handler = self.context_handlers[conversational_context]
            return await handler(ctx)

        # Smart fallback based on user context
        return await self._handle_smart_fallback(ctx)


    async def _try_handlers(self, handlers: Dict, ctx: ConversationContext) -> Optional[str]:
        """Try handlers from a dictionary with confidence thresholds"""
        for handler_intent, (handler, min_confidence) in handlers.items():
            if ctx.intent == handler_intent and ctx.confidence >= min_confidence:
                return await handler(ctx)
        return None

    # Handler methods
    @with_error_handling()
    async def _handle_product_search(self, ctx: ConversationContext) -> Optional[str]:
        """
        MODIFIED: More aggressive product search that prioritizes showing results 
        over asking questions.
        """
        # Combine all entities and the original message for a broader search query.
        entities = ctx.entities
        all_keywords = (
            entities.get("products", []) +
            entities.get("materials", []) +
            entities.get("gemstones", []) +
            entities.get("styles", [])
        )

        # If specific keywords are found, build a targeted query.
        if all_keywords:
            # Storefront API uses a simpler search query format.
            final_query = " ".join(set(all_keywords))
        else:
            # If no entities are extracted, use the user's raw message as the query..
            # This is the key change to prevent the bot from asking questions.
            final_query = ctx.original_message

        logger.info(f"Executing product search with query: '{final_query}'")

        try:
            products, total_count = await shopify_service.get_products(
                query=final_query,
                limit=MessageLimits.MAX_PRODUCTS_PER_CARD,
                sort_key="RELEVANCE"
            )

            # If the specific search yields no results, try a broader search using OR
            if not products and 'AND' in final_query:
                broad_query = final_query.replace(" AND ", " OR ")
                logger.info(f"No results with AND, trying broad query: '{broad_query}'")
                products, total_count = await shopify_service.get_products(
                    query=broad_query,
                    limit=MessageLimits.MAX_PRODUCTS_PER_CARD,
                    sort_key="RELEVANCE"
                )

            if not products:
                # IMPORTANT: Only if there are NO results should we ask for clarification.
                return await self._handle_unclear_search_request(ctx)

            # Update user preferences based on successful search
            asyncio.create_task(update_user_search_preferences(ctx.customer, ctx.entities))

            # Send results
            await send_enhanced_product_results(products, ctx, total_count)
            return None

        except Exception as e:
            logger.error(f"Product search error: {e}")
            return ErrorMessages.SEARCH_ERROR



    async def _handle_purchase_intent(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced purchase intent handling with context awareness"""
        # Check if user is referring to a specific product
        last_product_raw = await cache_service.get(f"state:last_single_product:{ctx.phone_number}")
        if last_product_raw:
            try:
                last_product = Product.parse_raw(last_product_raw)
                return await handle_buy_request(last_product.id, ctx)
            except Exception as e:
                logger.error(f"Error parsing last product: {e}")

        # Check if user is referring to products from last search
        last_products_raw = await cache_service.get(f"state:last_product_list:{ctx.phone_number}")
        if last_products_raw:
            try:
                products_data = json.loads(last_products_raw)
                if len(products_data) == 1:
                    return await handle_buy_request(products_data[0]["id"], ctx)
                else:
                    return "Which specific product would you like to purchase? Please tap on the product you want to buy."
            except Exception as e:
                logger.error(f"Error parsing last product list: {e}")

        # No specific product context, treat as search with buying intent
        if any(ctx.entities.get(category) for category in ["products", "materials", "gemstones"]):
            await cache_service.set(f"state:buying_intent:{ctx.phone_number}", "true", ttl=CacheConfig.PRODUCT_STATE_TTL)
            return await self._handle_product_search(ctx)

        return "I'd love to help you make a purchase! What type of jewelry are you looking for?"

    async def _handle_latest_arrivals(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced latest arrivals with personalization"""
        try:
            products, total_count = await shopify_service.get_products(
                query="",
                limit=MessageLimits.MAX_PRODUCTS_PER_CARD,
                sort_key="CREATED_AT"
            )

            if not products:
                return "I couldn't fetch the latest arrivals right now. Please try again in a moment!"

            # Mix in personalized results if user has preferences
            if ctx.favorite_categories:
                try:
                    pref_query = " OR ".join([f"tag:{cat}" for cat in ctx.favorite_categories[:3]])
                    pref_products, _ = await shopify_service.get_products(
                        query=pref_query, limit=3, sort_key="CREATED_AT"
                    )
                    if pref_products:
                        all_product_ids = {p.id for p in products}
                        unique_pref_products = [p for p in pref_products if p.id not in all_product_ids]
                        products = unique_pref_products[:2] + products[:8]
                except Exception as e:
                    logger.warning(f"Error adding personalized latest arrivals: {e}")

            personalized_greeting = f"Hi {ctx.customer_name}! " if ctx.customer_name else ""
            header_text = f"{personalized_greeting}Here are our latest arrivals! ✨"
            body_text = "Fresh designs, just added to our collection 🆕"

            await send_product_card(products, ctx, header_text, body_text)
            return None

        except Exception as e:
            logger.error(f"Error in enhanced latest arrivals: {e}")
            return "I'm having trouble fetching our latest arrivals right now. Please try again!"

    async def _handle_bestsellers(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced bestsellers with social proof"""
        try:
            products, _ = await shopify_service.get_products(
                query="",
                limit=MessageLimits.MAX_PRODUCTS_PER_CARD,
                sort_key="BEST_SELLING"
            )

            if not products:
                return "I couldn't fetch our bestsellers right now. Please try again in a moment!"

            header_text = "Our customers' top picks! 🌟"
            body_text = "These are the pieces everyone's talking about ✨"

            await send_product_card(products, ctx, header_text, body_text)
            return None

        except Exception as e:
            logger.error(f"Error in enhanced bestsellers: {e}")
            return "I'm having trouble fetching our bestsellers right now. Please try again!"

    async def _handle_order_inquiry(self, ctx: ConversationContext) -> str:
        """Enhanced order inquiry with better formatting"""
        try:
            orders = await shopify_service.search_orders_by_phone(ctx.phone_number)

            if not orders:
                return strings.NO_ORDERS_FOUND + "\n\nWould you like to browse our latest collection instead?"

            return format_orders_response_enhanced(orders, ctx.customer)

        except Exception as e:
            logger.error(f"Error in order inquiry: {e}")
            return "I'm having trouble accessing your order information right now. Please try again or contact our support team."

    async def _handle_support_request(self, ctx: ConversationContext) -> str:
        return strings.SUPPORT_GENERAL_RESPONSE + "\n\nHow can I assist you today?"
    
    async def _handle_complaint_inquiry(self, ctx: ConversationContext) -> str:
        # This uses the specific, detailed string for complaints
        return strings.SUPPORT_COMPLAINT_RESPONSE

    async def _handle_human_escalation(self, ctx: ConversationContext) -> str:
        return strings.HUMAN_ESCALATION + "\n\nOur team will get back to you as soon as possible! 🙏"

    async def _handle_urgent_request(self, ctx: ConversationContext) -> str:
        return strings.HUMAN_ESCALATION + "\n\n⚡ *URGENT REQUEST FLAGGED* - Our team has been notified!"

    async def _handle_shipping_inquiry(self, ctx: ConversationContext) -> str:
        return ("📦 *Shipping Information:*\n\n"
               "• Free shipping on orders above ₹999\n"
               "• Standard delivery: 3-5 business days\n"
               "• Express delivery: 1-2 business days\n\n"
               "Which specific shipping details do you need help with?")

    async def _handle_contact_inquiry(self, ctx: ConversationContext) -> str:
        return ("📞 *Contact Information:*\n\n"
               "• WhatsApp: This number\n"
               "• Email: support@feelori.com\n"
               "• Phone: +91-XXXXXXXXXX\n\n"
               "How can we help you today?")
    async def _handle_reseller_inquiry(self, ctx: ConversationContext) -> str:
        return strings.RESELLER_INFO

    async def _handle_bulk_order_inquiry(self, ctx: ConversationContext) -> str:
        return strings.BULK_ORDER_INFO

    async def _handle_discount_inquiry(self, ctx: ConversationContext) -> str:
        return ("🎉 *Current Offers:*\n\n"
               "• Free shipping on orders above ₹999\n"
               "• First-time buyers get 10% off\n"
               "• Check our latest collection for seasonal discounts\n\n"
               "What are you looking to purchase?")

    async def _handle_stock_inquiry(self, ctx: ConversationContext) -> str:
        return ("📦 *Stock Information:*\n\n"
               "Are you asking about a specific product? "
               "Please share the product name or show me what you're interested in, "
               "and I'll check availability for you!")

    async def _handle_price_inquiry(self, ctx: ConversationContext) -> str:
        return ("💰 I'd love to help with pricing! \n\n"
               "Are you asking about:\n"
               "• A specific product you're viewing?\n"
               "• Price range for a category?\n"
               "• Our current offers?\n\n"
               "Let me know and I'll get you the details! ✨")

    async def _handle_size_inquiry(self, ctx: ConversationContext) -> str:
        return ("📏 *Size Information:*\n\n"
               "I can help you with sizing! Are you asking about:\n"
               "• Ring sizes?\n"
               "• Chain lengths?\n"
               "• Bangle/bracelet sizes?\n\n"
               "Which product are you interested in? I'll provide specific measurements! 📐")

    async def _handle_more_results(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced more results with pagination"""
        try:
            last_search_raw = await cache_service.get(f"state:last_search:{ctx.phone_number}")
            if not last_search_raw:
                return "More of what? Please search for a product first, and I'll show you additional options!"

            last_search = json.loads(last_search_raw)
            query = last_search.get("query", "")
            current_page = last_search.get("page", 1)

            # Get next page of results
            new_analysis = rules_engine.process_message(query)
            entities = new_analysis.get("entities", {})

            # Build query similar to original search
            query_parts = []
            for category in ["products", "materials", "styles", "gemstones"]:
                items = entities.get(category, [])
                if items:
                    category_query = " OR ".join([f"tag:{item}" for item in items])
                    query_parts.append(f"({category_query})")

            final_query = " OR ".join(query_parts) if query_parts else ""

            # Get products with offset
            offset = current_page * MessageLimits.MAX_PRODUCTS_PER_CARD
            products, total_count = await shopify_service.get_products(
                query=final_query,
                limit=MessageLimits.MAX_PRODUCTS_PER_CARD,
                offset=offset
            )

            if not products:
                return "That's all I have for now! Would you like to try a different search?"

            # Update search state
            last_search["page"] = current_page + 1
            await cache_service.set(f"state:last_search:{ctx.phone_number}", json.dumps(last_search), ttl=CacheConfig.SEARCH_STATE_TTL)

            header_text = f"Here are {len(products)} more options! ✨"
            body_text = f"Page {current_page + 1} of results"

            await send_product_card(products, ctx, header_text, body_text)
            return None

        except Exception as e:
            logger.error(f"Error in more results: {e}")
            return "I'm having trouble getting more results. Please try your search again!"

    # Interactive handlers
    async def _handle_interactive_purchase(self, ctx: ConversationContext) -> Optional[str]:
        product_id = ctx.original_message.replace("buy_", "")
        return await handle_buy_request(product_id, ctx)

    async def _handle_interactive_more_info(self, ctx: ConversationContext) -> Optional[str]:
        """Handle 'more info' interactive button responses - MODIFIED FOR SAFETY"""
        product_id = ctx.original_message.replace("more_", "")
        product = await shopify_service.get_product_by_id(product_id)

        if not product:
            return ErrorMessages.PRODUCT_NOT_FOUND

        # Cache the product for follow-up questions
        await cache_service.set(
            f"state:last_single_product:{ctx.phone_number}",
            product.json(),
            ttl=CacheConfig.PRODUCT_STATE_TTL
        )

        # Create detailed response
        details = [f"*{product.title}*\n"]

        if product.description:
            clean_desc = re.sub(r'<[^>]+>', '', product.description)
            if len(clean_desc) > 300:
                clean_desc = clean_desc[:300] + "..."
            details.append(f"📝 *Description:*\n{clean_desc}\n")

        # --- START OF THE FIX ---
        # Safely check if the 'variants' attribute and its properties exist before using them
        if hasattr(product, 'variants') and product.variants:
            variant = product.variants[0]
            if hasattr(variant, 'price') and variant.price:
                details.append(f"💰 *Price:* ₹{variant.price}")

            if hasattr(variant, 'available') and variant.available is False:
                details.append("❌ *Currently out of stock*")
            else:
                details.append("✅ *Available now*")
        # --- END OF THE FIX ---

        if product.tags:
            relevant_tags = [tag for tag in product.tags[:5] if len(tag) > 2]
            if relevant_tags:
                details.append(f"🏷️ *Features:* {', '.join(relevant_tags)}")

        product_url = shopify_service.get_product_page_url(product.handle)
        details.append(f"\n🔗 View full details: {product_url}")

        return "\n".join(details)

    async def _handle_interactive_similar(self, ctx: ConversationContext) -> Optional[str]:
        """Handle 'similar products' interactive button responses"""
        product_id = ctx.original_message.replace("similar_", "")
        product = await shopify_service.get_product_by_id(product_id)

        if not product:
            return "I couldn't find that product to show similar items."

        # Use product tags to find similar items
        if product.tags:
            tag_query = " OR ".join([f"tag:{tag}" for tag in product.tags[:3]])

            try:
                similar_products, _ = await shopify_service.get_products(
                    query=tag_query,
                    limit=MessageLimits.MAX_PRODUCTS_PER_CARD + 1
                )

                # Remove the original product from results
                similar_products = [p for p in similar_products if p.id != product.id][:MessageLimits.MAX_PRODUCTS_PER_CARD]

                if similar_products:
                    truncated_title = (product.title[:35] + '...') if len(product.title) > 35 else product.title
                    header_text = f"Items similar to {truncated_title} ✨"
                    body_text = "You might also love these pieces!"
                    await send_product_card(similar_products, ctx, header_text, body_text)
                    return None

            except Exception as e:
                logger.error(f"Error finding similar products: {e}")

        return f"I'd love to show you similar items to *{product.title}*! What specific aspects are you looking for? (style, material, price range, etc.)"

    async def _handle_interactive_option(self, ctx: ConversationContext) -> Optional[str]:
        """Handle product option selection from interactive buttons"""
        variant_id = ctx.original_message.replace("option_", "")

        try:
            variant = await shopify_service.get_variant_by_id(variant_id)
            if not variant:
                return "Sorry, that option is no longer available."

            if not variant.get("available", True):
                return f"Sorry, that option is currently out of stock. Would you like to see other available options?"

            cart_url = shopify_service.get_add_to_cart_url(variant_id)

            variant_title = variant.get("title", "Selected option")
            product_title = variant.get("product_title", "this item")
            price = variant.get("price", "Price unavailable")

            return (f"Perfect choice! 🛍️\n\n"
                   f"*{product_title}*\n"
                   f"Option: {variant_title}\n"
                   f"💰 Price: ₹{price}\n\n"
                   f"👆 Add to cart and checkout:\n{cart_url}")

        except Exception as e:
            logger.error(f"Error handling option selection: {e}")
            return "I'm having trouble processing that selection. Please try again or contact support."

    async def _handle_product_detail(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced product detail view with rich information"""
        product_id = ctx.original_message.replace("product_", "")
        product = await shopify_service.get_product_by_id(product_id)

        if not product:
            return ErrorMessages.PRODUCT_NOT_FOUND

        # Cache for follow-up questions
        await cache_service.set(f"state:last_single_product:{ctx.phone_number}", product.json(), ttl=CacheConfig.PRODUCT_STATE_TTL)

        # Send enhanced product detail card
        try:
            await whatsapp_service.send_product_detail_with_buttons(ctx.phone_number, product)
            return None
        except Exception as e:
            logger.error(f"Error sending product detail card: {e}")
            return await self._handle_interactive_more_info(ctx)

    # Context handlers
    async def _handle_casual_greeting(self, ctx: ConversationContext) -> Optional[str]:
        return await handle_greeting(ctx, is_casual=True)

    async def _handle_business_greeting(self, ctx: ConversationContext) -> Optional[str]:
        return await handle_greeting(ctx, is_casual=False)

    async def _handle_thank_you(self, ctx: ConversationContext) -> str:
        name_part = f"{ctx.customer_name}! " if ctx.customer_name else "! "
        return f"You're so welcome{name_part}😊 Is there anything else I can help you find today?"

    async def _handle_affirmative_response(self, ctx: ConversationContext) -> Optional[str]:
        return "Great! How can I help you today?"

    async def _handle_negative_response(self, ctx: ConversationContext) -> Optional[str]:
        return "No problem! Is there something else I can help you with?"

    # Helper methods for complex handlers
    async def _handle_visual_search(self, ctx: ConversationContext) -> Optional[str]:
        """Handle image-based product search"""
        if not ctx.original_message.startswith("visual_search_"):
            return "I can see you sent an image, but I'm having trouble processing it right now."

        media_id = ctx.original_message.replace("visual_search_", "")

        await whatsapp_service.send_message(
            ctx.phone_number,
            "I can see your image! 📸 While I'm learning to recognize jewelry in photos, "
            "could you describe what type of piece you're looking for? For example: "
            "'gold necklace like this' or 'similar earrings'."
        )

        return None

    async def _handle_follow_up_response(self, ctx: ConversationContext) -> Optional[str]:
        """Handle follow-up responses based on previous bot questions"""
        last_question = ctx.context_data.get("last_bot_question")

        follow_up_handlers = {
            "offer_bestsellers": self._handle_bestsellers_follow_up,
            "offer_alternatives": self._handle_alternatives_follow_up,
            "size_clarification": self._handle_size_clarification_follow_up,
            "budget_inquiry": self._handle_budget_follow_up,
        }

        handler = follow_up_handlers.get(last_question)
        if handler:
            await cache_service.delete(f"state:last_bot_question:{ctx.phone_number}")
            return await handler(ctx)

        return None

    async def _handle_contextual_product_question(
        self,
        ctx: ConversationContext,
        last_product_raw: str
    ) -> Optional[str]:
        """Handle questions about previously shown products - MODIFIED FOR SAFETY"""
        try:
            last_product = Product.parse_raw(last_product_raw)

            # Analyze the question about the product
            if any(word in ctx.original_message.lower() for word in ["buy", "purchase", "cart", "order"]):
                return await handle_buy_request(last_product.id, ctx)

            # --- START OF THE FIX ---
            elif any(word in ctx.original_message.lower() for word in ["price", "cost", "much"]):
                # Safely check if variants and price exist on the cached product object
                if hasattr(last_product, 'variants') and last_product.variants and hasattr(last_product.variants[0], 'price'):
                    price = last_product.variants[0].price
                    return f"*{last_product.title}* is priced at ₹{price}. Would you like to purchase it?"
                else:
                    # If price is not in the cached object, provide a helpful link instead of crashing.
                    product_url = shopify_service.get_product_page_url(last_product.handle)
                    return f"You can find the most up-to-date pricing for *{last_product.title}* right here: {product_url}"
            # --- END OF THE FIX ---

            elif any(word in ctx.original_message.lower() for word in ["size", "dimensions", "length"]):
                return f"For sizing details of *{last_product.title}*, please check the product page or ask our team for specific measurements!"

            elif any(word in ctx.original_message.lower() for word in ["similar", "like", "other"]):
                # Create a new context object to pass to the similar handler
                similar_ctx = ConversationContext(
                    phone_number=ctx.phone_number,
                    customer=ctx.customer,
                    analysis=ctx.analysis,
                    original_message=f"similar_{last_product.id}"
                )
                return await self._handle_interactive_similar(similar_ctx)

            # Generic response for other product-related questions
            return f"I can help you with *{last_product.title}*! What would you like to know specifically?"

        except Exception as e:
            logger.error(f"Error handling contextual product question: {e}", exc_info=True)
            return "I'm having trouble with that question. Could you be more specific?"

    async def _handle_contextual_list_question(self, ctx: ConversationContext) -> Optional[str]:
        """Handles generic questions asked in reply to a product list."""
        return "I've provided a few options above. To see more details like the price for a specific item, please tap the 'View Details' button on the product you're interested in."

    async def _handle_unclear_search_request(self, ctx: ConversationContext) -> Optional[str]:
        """
        MODIFIED: Handles cases where a search returns no results by offering 
        guided categories instead of open-ended questions.
        """
        # Offer quick reply buttons for major categories.
        categories = {
            "💍 Rings": "rings",
            "📿 Necklaces": "necklaces",
            "💎 Earrings": "earrings",
            "⭐ Bestsellers": "bestsellers"
        }
        await whatsapp_service.send_quick_replies(
            ctx.phone_number,
            f"I couldn't find any results for '{ctx.original_message}'. 😔\n\nPerhaps you'd like to browse one of our popular categories?",
            categories
        )
        return None

    

    async def _handle_no_search_results(self, ctx: ConversationContext) -> Optional[str]:
        """Enhanced no results handling with smart suggestions"""
        products = ctx.entities.get("products", [])
        materials = ctx.entities.get("materials", [])
        styles = ctx.entities.get("styles", [])

        # Build helpful alternative suggestions
        suggestions = []
        if products:
            suggestions.append("✨ Check our bestsellers")
            suggestions.append(f"🔍 Browse all {products[0]} styles")

        if materials:
            suggestions.append(f"💎 See our {materials[0]} collection")

        if not suggestions:
            suggestions = ["✨ View our latest arrivals", "⭐ Check our bestsellers", "💎 Browse all categories"]

        response = f"I couldn't find exactly what you're looking for with '{ctx.original_message}' 😔\n\nHere are some alternatives:\n"
        response += "\n".join([f"• {suggestion}" for suggestion in suggestions[:3]])

        # Set up context for follow-up
        await cache_service.set(f"state:last_bot_question:{ctx.phone_number}", "offer_alternatives", ttl=CacheConfig.PRODUCT_STATE_TTL)

        await whatsapp_service.send_message(ctx.phone_number, response)
        return None

    async def _handle_smart_fallback(self, ctx: ConversationContext) -> str:
        """Smart fallback that considers user context and message content"""
        # For new users, be more guided
        if ctx.is_new_customer:
            return await self._handle_new_user_fallback(ctx)

        # For experienced users, be more flexible
        return await self._handle_experienced_user_fallback(ctx)

    async def _handle_new_user_fallback(self, ctx: ConversationContext) -> Optional[str]:
        """Fallback for new users with guided experience"""
        categories = {
            "💍 Rings": "rings",
            "📿 Necklaces": "necklaces",
            "💎 Earrings": "earrings",
            "🔗 Bracelets": "bracelets",
            "⭐ Bestsellers": "bestsellers"
        }

        await whatsapp_service.send_quick_replies(
            ctx.phone_number,
            f"Welcome to FeelOri! 🌟 I'd love to help you find beautiful jewelry. What catches your eye?",
            categories
        )

        return None

    async def _handle_experienced_user_fallback(self, ctx: ConversationContext) -> str:
        """Fallback for experienced users using AI"""
        try:
            context = {
                "conversation_history": ctx.customer.get("conversation_history", [])[-MessageLimits.MAX_CONVERSATION_HISTORY:],
                "preferences": ctx.customer.get("preferences", {}),
                "favorite_categories": ctx.favorite_categories
            }
            return await ai_service.generate_response(ctx.original_message, context)
        except Exception as e:
            logger.error(f"AI fallback error: {e}")
            return "I didn't quite catch that. Could you tell me what type of jewelry you're interested in?"

    # Follow-up handlers
    async def _handle_bestsellers_follow_up(self, ctx: ConversationContext) -> Optional[str]:
        """Handle follow-up when user was offered bestsellers"""
        if ctx.analysis.get("conversational_context") == "affirmative_response":
            return await self._handle_bestsellers(ctx)
        else:
            return "No problem! What specific type of jewelry are you looking for?"

    async def _handle_alternatives_follow_up(self, ctx: ConversationContext) -> Optional[str]:
        """Handle follow-up when user was offered alternatives"""
        # Try to extract new search intent from their response
        if any(ctx.entities.get(category) for category in ["products", "materials", "styles", "gemstones"]):
            return await self._handle_product_search(ctx)

        return "I'd be happy to help you find something! Could you describe what you're looking for?"

    async def _handle_size_clarification_follow_up(self, ctx: ConversationContext) -> Optional[str]:
        """Handle size clarification follow-up"""
        # Extract size information from the response
        size_keywords = ["small", "medium", "large", "xs", "s", "m", "l", "xl", "6", "7", "8", "9", "10"]
        found_sizes = [word for word in ctx.original_message.lower().split() if word in size_keywords]

        if found_sizes:
            return f"Perfect! I've noted you prefer size {found_sizes[0]}. Let me show you options in that size."

        return "Could you please specify the size you're looking for? (e.g., small, medium, large, or specific measurements)"

    async def _handle_budget_follow_up(self, ctx: ConversationContext) -> Optional[str]:
        """Handle budget inquiry follow-up"""
        # Extract price/budget information
        price_matches = re.findall(r'₹?(\d+)', ctx.original_message)
        if price_matches:
            budget = price_matches[0]
            return f"Great! I'll show you beautiful pieces under ₹{budget}. What type of jewelry are you interested in?"

        return "What's your preferred price range? This will help me show you the best options!"


# Global router instance
message_router = MessageRouter()


@asynccontextmanager
async def conversation_context(phone_number: str):
    """Context manager for handling conversation state"""
    try:
        # Update last interaction
        await cache_service.set(
            f"last_interaction:{phone_number}",
            datetime.utcnow().isoformat(),
            ttl=CacheConfig.CONVERSATION_CONTEXT_TTL
        )
        yield
    finally:
        # Cleanup or final state updates can go here
        pass

# --- Enhanced Core Message Processing ---

@with_metrics("process_message")
@with_error_handling()
async def process_message(phone_number: str, message_text: str, message_type: str, quoted_wamid: str | None) -> str | None:
    """
    Enhanced message processing with better error handling and performance tracking.
    """
    clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)

    # Early return for packing department
    if clean_phone == settings.packing_dept_whatsapp_number:
        return strings.PACKING_DEPT_REDIRECT

    async with conversation_context(clean_phone):
        message_counter.labels(status="received", message_type=message_type).inc()

        customer = await get_or_create_customer(clean_phone)

        # Enhanced analysis with context
        analysis = await analyze_message_with_context(message_text, customer, clean_phone)
        logger.info(f"Enhanced analysis for {clean_phone}: {analysis}")

        # Create conversation context object
        ctx = ConversationContext(
            phone_number=clean_phone,
            customer=customer,
            analysis=analysis,
            original_message=message_text,
            message_type=message_type,
            quoted_wamid=quoted_wamid
        )

        # Route with enhanced context awareness
        response = await message_router.route_message(ctx)

        # Update conversation history asynchronously
        asyncio.create_task(update_conversation_history(customer, message_text, response))

        return response[:MessageLimits.MAX_RESPONSE_LENGTH] if response else None

async def analyze_message_with_context(message_text: str, customer: Dict, phone_number: str) -> Dict:
    """Enhanced message analysis that includes conversation context"""
    # Get base analysis from rules engine
    analysis = rules_engine.process_message(message_text)

    # Enhance with conversation context
    conversation_history = customer.get("conversation_history", [])[-5:]
    recent_intents = [msg.get("intent") for msg in conversation_history if msg.get("intent")]

    # Check for context-dependent intents
    last_bot_question = await cache_service.get(f"state:last_bot_question:{phone_number}")
    if last_bot_question:
        analysis["context"] = {
            "last_bot_question": last_bot_question,
            "is_follow_up": True
        }

    # Add user preference context
    preferences = customer.get("preferences", {})
    analysis["user_context"] = {
        "favorite_categories": customer.get("favorite_categories", []),
        "recent_intents": recent_intents,
        "preferences": preferences,
        "total_messages": customer.get("total_messages", 0)
    }

    return analysis

@with_error_handling()
async def get_or_create_customer(phone_number: str) -> Dict:
    """Enhanced customer retrieval with better caching and error handling"""
    cache_key = f"customer:v3:{phone_number}"  # Updated version

    # Try cache first
    try:
        cached_customer_raw = await cache_service.get(cache_key)
        if cached_customer_raw:
            customer = json.loads(cached_customer_raw)
            # Validate customer data structure
            if _validate_customer_data(customer):
                return customer
            else:
                logger.warning(f"Invalid customer data in cache for {phone_number}, refreshing...")
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"Failed to decode customer from cache for {phone_number}: {e}")

    # Fetch from database
    customer_data = await db_service.get_customer(phone_number)
    if customer_data:
        # Ensure data structure is complete
        customer_data = _normalize_customer_data(customer_data)
        await cache_service.set(cache_key, json.dumps(customer_data, default=str), ttl=CacheConfig.CUSTOMER_TTL)
        return customer_data

    # Create new customer
    new_customer = _create_new_customer(phone_number)
    await db_service.create_customer(new_customer)
    await cache_service.set(cache_key, json.dumps(new_customer, default=str), ttl=CacheConfig.CUSTOMER_TTL)
    active_customers_gauge.inc()
    return new_customer

def _validate_customer_data(customer: Dict) -> bool:
    """Validates customer data structure"""
    required_fields = ['id', 'phone_number', 'created_at', 'conversation_history', 'preferences']
    return all(field in customer for field in required_fields)

def _normalize_customer_data(customer: Dict) -> Dict:
    """Ensures customer data has all required fields with defaults"""
    defaults = {
        'conversation_history': [],
        'preferences': {},
        'favorite_categories': [],
        'total_messages': 0,
        'last_interaction': datetime.utcnow(),
        'journey_stage': JourneyStage.BROWSING.value,
        'engagement_score': 0.0
    }

    for key, default_value in defaults.items():
        if key not in customer:
            customer[key] = default_value

    return customer

def _create_new_customer(phone_number: str) -> Dict:
    """Creates a new customer with complete data structure"""
    return {
        "id": str(uuid.uuid4()),
        "phone_number": phone_number,
        "created_at": datetime.utcnow(),
        "conversation_history": [],
        "preferences": {},
        "last_interaction": datetime.utcnow(),
        "total_messages": 0,
        "favorite_categories": [],
        "journey_stage": JourneyStage.NEW.value,
        "engagement_score": 0.0
    }

# --- Enhanced Product and Purchase Handlers ---

@with_error_handling()
async def handle_buy_request(product_id: str, ctx: ConversationContext) -> Optional[str]:
    """Enhanced buy request with better variant handling and personalization"""

    product = await shopify_service.get_product_by_id(product_id)
    if not product:
        return ErrorMessages.PRODUCT_NOT_FOUND

    # Get product variants
    variants = await shopify_service.get_product_variants(product.id)

    if not variants:
        # Product exists but no variants available
        product_url = shopify_service.get_product_page_url(product.handle)
        return f"*{product.title}* is currently unavailable. You can check for updates here: {product_url}"

    # Single variant - direct purchase
    if len(variants) == 1:
        variant = variants[0]
        if not variant.get("available", True):
            return f"Sorry, *{product.title}* is currently out of stock. Would you like me to notify you when it's back?"

        cart_url = shopify_service.get_add_to_cart_url(variant["id"])

        # Personalized response based on customer history
        greeting = f"Perfect choice{f', {ctx.customer_name}' if ctx.customer_name else ''}! 🛍️"

        return f"{greeting}\n\nI've prepared *{product.title}* for you.\n\n💰 Price: ₹{variant.get('price', 'N/A')}\n\n👆 Complete your purchase here:\n{cart_url}"

    # Multiple variants - show options
    else:
        available_variants = [v for v in variants if v.get("available", True)][:MessageLimits.MAX_VARIANTS_DISPLAY]

        if not available_variants:
            return f"Sorry, all variants of *{product.title}* are currently out of stock."

        # Send variant selection
        variant_options = {}
        for variant in available_variants:
            variant_title = variant.get("title", "Default")
            variant_price = variant.get("price", "N/A")
            option_text = f"{variant_title} - ₹{variant_price}"
            variant_options[option_text] = f"option_{variant['id']}"

        await whatsapp_service.send_quick_replies(
            ctx.phone_number,
            f"Please select an option for *{product.title}*:",
            variant_options
        )
        return None

@with_error_handling()
async def handle_greeting(ctx: ConversationContext, is_casual: bool = True) -> Optional[str]:
    """Enhanced greeting with personalization and context"""
    history = ctx.customer.get("conversation_history", [])

    # New user greeting
    if not history:
        greeting = f"Hello {ctx.customer_name + ', ' if ctx.customer_name else ''}welcome to FeelOri! 🌟\n\n"
        if not is_casual:
            greeting += "I'm here to help you find beautiful jewelry. "
        greeting += "What catches your eye today?"

        # Add quick suggestions for new users
        if not is_casual:
            categories = {
                "✨ Latest Arrivals": "latest_arrivals",
                "⭐ Bestsellers": "bestsellers",
                "💍 Rings": "rings",
                "📿 Necklaces": "necklaces"
            }
            await whatsapp_service.send_quick_replies(ctx.phone_number, greeting, categories)
            return None

        return greeting

    # Returning user greeting
    greeting = f"Welcome back{f', {ctx.customer_name}' if ctx.customer_name else ''}! 👋"

    # Add personalized touch for returning users
    if ctx.favorite_categories:
        top_category = ctx.favorite_categories[0].title()
        greeting += f" Still loving our {top_category} collection?"

    return greeting

# --- Enhanced Helper Functions ---

async def send_enhanced_product_results(
    products: List[Product], ctx: ConversationContext, total_count: int
) -> None:
    """Send enhanced product results with better formatting and a follow-up question."""

    # Cache search state
    search_data = {
        "query": ctx.original_message,
        "page": 1,
        "total_count": total_count,
        "timestamp": datetime.utcnow().isoformat()
    }
    await cache_service.set(f"state:last_search:{ctx.phone_number}", json.dumps(search_data), ttl=CacheConfig.SEARCH_STATE_TTL)
    await cache_service.set(f"state:last_product_list:{ctx.phone_number}", json.dumps([p.dict() for p in products]), ttl=CacheConfig.PRODUCT_STATE_TTL)

    # Create enhanced header based on search results
    if total_count > len(products):
        header_text = f"Found {total_count} items! Here are the top {len(products)} for you ✨"
    else:
        header_text = f"Found {len(products)} perfect match{'es' if len(products) != 1 else ''} for you ✨"

    # Check if user has buying intent
    buying_intent = await cache_service.get(f"state:buying_intent:{ctx.phone_number}")
    if buying_intent:
        body_text = "Tap any item to purchase or get more details!"
        await cache_service.delete(f"state:buying_intent:{ctx.phone_number}")
    else:
        body_text = "Here are a few options I found. Let me know if you'd like to see more, or describe the style you're looking for!"

    await send_product_card(products, ctx, header_text, body_text)

# In app/services/order_service.py
async def send_product_card(products: List[Product], ctx: ConversationContext, header_text: str, body_text: str) -> None:
    """
    MODIFIED: Reinstates the graceful fallback to an interactive list message
    before resorting to a plain text list.
    """
    try:
        # We still attempt to get the catalog_id, but we will pass it to the
        # whatsapp_service even if it's None. The service itself should handle
        # falling back to an interactive list message.
        catalog_id = await whatsapp_service.get_catalog_id()
        product_items = [{"product_retailer_id": p.sku} for p in products if p.sku]

        # This service call is expected to try the catalog first, then fall back
        # to an interactive list if the catalog_id is None or items are missing.
        await whatsapp_service.send_multi_product_message(
            to=ctx.phone_number,
            header_text=header_text,
            body_text=body_text,
            footer_text="Powered by FeelOri ✨",
            catalog_id=catalog_id,
            section_title="Products",
            product_items=product_items,
            fallback_products=products
        )
    except Exception as e:
        # This 'except' block is now the FINAL fallback for when the API call
        # itself fails completely. It will resort to a simple text list.
        logger.error(f"Error sending rich product message, falling back to text: {e}")
        await send_text_product_list(ctx.phone_number, products, header_text, body_text)


async def send_text_product_list(phone_number: str, products: List[Product], header_text: str, body_text: str) -> None:
    """Fallback text-based product listing - MODIFIED FOR SAFETY"""
    try:
        message_parts = [f"*{header_text}*\n", body_text, "\n"]

        for i, product in enumerate(products[:5], 1):
            price_info = "Price on request"
            # --- Start of Modification ---
            # Safely check for variants and price information
            if hasattr(product, 'variants') and product.variants:
                variant = product.variants[0]
                if hasattr(variant, 'price'):
                    price_info = f"₹{variant.price}"
            # --- End of Modification ---

            product_line = f"{i}. *{product.title}*\n   💰 {price_info}\n   🔗 {shopify_service.get_product_page_url(product.handle)}\n"
            message_parts.append(product_line)

        message_parts.append("\nTap any link to view details! ✨")
        final_message = "\n".join(message_parts)

        await whatsapp_service.send_message(phone_number, final_message)

    except Exception as e:
        logger.error(f"Error sending text product list: {e}")
        # This is the message the user is seeing.
        await whatsapp_service.send_message(phone_number, "I found some great products but I'm having trouble displaying them right now. Please try again!")

def format_orders_response_enhanced(orders: List[Dict], customer: Dict) -> str:
    """Enhanced order response formatting with better UX"""
    sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)

    customer_name = customer.get("name", "")
    greeting = f"Hi {customer_name}! " if customer_name else "Hi! "

    response_parts = [f"{greeting}I found {len(orders)} order(s) for your number:\n"]

    for order in sorted_orders[:MessageLimits.MAX_ORDERS_DISPLAY]:
        response_parts.append(format_single_order_enhanced(order))

    if len(orders) > MessageLimits.MAX_ORDERS_DISPLAY:
        remaining = len(orders) - MessageLimits.MAX_ORDERS_DISPLAY
        response_parts.append(f"... and {remaining} more order(s)")

    response_parts.append("\n💬 Reply with an order number for detailed tracking info!")

    return "\n".join(response_parts)

def format_single_order_enhanced(order: Dict) -> str:
    """Enhanced single order formatting"""
    try:
        order_date = datetime.fromisoformat(order["created_at"].replace("Z", "+00:00")).strftime("%d %b %Y")
    except (ValueError, TypeError, KeyError):
        order_date = "Date unavailable"

    order_number = order.get('order_number', 'N/A')
    status = (order.get("fulfillment_status") or "unfulfilled").replace("_", " ").title()

    # Enhanced status with emojis
    status_emoji = {
        "unfulfilled": "📦",
        "fulfilled": "✅",
        "partial": "🚛",
        "restocked": "🔄"
    }.get(order.get("fulfillment_status", "unfulfilled"), "📦")

    total_price = order.get('current_total_price', '0')
    currency = order.get('currency', 'INR')

    # Get item count
    line_items = order.get('line_items', [])
    item_count = sum(item.get('quantity', 0) for item in line_items)

    return (f"🛍️ *Order #{order_number}*\n"
           f"📅 {order_date}\n"
           f"💰 {currency} {total_price} ({item_count} item{'s' if item_count != 1 else ''})\n"
           f"{status_emoji} *{status}*\n")

# --- Conversation History Management ---

async def update_conversation_history(customer: dict, message: str, response: Optional[str]) -> None:
    """Update conversation history asynchronously"""
    try:
        phone_number = customer["phone_number"]

        history_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "user_message": message,
            "bot_response": response,
            "message_id": str(uuid.uuid4())
        }

        # Add to database
        await db_service.add_conversation_entry(phone_number, history_entry)

        # Update cache
        cache_key = f"customer:v3:{phone_number}"
        customer["conversation_history"] = customer.get("conversation_history", [])[-MessageLimits.MAX_CONVERSATION_HISTORY:]
        customer["conversation_history"].append(history_entry)
        customer["total_messages"] = customer.get("total_messages", 0) + 1
        customer["last_interaction"] = datetime.utcnow()

        await cache_service.set(cache_key, json.dumps(customer, default=str), ttl=CacheConfig.CUSTOMER_TTL)

    except Exception as e:
        logger.error(f"Error updating conversation history: {e}")

async def update_user_search_preferences(customer: dict, entities: dict) -> None:
    """Update user preferences based on successful searches"""
    try:
        phone_number = customer["phone_number"]
        favorite_categories = set(customer.get("favorite_categories", []))

        # Add searched categories to preferences
        for category in ["products", "materials", "styles", "gemstones"]:
            items = entities.get(category, [])
            favorite_categories.update(items[:2])  # Add top 2 items from each category

        # Keep only top 10 categories
        customer["favorite_categories"] = list(favorite_categories)[:10]

        # Update in database
        await db_service.update_customer_preferences(phone_number, {"favorite_categories": customer["favorite_categories"]})

    except Exception as e:
        logger.error(f"Error updating user preferences: {e}")

# --- Webhook Processing (Enhanced) ---

@with_metrics("process_webhook_message")
async def process_webhook_message(message: Dict, webhook_data: Dict):
    """Enhanced webhook message processing with better error handling"""
    try:
        message_id = message.get("id")
        from_number = message.get("from")

        if not message_id or not from_number:
            logger.warning(f"Invalid webhook message: missing id or from_number")
            return

        clean_phone = EnhancedSecurityService.sanitize_phone_number(from_number)

        # Check for duplicates
        if await message_queue.is_duplicate_message(message_id, clean_phone):
            logger.info(f"Duplicate message ignored: {message_id}")
            return

        # Extract profile information
        contacts = webhook_data.get("contacts", [])
        profile_name = contacts[0].get("profile", {}).get("name") if contacts else None

        # Process different message types
        message_type = message.get("type", "unknown")
        message_text = ""

        if message_type == "text":
            message_text = EnhancedSecurityService.validate_message_content(
                message.get("text", {}).get("body", "")
            )
        elif message_type == "interactive":
            message_text = extract_interactive_response(message.get("interactive", {}))
        elif message_type == "image":
            media_id = message.get("image", {}).get("id")
            message_text = f"visual_search_{media_id}" if media_id else ""
        elif message_type in ["audio", "voice"]:
            # For future voice message support
            message_text = "voice_message"
        elif message_type == "document":
            # Handle document uploads (for future features)
            message_text = "document_upload"

        if message_text:
            # Add to processing queue
            queue_item = {
                "from_number": clean_phone,
                "message_text": message_text,
                "message_type": message_type,
                "profile_name": profile_name,
                "quoted_wamid": message.get("context", {}).get("id"),
                "timestamp": datetime.utcnow().isoformat()
            }

            await message_queue.add_message(queue_item)
            logger.info(f"Message queued for processing: {clean_phone} - {message_type}")
        else:
            logger.warning(f"Empty message content for {message_type} from {clean_phone}")

    except Exception as e:
        logger.error(f"Error processing webhook message: {e}", exc_info=True)

def extract_interactive_response(interactive_data: Dict) -> str:
    """Extract response from interactive message data"""
    reply_type = interactive_data.get("type")

    if reply_type == "list_reply":
        return interactive_data.get("list_reply", {}).get("id", "")
    elif reply_type == "button_reply":
        return interactive_data.get("button_reply", {}).get("id", "")
    elif reply_type == "product_list_reply":
        return interactive_data.get("product_list_reply", {}).get("product_retailer_id", "")

    return ""

@with_metrics("handle_status_update")
async def handle_status_update(status_data: Dict):
    """Enhanced status update handling with retry mechanism"""
    try:
        wamid = status_data.get("id")
        status = status_data.get("status")
        recipient_raw = status_data.get("recipient_id")

        if not all([wamid, status, recipient_raw]):
            logger.warning("Invalid status update data")
            return

        recipient_phone = EnhancedSecurityService.sanitize_phone_number(recipient_raw)

        # Retry mechanism for database updates
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = await db_service.db.customers.update_one(
                    {"phone_number": recipient_phone, "conversation_history.wamid": wamid},
                    {"$set": {"conversation_history.$.status": status, "conversation_history.$.status_updated_at": datetime.utcnow()}}
                )

                if result.modified_count > 0:
                    logger.info(f"Message status updated: {wamid} -> {status}")

                    # Update cache if available
                    cache_key = f"customer:v3:{recipient_phone}"
                    cached_customer = await cache_service.get(cache_key)
                    if cached_customer:
                        try:
                            customer_data = json.loads(cached_customer)
                            for entry in customer_data.get("conversation_history", []):
                                if entry.get("wamid") == wamid:
                                    entry["status"] = status
                                    entry["status_updated_at"] = datetime.utcnow().isoformat()
                                    break
                            await cache_service.set(cache_key, json.dumps(customer_data, default=str), ttl=CacheConfig.CUSTOMER_TTL)
                        except Exception as cache_error:
                            logger.warning(f"Error updating cached customer data: {cache_error}")

                    return

                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))  # Exponential backoff

            except Exception as retry_error:
                logger.warning(f"Status update attempt {attempt + 1} failed: {retry_error}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0 * (attempt + 1))

        logger.warning(f"Could not update message status after {max_retries} attempts: {wamid}")

    except Exception as e:
        logger.error(f"Error handling status update: {e}", exc_info=True)

# --- Advanced Features ---

@with_error_handling()
async def handle_abandoned_checkout(payload: dict):
    """Enhanced abandoned checkout handling with better personalization"""
    try:
        # Skip if checkout was completed
        if payload.get("completed_at"):
            logger.info("Skipping reminder for completed checkout")
            return

        # Extract contact information
        phone_number = (payload.get("phone") or
                       (payload.get("shipping_address") or {}).get("phone") or
                       (payload.get("billing_address") or {}).get("phone"))

        checkout_url = payload.get("abandoned_checkout_url")
        customer_email = payload.get("email")

        if not phone_number or not checkout_url:
            logger.warning("Insufficient data for abandoned checkout reminder")
            return

        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        customer = await get_or_create_customer(clean_phone)

        # Extract checkout details for personalization
        line_items = payload.get("line_items", [])
        total_price = payload.get("total_price")
        currency = payload.get("currency", "INR")

        # Build personalized message
        customer_name = customer.get("name") or payload.get("customer", {}).get("first_name") or "there"

        if line_items:
            item_count = len(line_items)
            first_item = line_items[0].get("title", "beautiful item")
            if item_count == 1:
                items_text = f"*{first_item}*"
            else:
                items_text = f"*{first_item}* and {item_count - 1} other item{'s' if item_count > 2 else ''}"
        else:
            items_text = "some beautiful items"

        price_text = f" (Total: {currency} {total_price})" if total_price else ""

        message = (f"Hi {customer_name}! 👋\n\n"
                  f"I noticed you were interested in {items_text}{price_text}. "
                  f"They're still waiting for you! ✨\n\n"
                  f"Complete your purchase here:\n{checkout_url}\n\n"
                  f"Need help deciding? Just reply and I'll assist you! 💎")

        await whatsapp_service.send_message(clean_phone, message)
        logger.info(f"Abandoned checkout reminder sent to {clean_phone}")

    except Exception as e:
        logger.error(f"Error handling abandoned checkout: {e}", exc_info=True)

@with_error_handling()
async def handle_order_fulfillment(payload: dict):
    """Handle order fulfillment notifications"""
    try:
        # Extract order and customer information
        order_id = payload.get("id")
        order_number = payload.get("order_number")
        phone_number = (payload.get("phone") or
                       (payload.get("shipping_address") or {}).get("phone") or
                       (payload.get("billing_address") or {}).get("phone"))

        if not phone_number or not order_number:
            logger.warning("Insufficient data for order fulfillment notification")
            return

        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        customer = await get_or_create_customer(clean_phone)

        # Get tracking information
        fulfillments = payload.get("fulfillments", [])
        tracking_number = None
        tracking_company = None

        if fulfillments:
            latest_fulfillment = fulfillments[-1]
            tracking_number = latest_fulfillment.get("tracking_number")
            tracking_company = latest_fulfillment.get("tracking_company")

        customer_name = customer.get("name", "")
        greeting = f"Great news{f', {customer_name}' if customer_name else ''}! 🎉"

        message = (f"{greeting}\n\n"
                  f"Your order #{order_number} has been shipped! 📦✨\n\n")

        if tracking_number:
            message += f"📋 *Tracking Number:* {tracking_number}\n"
        if tracking_company:
            message += f"🚚 *Courier:* {tracking_company}\n"

        message += (f"\nYour beautiful jewelry is on its way to you! "
                   f"You should receive it within 3-5 business days.\n\n"
                   f"Questions? Just reply here! 💝")

        await whatsapp_service.send_message(clean_phone, message)
        logger.info(f"Order fulfillment notification sent for order {order_number}")

    except Exception as e:
        logger.error(f"Error handling order fulfillment: {e}", exc_info=True)


async def send_packing_alert_background(order_payload: Dict):
    """Sends a notification to the packing department about a new order."""
    try:
        packing_phone = settings.packing_dept_whatsapp_number
        if not packing_phone:
            logger.warning("Packing alert not sent: PACKING_DEPT_WHATSAPP_NUMBER is not set.")
            return

        dashboard_url = settings.dashboard_url
        order_number = order_payload.get("order_number")

        message_body = (
            f"🎉 New Order Received! #{order_number}\n\n"
            f"This has been added to your packing queue. View the full list here:\n"
            f"{dashboard_url}"
        )
        await whatsapp_service.send_message(packing_phone, message_body)
    except Exception as e:
        logger.error(f"send_packing_alert_background_error: {e}")
# --- Customer Journey and Engagement Tracking ---

async def update_customer_journey_stage(phone_number: str, action: str):
    """Update customer journey stage based on actions"""
    try:
        customer = await get_or_create_customer(phone_number)
        current_stage = JourneyStage(customer.get("journey_stage", JourneyStage.NEW.value))
        new_stage = current_stage

        # Define stage transitions
        stage_transitions = {
            "search": JourneyStage.BROWSING,
            "view_product": JourneyStage.BROWSING,
            "add_to_cart": JourneyStage.CONSIDERING,
            "purchase_intent": JourneyStage.CONSIDERING,
            "complete_purchase": JourneyStage.PURCHASING,
            "repeat_purchase": JourneyStage.LOYAL,
        }

        if action in stage_transitions:
            new_stage = stage_transitions[action]

        # Update if stage changed
        if new_stage != current_stage:
            await db_service.update_customer_data(phone_number, {"journey_stage": new_stage.value})

            # Update cache
            cache_key = f"customer:v3:{phone_number}"
            customer["journey_stage"] = new_stage.value
            await cache_service.set(cache_key, json.dumps(customer, default=str), ttl=CacheConfig.CUSTOMER_TTL)

            logger.info(f"Customer {phone_number} journey stage updated: {current_stage.value} -> {new_stage.value}")

    except Exception as e:
        logger.error(f"Error updating customer journey stage: {e}")

async def calculate_engagement_score(customer: Dict) -> float:
    """Calculate customer engagement score based on interactions"""
    try:
        score = 0.0

        # Base score from total messages
        total_messages = customer.get("total_messages", 0)
        score += min(total_messages * 0.1, 2.0)  # Max 2 points from messages

        # Recency bonus
        last_interaction = customer.get("last_interaction")
        if last_interaction:
            if isinstance(last_interaction, str):
                last_interaction = datetime.fromisoformat(last_interaction.replace("Z", "+00:00"))

            days_since = (datetime.utcnow() - last_interaction.replace(tzinfo=None)).days
            if days_since < 1:
                score += 1.0  # Very recent
            elif days_since < 7:
                score += 0.5  # Recent

        # Journey stage bonus
        journey_stage = customer.get("journey_stage", JourneyStage.NEW.value)
        stage_scores = {
            JourneyStage.NEW.value: 0.0,
            JourneyStage.BROWSING.value: 0.5,
            JourneyStage.CONSIDERING.value: 1.0,
            JourneyStage.PURCHASING.value: 2.0,
            JourneyStage.LOYAL.value: 3.0
        }
        score += stage_scores.get(journey_stage, 0.0)

        # Favorite categories bonus (shows engagement)
        favorite_categories = customer.get("favorite_categories", [])
        score += min(len(favorite_categories) * 0.1, 0.5)

        return min(score, 10.0)  # Cap at 10

    except Exception as e:
        logger.error(f"Error calculating engagement score: {e}")
        return 0.0

# --- Analytics and Metrics ---

async def track_search_analytics(phone_number: str, search_query: str, results_count: int):
    """Track search analytics for business insights"""
    try:
        analytics_data = {
            "phone_number": phone_number,
            "search_query": search_query,
            "results_count": results_count,
            "timestamp": datetime.utcnow(),
            "success": results_count > 0
        }

        # Store in analytics collection
        await db_service.store_search_analytics(analytics_data)

    except Exception as e:
        logger.error(f"Error tracking search analytics: {e}")

async def track_conversion_funnel(phone_number: str, event: str, product_id: str = None):
    """Track conversion funnel events"""
    try:
        funnel_data = {
            "phone_number": phone_number,
            "event": event,  # view, add_to_cart, purchase, etc.
            "product_id": product_id,
            "timestamp": datetime.utcnow()
        }

        await db_service.store_funnel_analytics(funnel_data)

    except Exception as e:
        logger.error(f"Error tracking conversion funnel: {e}")

# --- Scheduled Tasks ---

async def send_daily_engagement_summary():
    """Send daily engagement summary to admin"""
    try:
        # Get daily metrics
        today = datetime.utcnow().date()
        metrics = await db_service.get_daily_metrics(today)

        summary = (f"📊 *Daily Summary - {today}*\n\n"
                  f"👥 Active Users: {metrics.get('active_users', 0)}\n"
                  f"💬 Messages: {metrics.get('total_messages', 0)}\n"
                  f"🔍 Searches: {metrics.get('searches', 0)}\n"
                  f"🛒 Purchase Intents: {metrics.get('purchase_intents', 0)}\n"
                  f"⭐ Top Search: {metrics.get('top_search', 'N/A')}")

        # Send to admin numbers
        admin_numbers = settings.admin_whatsapp_numbers
        for admin_number in admin_numbers:
            await whatsapp_service.send_message(admin_number, summary)

    except Exception as e:
        logger.error(f"Error sending daily engagement summary: {e}")

async def cleanup_expired_cache():
    """Clean up expired cache entries"""
    try:
        # This would typically be handled by Redis TTL, but we can add custom cleanup
        expired_patterns = [
            "state:*",
            "temp:*"
        ]

        for pattern in expired_patterns:
            await cache_service.cleanup_pattern(pattern)

        logger.info("Cache cleanup completed")

    except Exception as e:
        logger.error(f"Error during cache cleanup: {e}")

# --- Scheduler Setup ---

def setup_scheduled_tasks():
    """Setup scheduled background tasks"""
    try:
        # Daily summary at 9 PM
        if not scheduler.get_job('daily_summary'):
            scheduler.add_job(
                send_daily_engagement_summary,
                'cron',
                hour=21,
                minute=0,
                id='daily_summary'
            )
        
        # Cache cleanup every 6 hours
        if not scheduler.get_job('cache_cleanup'):
            scheduler.add_job(
                cleanup_expired_cache,
                'interval',
                hours=6,
                id='cache_cleanup'
            )
    except Exception as e:
        logger.error(f"Error setting up scheduled tasks: {e}")

# --- Initialization ---

def initialize_service():
    """Initialize the order service with all components"""
    try:
        # Setup scheduled tasks
        setup_scheduled_tasks()
        
        logger.info("Order service initialized successfully")
        
    except Exception as e:
        logger.error(f"Error initializing order service: {e}")

# Initialize on module load
initialize_service()

# --- Public API ---

__all__ = [
    'process_message',
    'process_webhook_message', 
    'handle_status_update',
    'handle_abandoned_checkout',
    'handle_order_fulfillment',
    'update_customer_journey_stage',
    'ConversationContext',
    'MessageRouter',
    'JourneyStage'
]