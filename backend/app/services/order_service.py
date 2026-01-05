# /app/services/order_service.py

import re
import json
import asyncio
import logging
import tenacity
from datetime import datetime
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any
from pymongo import ReturnDocument


from rapidfuzz import process, fuzz

from app.config.settings import settings
from app.config import rules as default_rules  # Keep as fallback for constants not yet in BusinessConfig
from app.models.domain import Product
from app.services.security_service import EnhancedSecurityService
from app.services.ai_service import ai_service
from app.services.shopify_service import shopify_service
from app.services.whatsapp_service import whatsapp_service
from app.services.db_service import db_service
from app.services.cache_service import cache_service
from app.services import security_service
from app.services.string_service import string_service
from app.services.rule_service import rule_service
from app.utils.queue import message_queue
from app.services.order_service_constants import CacheKeys, TriageStates, TriageButtons
# from apscheduler.schedulers.asyncio import AsyncIOScheduler


logger = logging.getLogger(__name__)
#  scheduler = AsyncIOScheduler()

@dataclass
class SearchConfig:
    """Configuration for search behavior."""
    customer: Optional[Dict] = None
    STOP_WORDS: Set[str] = field(default_factory=lambda: {
        "a", "about", "an", "any", "are", "authentic", "buy", "can", "do",
        "does", "find", "for", "get", "genuine", "give", "have", "help",
        "how", "i", "im", "is", "looking", "material", "me", "need",
        "please", "quality", "real", "send", "show", "some", "tell",
        "to", "want", "what", "when", "where", "which", "why", "you"
    })
    QUESTION_INDICATORS: Set[str] = field(default_factory=lambda: {
        "what", "are", "is", "does", "do", "how", "why", "which",
        "when", "where", "real", "genuine", "authentic", "quality", "material"
    })
    CATEGORY_EXCLUSIONS: Dict[str, List[str]] = field(default_factory=lambda: {
        "earring": ["necklace", "set", "haram", "bracelet"],
        "necklace": ["earring", "bracelet", "ring"],
        "bangle": ["necklace", "set", "earring", "ring"],
        "ring": ["necklace", "set", "earring", "bracelet"]
    })
    MIN_WORD_LENGTH: int = 2
    MAX_SEARCH_RESULTS: int = 5
    QA_RESULT_LIMIT: int = 1

class QueryBuilder:
    """Builds search queries from natural language messages."""
    def __init__(self, config: SearchConfig, customer: Optional[Dict] = None):
        self.config = config
        self.customer = customer

    def _fuzzy_correct_keywords(self, keywords: List[str]) -> List[str]:
        corrected = []
        for kw in keywords:
            match, score, _ = process.extractOne(kw, default_rules.VALID_KEYWORDS, scorer=fuzz.token_sort_ratio)
            if score >= 80:
                if kw != match: 
                    logger.info(f"Fuzzy keyword correction: {kw} -> {match} (Score: {score})")
                corrected.append(match)
            else:
                corrected.append(kw)
        return corrected

    def _extract_keywords(self, message: str) -> List[str]:
        words = [w.lower().strip() for w in default_rules.WORD_RE.findall(message.lower()) if len(w) >= self.config.MIN_WORD_LENGTH and w.lower() not in self.config.STOP_WORDS]
        normalized = [default_rules.PLURAL_MAPPINGS.get(w, w) for w in words]
        return list(dict.fromkeys(self._fuzzy_correct_keywords(normalized)))

    # --- THIS IS THE CORRECTED FUNCTION ---
    def _build_prioritized_query(self, keywords: List[str]) -> str:
        """Builds a simple AND-based search query."""
        if not keywords:
            return ""
        # This creates a simpler, more effective query like "ruby AND necklace"
        return " AND ".join(keywords)

    def _apply_exclusions(self, query: str, keywords: List[str]) -> str:
        if not query or not keywords: 
            return query
        primary_category = keywords[0]
        exclusions = self.config.CATEGORY_EXCLUSIONS.get(primary_category, [])
        if exclusions:
            exclusion_query = ' AND NOT ' + ' AND NOT '.join(exclusions)
            query += exclusion_query
            logger.info(f"Applied query exclusions for '{primary_category}': {exclusions}")
        return query

    def _parse_price_filter(self, message: str) -> Tuple[Optional[Dict], List[str]]:
        less_than_match = re.search(r'\b(under|below|less than|<)\s*₹?(\d+k?)\b', message, re.IGNORECASE)
        if less_than_match:
            price_str = less_than_match.group(2).replace('k', '000')
            return {"price": {"lessThan": float(price_str)}}, less_than_match.group(0).split()

        greater_than_match = re.search(r'\b(over|above|more than|>)\s*₹?(\d+k?)\b', message, re.IGNORECASE)
        if greater_than_match:
            price_str = greater_than_match.group(2).replace('k', '000')
            return {"price": {"greaterThan": float(price_str)}}, greater_than_match.group(0).split()
            
        return None, []

    def build_query_parts(self, message: str) -> Tuple[str, Optional[Dict]]:
        price_filter, price_words = self._parse_price_filter(message)
        message_for_text_search = message
        if price_words:
            for word in price_words:
                message_for_text_search = message_for_text_search.replace(word, "")
        keywords = self._extract_keywords(message_for_text_search)
        text_query = self._build_prioritized_query(keywords)
        return text_query, price_filter
        #return self._apply_exclusions(text_query, keywords), price_filter

class QuestionDetector:
    """Detects if a message is a question."""
    def __init__(self, config: SearchConfig):
        self.config = config
    def is_question(self, message: str) -> bool:
        message_lower = message.lower()
        words = set(message_lower.split())
        return not words.isdisjoint(self.config.QUESTION_INDICATORS) or '?' in message


# --- Core Message Processing Orchestration ---

async def _handle_security_verification(clean_phone: str, message_text: str, customer: Dict) -> Optional[str]:
    """Checks if the user is in an order verification state and handles their response."""
    verification_state_raw = await cache_service.get(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
    if not verification_state_raw:
        return None

    try:
        verification_state = json.loads(verification_state_raw)
        expected_last_4 = verification_state["expected_last_4"]
        order_name = verification_state["order_name"]

        if message_text and message_text.strip() == expected_last_4:
            await cache_service.set(CacheKeys.ORDER_VERIFIED.format(phone=clean_phone, order_name=order_name), "1", ttl=60)
            await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))

            response = await handle_order_detail_inquiry(order_name, customer)
            return response[:4096] if response else None
        else:
            await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
            return "That's not correct. Please try asking for your order status again."
    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Invalid verification state for {clean_phone}. Clearing state.")
        await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
        return None # Fall through to normal processing

async def _handle_triage_flow(clean_phone: str, message_text: str, message_type: str, business_id: str = "feelori") -> Optional[str]:
    """Handles the entire automated triage state machine."""
    triage_state_raw = await cache_service.get(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
    if not triage_state_raw and not message_text.startswith(TriageButtons.SELECT_ORDER_PREFIX):
        return None # Not in a triage flow

    triage_state = {}
    if triage_state_raw:
        try:
            triage_state = json.loads(triage_state_raw)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted triage state for {clean_phone}, clearing.")
            await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
            return None

    current_state = triage_state.get("state")

    if current_state == TriageStates.AWAITING_ORDER_CONFIRM:
        await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
        order_to_confirm = triage_state.get("order_number")
        if message_text == TriageButtons.CONFIRM_YES:
            await _send_triage_issue_list(clean_phone, order_to_confirm)
            return "[Bot is handling triage step 2: issue selection]"
        else:
            new_state = {"state": TriageStates.AWAITING_ORDER_NUMBER}
            await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=clean_phone), json.dumps(new_state), ttl=900)
            return "No problem. Please reply with the correct order number (e.g., #FO1039)."

    elif current_state == TriageStates.AWAITING_ORDER_NUMBER:
        order_number = message_text.strip()
        if re.fullmatch(r'#?[A-Z]{0,3}\d{4,6}', order_number, re.IGNORECASE):
            await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
            await _send_triage_issue_list(clean_phone, order_number)
            return "[Bot is handling triage step 2: issue selection]"
        else:
            return "That doesn't look like a valid order number. Please try again (e.g., #FO1039)."

    elif current_state == TriageStates.AWAITING_ISSUE_SELECTION:
        order_number = triage_state.get("order_number")
        await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
        if message_text == TriageButtons.ISSUE_DAMAGED:
            new_state = {"state": TriageStates.AWAITING_PHOTO, "order_number": order_number}
            await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=clean_phone), json.dumps(new_state), ttl=900)
            return "I understand. To process this, please reply with a photo of the damaged item."
        else:
            issue_text = message_text.replace("triage_issue_", "").replace("_", " ")
            logger.info(f"Triage: Escalating for {clean_phone}, Order: {order_number}, Issue: {issue_text}")
            
            # 1. Create the Ticket
            triage_ticket = {
                "customer_phone": clean_phone, "order_number": order_number,
                "issue_type": issue_text, "image_media_id": None, "status": "human_needed",
                "business_id": "feelori",
                "assigned_to": None,
                "created_at": datetime.utcnow()
            }
            await db_service.db.triage_tickets.insert_one(triage_ticket)

            # 2. CRITICAL FIX: Update Conversation Status so UI shows the banner
            await db_service.db.conversations.update_one(
                {"external_user_id": clean_phone, "tenant_id": business_id},
                {
                    "$set": {
                        "status": "human_needed", 
                        "ai_enabled": False, 
                        "ai_paused_by": "system"
                    }
                }
            )

            return string_service.get_formatted_string("HUMAN_ESCALATION", business_id=business_id)

    elif current_state == TriageStates.AWAITING_PHOTO and (message_type == "image" or message_text.startswith("visual_search_")):
        order_number = triage_state.get("order_number")
        await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
        image_id = "N/A"
        if message_text.startswith("visual_search_"):
             image_id = message_text.replace("visual_search_", "").split("_caption_")[0]
        logger.info(f"Triage: Got photo (Media ID: {image_id}) for {clean_phone}, Order: {order_number}. Escalating.")
        triage_ticket = {
            "customer_phone": clean_phone, "order_number": order_number,
            "issue_type": "damaged_item", "image_media_id": image_id, "status": "human_needed",
            "business_id": "feelori",
            "assigned_to": None,
            "created_at": datetime.utcnow()
        }
        await db_service.db.triage_tickets.insert_one(triage_ticket)
        return string_service.get_formatted_string("HUMAN_ESCALATION", business_id=business_id)

    if message_text.startswith(TriageButtons.SELECT_ORDER_PREFIX):
        order_number = message_text.replace(TriageButtons.SELECT_ORDER_PREFIX, "")
        if current_state:
            await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
        await _send_triage_issue_list(clean_phone, order_number)
        return "[Bot is handling triage step 2: issue selection]"

    return None # Fall through if no triage state was handled


def _can_update_status_to_open(conversation: dict) -> bool:
    """
    Returns False if the conversation is locked in 'human_needed' state by a ticket.
    """
    if not conversation:
        return True # New conversation, always allow
    if conversation.get("status") == "human_needed":
        # If AI is disabled, it means a human ticket is active. Do not flip to open.
        if not conversation.get("ai_enabled"):
            return False
    return True


async def process_message(phone_number: str, message_text: str, message_type: str, quoted_wamid: str | None, business_id: str = "feelori", profile_name: str = None) -> str | None:
    """
    Processes an incoming message, handling triage states before
    routing to the AI-first intent model.
    """
    try:
        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
        
        # Step A: Upsert conversation early and capture conversation_id for real-time syncing
        # 1. Fetch existing conversation first (read-only)
        conversation = await db_service.db.conversations.find_one(
            {"external_user_id": clean_phone, "tenant_id": business_id}
        )
        
        # 2. Decide if we allowed to force status to "open"
        new_status = "open" # Default for new or normal chats
        if not _can_update_status_to_open(conversation):
            new_status = conversation.get("status") # Keep existing status (human_needed)

        # 3. Perform the Upsert with the calculated status
        now = datetime.utcnow()
        conversation = await db_service.db.conversations.find_one_and_update(
            {"external_user_id": clean_phone, "tenant_id": business_id},
            {
                "$set": {
                    "external_user_id": clean_phone,
                    "tenant_id": business_id,
                    "last_message": {"type": "text", "text": message_text[:200]},
                    "last_message_at": now,
                    "updated_at": now,
                    "status": new_status, # <--- Uses the safe status
                },
                "$setOnInsert": {
                    "created_at": now,
                    "ai_enabled": True,
                    "ai_paused_by": None,
                    "assigned_to": None
                }
            },
            upsert=True,
            return_document=ReturnDocument.AFTER
        )
        conversation_id = conversation["_id"]
        
        # --- CRITICAL FIX: Link & Normalize inbound message log ---
        # We find the most recent unlinked inbound message and attach it.
        # We also copy 'message_text' to 'text' so the Frontend schema is consistent.
        # IMPORTANT: This must happen BEFORE bot suppression so messages are always linked.
        await db_service.db.message_logs.find_one_and_update(
            {
                "business_id": business_id,
                "phone": clean_phone,
                "direction": "inbound",
                "conversation_id": {"$exists": False}
            },
            {
                "$set": {
                    "conversation_id": conversation_id,
                    "type": "text",         # Standardize type
                    "text": message_text    # Normalize content -> text for Frontend
                }
            },
            sort=[("timestamp", -1)] 
        )
        # -------------------------------------------------------------
        
        # Bot Suppression: STRICTLY respect the ai_enabled flag.
        # If AI is disabled (whether by Manual Toggle OR Human Ticket), do not reply.
        # We default to True to ensure old conversations don't break.
        # NOTE: This check happens AFTER message linking to ensure messages are visible in UI.
        if conversation.get("ai_enabled", True) is False:
            logger.info(f"Bot suppressed for {clean_phone}: AI is explicitly disabled.")
            return None
        
        if clean_phone == settings.packing_dept_whatsapp_number:
            return string_service.get_formatted_string("PACKING_DEPT_REDIRECT", business_id=business_id)

        # --- BOT SUPPRESSION CHECK ---
        # Check if there is an active ticket handled by a human
        active_ticket = await db_service.db.triage_tickets.find_one({
            "customer_phone": clean_phone,
            "status": "human_needed",
            "business_id": "feelori"
        })

        if active_ticket:
            logger.info(f"Bot suppressed for {clean_phone}: Human agent active on ticket {active_ticket.get('_id')}")
            
            # --- FIX: Sync status for UI consistency ---
            # Revert status to 'human_needed' because the initial upsert above
            # incorrectly flipped it to 'open'.
            await db_service.db.conversations.update_one(
                {"_id": conversation_id},
                {
                    "$set": {
                        "status": "human_needed",
                        "ai_enabled": False,
                        "ai_paused_by": "system"
                    }
                }
            )
            
            return None  # Stop processing (Bot stays silent)
        # -----------------------------

        # --- OPT-OUT / DND COMPLIANCE CHECK ---
        # Check for STOP/UNSUBSCRIBE commands (case-insensitive)
        message_upper = message_text.strip().upper()
        if message_upper in ["STOP", "UNSUBSCRIBE"]:
            await db_service.toggle_opt_out(clean_phone, True)
            logger.info(f"Opt-out requested by {clean_phone[:4]}...")
            return "You have been unsubscribed from updates. Reply START to resubscribe."
        
        # Check for START command (case-insensitive)
        if message_upper == "START":
            await db_service.toggle_opt_out(clean_phone, False)
            logger.info(f"Opt-in requested by {clean_phone[:4]}...")
            return "You have been resubscribed! 🎉"
        # ----------------------------------------

        customer = await get_or_create_customer(clean_phone, profile_name=profile_name)

        # --- BROADCAST REPLY CHECK ---
        # Check if this is a reply to a broadcast message
        last_outbound = await db_service.get_last_outbound_message(clean_phone)
        if last_outbound and last_outbound.get("source") == "broadcast":
            logger.info(f"Detected reply to broadcast for {clean_phone}")
            # Create a ticket for broadcast reply
            triage_ticket = {
                "customer_phone": clean_phone,
                "order_number": "N/A",  # Broadcast replies may not have an order
                "issue_type": "broadcast_reply",
                "status": "human_needed",
                "business_id": "feelori",
                "assigned_to": None,
                "image_media_id": None,
                "created_at": datetime.utcnow()
            }
            await db_service.db.triage_tickets.insert_one(triage_ticket)
            return "Thanks for replying to our update! A team member will be with you shortly."
        # -----------------------------

        # --- Refactored State Handling ---
        if response := await _handle_security_verification(clean_phone, message_text, customer):
            return response
        if response := await _handle_triage_flow(clean_phone, message_text, message_type, business_id=business_id):
            return response
        # --- End of Refactored State Handling ---

        last_question_raw = await cache_service.redis.get(CacheKeys.LAST_BOT_QUESTION.format(phone=clean_phone))
        if last_question_raw:
            last_question = last_question_raw.decode()
            await cache_service.redis.delete(CacheKeys.LAST_BOT_QUESTION.format(phone=clean_phone))
            clean_msg = message_text.lower().strip()
            if last_question == "offer_bestsellers" and clean_msg in default_rules.AFFIRMATIVE_RESPONSES:
                return await handle_bestsellers(customer=customer)
            if last_question == "offer_unfiltered_products" and clean_msg in default_rules.AFFIRMATIVE_RESPONSES:
                return await handle_show_unfiltered_products(customer=customer)
            if clean_msg in default_rules.NEGATIVE_RESPONSES:
                return "No problem! Let me know if there's anything else I can help you find. ✨"

        if message_type == "interactive" or message_text.startswith("visual_search_"):
            intent = await analyze_intent(message_text, message_type, customer, quoted_wamid)
            response = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
            return response[:4096] if response else None

        # --- THIS IS THE FIX ---
        # 1. Check for an order number format BEFORE calling the AI.
        if re.fullmatch(r'#?[A-Z]{0,3}\d{4,6}', message_text.strip(), re.IGNORECASE):
            logger.info("Order number format detected. Routing directly to order_detail_inquiry.")
            # 2. If it matches, bypass the AI and route directly.
            response = await route_message("order_detail_inquiry", clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
            return response[:4096] if response else None
        # --- END OF FIX ---

        if quoted_wamid:
            last_product_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=clean_phone))
            if last_product_raw:
                logger.info(f"Detected contextual reply (quoted_wamid: {quoted_wamid}) about a product.")
                intent = "contextual_product_question"
                response = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
                return response[:4096] if response else None
        
        logger.debug(f"Classifying intent with AI for: '{message_text}'")
        
        last_product_raw = await cache_service.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=clean_phone))
        last_product_context = None
        if last_product_raw:
            try:
                last_product_context = json.loads(last_product_raw)
            except json.JSONDecodeError:
                logger.warning(f"Corrupted product context for {clean_phone}")
        is_reply = bool(quoted_wamid)

        intent_prompt = f"""
        You are an intelligent AI assistant for an Indian jewelry e-commerce WhatsApp store.
        Your task is to analyze the user's message and the conversation context, then classify the intent.
        Your response MUST be a single, valid JSON object with "intent" (string) and "keywords" (array of strings).

        **Conversation Context:**
        - User's Message: "{message_text}"
        - Is this a reply to a previous message?: {"Yes" if is_reply else "No"}
        - Last product shown to the user: {json.dumps(last_product_context)}

        **Possible Intents:**
        - 'human_escalation': User has a problem, is angry, or wants a person. (e.g., "this is broken", "issue with my order"). **Prioritize this if there is any doubt.**
        - 'product_search': User is looking for a product/category. Keywords should be product terms (e.g., "show me gold necklaces" -> ["gold", "necklaces"]).
        - 'product_inquiry': User is asking a specific question about a product. **If the user replies to a product and asks a vague question like "how much?", this is the correct intent.** Keywords should be the full question.
        - 'order_inquiry': User is asking for the status of an existing order.
        - 'price_inquiry': User is asking about price *without* referring to a specific product.
        - 'shipping_inquiry': User is asking about shipping policies or timelines.
        - 'discount_inquiry': User is asking for a discount or coupon.
        - 'greeting': A simple greeting.
        - 'thank_you': The user is expressing thanks.
        - 'smalltalk': Casual, non-transactional conversation.

        **Classification Rules:**
        1.  **Context is Key:** If `is_reply` is "Yes" or `last_product_shown` is not "None", the user's message is likely a `product_inquiry`.
        2.  **Keywords:** Always return an array of strings, even if it's empty. For inquiries, extract key terms (e.g., "shipping", "delhi").
        3.  **Format:** Always use lowercase intent names.

        Classify the message now based on the message and the context.
        """

        ai_result = {}
        try:
            last_product_raw = await cache_service.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=clean_phone))
            last_product_context = None
            if last_product_raw:
                try:
                    last_product_context = json.loads(last_product_raw)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted product context for {clean_phone}")

            ai_response = await asyncio.wait_for(
                ai_service.get_ai_json_response(prompt=intent_prompt),
                timeout=15.0
            )
            ai_result = ai_response or {}
            ai_intent = ai_result.get("intent", "rule_based")
            ai_keywords = ai_result.get("keywords", [])
            if not ai_keywords or isinstance(ai_keywords, str):
                qb = QueryBuilder(SearchConfig())
                ai_keywords = qb._extract_keywords(message_text) or [message_text]

            logger.info(f"AI classified intent as '{ai_intent}' with keywords: {ai_keywords}")

        except asyncio.TimeoutError:
            logger.warning("AI intent classification timed out. Falling back to rule-based.")
            ai_intent = "rule_based"
            qb = QueryBuilder(SearchConfig())
            ai_keywords = qb._extract_keywords(message_text) or [message_text]
        except Exception:
            logger.exception("AI intent classification failed. Falling back to rule-based.")
            ai_intent = "rule_based"
            qb = QueryBuilder(SearchConfig())
            ai_keywords = qb._extract_keywords(message_text) or [message_text]

        if ai_intent == "product_search":
            response = await handle_product_search(message=ai_keywords, customer=customer, phone_number=clean_phone, quoted_wamid=quoted_wamid, business_id=business_id)
        
        elif ai_intent == "human_escalation":
             response = await handle_human_escalation(phone_number=clean_phone, customer=customer, business_id=business_id)

        elif ai_intent == "product_inquiry":
            try:
                answer = await asyncio.wait_for(
                    ai_service.get_product_qa(
                        query=" ".join(ai_keywords),
                        product=None,
                        business_id=business_id
                    ),
                    timeout=15.0
                )
                response = answer
            except asyncio.TimeoutError:
                logger.warning(f"AI product Q&A timed out for query: {' '.join(ai_keywords)}")
                response = await _handle_error(customer)
            except Exception:
                logger.exception(f"AI product Q&A failed for query: {' '.join(ai_keywords)}")
                response = await _handle_error(customer)

        elif ai_intent == "greeting":
            response = await handle_greeting(phone_number=clean_phone, customer=customer, message=ai_keywords, quoted_wamid=quoted_wamid, business_id=business_id)

        elif ai_intent == "smalltalk":
            response = await handle_general_inquiry(
                message=ai_keywords,
                customer=customer,
                phone_number=clean_phone,
                quoted_wamid=quoted_wamid,
                business_id=business_id  # <--- CRITICAL FIX
            )

        else: 
            logger.debug("AI intent not definitive, running rule-based analyzer.")
            intent = await analyze_intent(message_text, message_type, customer, quoted_wamid)
            response = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
        
        # Step C: Log AI reply and update conversation when response is generated
        if response:
            reply_now = datetime.utcnow()
            response_text = response[:4096]  # Use the same truncation as return statement
            
            # Log AI reply (No wamid yet - rely on sparse index)
            await db_service.db.message_logs.insert_one({
                "tenant_id": business_id,
                "business_id": business_id,
                "conversation_id": conversation_id,
                "phone": clean_phone,
                "direction": "outbound",
                "source": "ai",
                "message_type": "text",
                "text": response_text,
                "content": response_text,
                "status": "sending",
                "timestamp": reply_now,
                "created_at": reply_now
            })
            
            # Update Conversation Preview
            await db_service.db.conversations.update_one(
                {"_id": conversation_id, "tenant_id": business_id},
                {
                    "$set": {
                        "last_message": {"type": "text", "text": response_text[:200]},
                        "last_message_at": reply_now,
                        "updated_at": reply_now
                    }
                }
            )
            # -----------------------------------------------------------

        return response[:4096] if response else None
        
    except Exception as e:
        logger.error(f"Message processing error for {phone_number}: {e}", exc_info=True)
        return string_service.get_formatted_string("ERROR_GENERAL", business_id=business_id)


# --- Helper function to get or create a customer ---
async def get_or_create_customer(phone_number: str, profile_name: str = None) -> Dict[str, Any]:
    """Retrieves an existing customer or creates a new one. Updates name if missing."""
    cached_customer = await cache_service.get(CacheKeys.CUSTOMER_DATA_V2.format(phone=phone_number))
    if cached_customer:
        try:
            return json.loads(cached_customer)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted customer cache for {phone_number}. Refetching.")

    customer = await db_service.get_customer(phone_number)
    if not customer:
        customer_data = {
            "phone_number": phone_number,
            "name": profile_name,
            "conversation_history": [],
            "last_interaction": datetime.utcnow(),
        }
        await db_service.create_customer(customer_data)
        customer = await db_service.get_customer(phone_number)
    elif profile_name and not customer.get("name"):
        # Update customer name if missing and profile_name is provided
        await db_service.update_customer_name(phone_number, profile_name)
        customer["name"] = profile_name
        # Invalidate cache to force refresh
        await cache_service.delete(CacheKeys.CUSTOMER_DATA_V2.format(phone=phone_number))
    
    await cache_service.set(CacheKeys.CUSTOMER_DATA_V2.format(phone=phone_number), json.dumps(customer, default=str), ttl=1800)
    return customer

def _get_whatsapp_product_id(product: Product) -> str:
    """
    Extracts the numeric product ID from a GID for WhatsApp Catalog,
    with added resilience and logging.
    """
    if not product or not product.id:
        logger.warning("Attempted to get WhatsApp product ID from an invalid product object.")
        return ""
    
    product_id_str = str(product.id)
    
    if 'gid://' in product_id_str:
        # Handle edge cases like trailing slashes before splitting
        numeric_id = product_id_str.rstrip('/').split('/')[-1]
        logger.debug(f"Converted GraphQL ID {product_id_str} to numeric ID {numeric_id}")
        return numeric_id
    
    logger.debug(f"Using existing numeric ID: {product_id_str}")
    return product_id_str


# --- Helper function to extract text from any message type ---
def get_message_text(message: Dict[str, Any]) -> str:
    """Extracts the textual content from various WhatsApp message types."""
    msg_type = message.get("type")
    if msg_type == "text":
        return message.get("text", {}).get("body", "")
    elif msg_type == "interactive":
        interactive = message.get("interactive", {})
        interactive_type = interactive.get("type")
        if interactive_type == "button_reply":
            return interactive.get("button_reply", {}).get("id", "")
        elif interactive_type == "list_reply":
            return interactive.get("list_reply", {}).get("id", "")
    elif msg_type == "image":
        return "[User sent an image]"
    return f"[Unsupported message type: '{msg_type}']"

def _analyze_interactive_intent(message: str) -> str:
    """Analyze intent for interactive messages based on their prefix."""
    for prefix, intent in default_rules.INTERACTIVE_PREFIXES.items():
        if message.startswith(prefix):
            return intent
    return "interactive_response"

def analyze_text_intent(message_lower: str) -> str:
    """Analyzes intent for text messages using rules from the database."""

    # --- THIS IS THE FIX --
    # The regex now accepts optional letters (A-Z) between the '#' and the numbers.
    if re.fullmatch(r'#?[A-Z]{0,3}\d{4,6}', message_lower.strip(), re.IGNORECASE):
        return "order_detail_inquiry"

    # We apply the same fix to the search regex.
    if re.search(r'#?[A-Z]{0,3}\d{4,6}', message_lower, re.IGNORECASE):
        return "order_detail_inquiry"
    # --- END OF FIX ---

    message_words = set(default_rules.WORD_RE.findall(message_lower))
    
    # Use the dynamic rules from the service
    for rule in rule_service.get_rules():
        single_word_patterns = set(rule.get("keywords", []))
        multi_word_phrases = rule.get("phrases", [])
        intent = rule.get("name")

        if single_word_patterns and not message_words.isdisjoint(single_word_patterns):
            return intent
        if multi_word_phrases and any(p in message_lower for p in multi_word_phrases):
            return intent
    
    # Fallback for conversational patterns
    has_high_priority_context = not message_words.isdisjoint(default_rules.HIGH_PRIORITY_CONTEXT_WORDS)
    is_greeting = not message_words.isdisjoint(default_rules.GREETING_KEYWORDS_SET)
    
    if is_greeting and not has_high_priority_context:
        return "greeting"
    if not message_words.isdisjoint(default_rules.THANK_KEYWORDS_SET):
        return "thank_you"
    
    return "general"

async def analyze_intent(message: str, message_type: str, customer: Dict, quoted_wamid: Optional[str] = None) -> str:
    """Main intent analysis orchestrator."""
    if not message: 
        return "general"
    message_lower = message.lower().strip()

    if message_type == "interactive":
        return _analyze_interactive_intent(message)
    if message.startswith("visual_search_"):
        return "visual_search"
    if quoted_wamid:
        last_product_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=customer['phone_number']))
        if last_product_raw:
            return "contextual_product_question"
    
    return analyze_text_intent(message_lower)

async def route_message(intent: str, phone_number: str, message: str, customer: Dict, quoted_wamid: Optional[str] = None, business_id: str = "feelori") -> Optional[str]:
    """Routes the message to the appropriate handler based on intent."""
    handler_map = {
        "product_search": handle_product_search,
        "contextual_product_question": handle_contextual_product_question,
        "latest_arrivals_inquiry": handle_latest_arrivals,
        "bestseller_inquiry": handle_bestsellers,
        "visual_search": handle_visual_search,
        "interactive_button_reply": handle_interactive_button_response,
        "product_detail": handle_product_detail,
        "more_results": handle_more_results,
        "human_escalation": handle_human_escalation,
        "order_inquiry": handle_order_inquiry,
        "order_detail_inquiry": handle_order_detail_inquiry,
        "support": handle_support_request,
        "shipping_inquiry": handle_shipping_inquiry,
        "contact_inquiry": handle_contact_inquiry,
        "review_inquiry": handle_review_inquiry,
        "reseller_inquiry": handle_reseller_inquiry,
        "bulk_order_inquiry": handle_bulk_order_inquiry,
        "discount_inquiry": handle_discount_inquiry,
        "price_feedback": handle_price_feedback,
        "price_inquiry": handle_price_inquiry,
        "greeting": handle_greeting,
        "thank_you": handle_thank_you,
        "general": handle_general_inquiry
    }
    handler = handler_map.get(intent, handle_general_inquiry)
    return await handler(phone_number=phone_number, message=message, customer=customer, quoted_wamid=quoted_wamid, business_id=business_id)


# --- Handler Functions ---

async def handle_product_search(message: List[str] | str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles a product search request with intelligent filtering."""
    try:
        business_id = kwargs.get("business_id", "feelori")
        config = SearchConfig()
        query_builder = QueryBuilder(config, customer=customer)

        # --- THIS IS THE CORRECTED LOGIC ---
        # 1. Preserve the original full message string for price parsing.
        original_message = message if isinstance(message, str) else " ".join(message)

        # 2. Coerce the input to a list of keywords for other logic.
        if isinstance(message, str):
            keywords = query_builder._extract_keywords(message)
        else:
            keywords = message

        # If after processing, there are no keywords, handle it as an unclear request.
        if not keywords:
            return await _handle_unclear_request(customer, original_message)
        
        # 3. Build query parts from the ORIGINAL message to keep price filters.
        text_query, price_filter = query_builder.build_query_parts(original_message)
        
        # 4. Use the clean keyword string for logging, display, and cache keys.
        message_str = " ".join(keywords)
        # --- END OF CORRECTED LOGIC ---

        # Check if any keyword is in our unavailable list.
        for keyword in keywords:
            if keyword in default_rules.UNAVAILABLE_CATEGORIES:
                # If it is, call the specific handler for unavailable products.
                return await _handle_no_results(customer, message_str)

        if not text_query and not price_filter:
            return await _handle_unclear_request(customer, message_str)

        filtered_products, unfiltered_count = await shopify_service.get_products(
            query=text_query, filters=price_filter, limit=config.MAX_SEARCH_RESULTS, business_id=business_id
        )

        if not filtered_products:
            if unfiltered_count > 0 and price_filter:
                price_cond = price_filter.get("price", {})
                price_str = f"under ₹{price_cond['lessThan']}" if "lessThan" in price_cond else f"over ₹{price_cond['greaterThan']}"
                response = f"I found {unfiltered_count} item(s), but none are {price_str}. 😔\n\nWould you like to see them anyway?"
                
                await cache_service.set(CacheKeys.LAST_SEARCH.format(phone=customer['phone_number']), json.dumps({"query": message_str, "page": 1}), ttl=900)
                await cache_service.set(CacheKeys.LAST_BOT_QUESTION.format(phone=customer['phone_number']), "offer_unfiltered_products", ttl=900)
                
                return response
            else:
                return await _handle_no_results(customer, message_str)

        return await _handle_standard_search(filtered_products, message_str, customer, business_id=business_id)
    except Exception:
        safe_msg = (original_message if 'original_message' in locals()
                    else (message if isinstance(message, str) else " ".join(message)))
        logger.exception(f"Error in product search for message: {safe_msg}")
        return await _handle_error(customer)


def _format_single_order(order: Dict, detailed: bool = False) -> str:
    """
    Formats order details into a user-friendly message.

    Args:
        order: Shopify order dictionary
        detailed: If True, returns a full detailed breakdown.
                  If False, returns a concise summary (for lists).
    """
    order_name = order.get("name") or f"#{order.get('order_number', 'N/A')}"
    
    # --- Date Formatting ---
    try:
        created_at = order.get("created_at") or order.get("processed_at")
        order_date = datetime.fromisoformat(created_at.replace("Z", "+00:00")).strftime("%d %b %Y")
    except Exception:
        order_date = "N/A"

    # --- Statuses ---
    fulfillment_status = (order.get("fulfillment_status") or "unfulfilled").replace("_", " ").title()
    financial_status = (order.get("financial_status") or "pending").replace("_", " ").title()

    # --- Price ---
    total_price = (
        f"{order.get('current_total_price_set', {}).get('shop_money', {}).get('amount', order.get('current_total_price', 'N/A'))} "
        f"{order.get('currency', '')}"
    )

    # --- Tracking Info ---
    tracking_info = ""
    if order.get("fulfillments"):
        ff = order["fulfillments"][0]
        tracking_info = f"\n🚚 Tracking: {ff.get('tracking_number')} via {ff.get('tracking_company')}"

    if not detailed:
        # --- Concise summary (for multiple orders) ---
        return (
            f"🛍️ Order {order_name}\n"
            f"📅 Placed: {order_date}\n"
            f"💰 Total: {total_price}\n"
            f"📋 Status: *{fulfillment_status}*{tracking_info}\n"
        )

    # --- Detailed breakdown (for single order view) ---
    line_items = [f"- {item['name']} (x{item['quantity']})" for item in order.get("line_items", [])]
    items_str = "\n".join(line_items) if line_items else "No items listed"

    return (
        f"Order *{order_name}* Details:\n\n"
        f"🗓️ Placed on: {order_date}\n"
        f"💰 Payment: {financial_status}\n"
        f"🚚 Fulfillment: {fulfillment_status}\n"
        f"🛍️ Items:\n{items_str}\n\n"
        f"Total: *{total_price}*{tracking_info}"
    )


async def handle_order_detail_inquiry(message: str, customer: Dict, **kwargs) -> str:
    """Handles a request for order details, including a new security verification step."""
    business_id = kwargs.get("business_id", "feelori")
    order_name_match = re.search(r'#?[a-zA-Z]*\d{4,}', message)
    if not order_name_match:
        return await _handle_unclear_request(customer, message)

    order_name = order_name_match.group(0).upper()
    if not order_name.startswith('#'):
        order_name = f"#{order_name}"

    # Use the local DB first to get order phone numbers for verification
    order_from_db = await db_service.db.orders.find_one({"order_number": order_name})
    if not order_from_db:
        # --- THIS IS THE FIX ---
        # Use get_formatted_string with order_number as a kwarg
        return string_service.get_formatted_string('ORDER_NOT_FOUND_SPECIFIC', business_id=business_id, order_number=order_name)
        # --- END OF FIX ---

    # --- ✅ NEW SECURITY LOGIC ---
    # 1. Check if the user has already been verified for this specific order in the last minute.
    is_verified = await cache_service.get(CacheKeys.ORDER_VERIFIED.format(phone=customer['phone_number'], order_name=order_name))
    
    if not is_verified:
        # 2. If not verified, get the real phone numbers associated with the order from our DB.
        order_phones = order_from_db.get("phone_numbers", [])
        if not order_phones:
            logger.warning(f"Security check failed: No phone numbers found in DB for order {order_name} to verify against.")
            # Use the corrected string here as well
            return string_service.get_formatted_string('ORDER_NOT_FOUND_SPECIFIC', business_id=business_id, order_number=order_name)

        # 3. Store the expected last 4 digits and set the user's state to 'awaiting_order_verification'.
        if not order_phones or not isinstance(order_phones[0], str) or len(order_phones[0]) < 4:
            logger.error(f"Invalid phone data for order {order_name}")
            return "Unable to verify this order. Please contact support."
        expected_last_4 = order_phones[0][-4:]
        await cache_service.set(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=customer['phone_number']), json.dumps({
            "order_name": order_name,
            "expected_last_4": expected_last_4
        }), ttl=300) # Give them 5 minutes to respond

        # 4. Send the challenge question to the user instead of the order details.
        await whatsapp_service.send_message(
            to_phone=customer["phone_number"],
            message=f"For your security, please reply with the last 4 digits of the phone number used to place order {order_name}."
        )
        return "" # Return an empty string to send no other messages right now.

    # --- END OF NEW SECURITY LOGIC ---

    # 5. If the user IS verified, fetch the full details from Shopify, clear the flag, and show the details.
    order_to_display = await shopify_service.get_order_by_name(order_name, business_id=business_id)
    if not order_to_display:
        # And use the corrected string here one last time for safety
        return string_service.get_formatted_string('ORDER_NOT_FOUND_SPECIFIC', business_id=business_id, order_number=order_name)
        
    await cache_service.delete(CacheKeys.ORDER_VERIFIED.format(phone=customer['phone_number'], order_name=order_name))
    return _format_single_order(order_to_display, detailed=True)


async def handle_show_unfiltered_products(customer: Dict, **kwargs) -> Optional[str]:
    """Shows products from the last search, ignoring any price filters."""
    phone_number = customer["phone_number"]
    last_search_raw = await cache_service.redis.get(CacheKeys.LAST_SEARCH.format(phone=phone_number))
    if not last_search_raw: 
        return "I've lost the context of your last search. Could you please search again?"
    
    try:
        last_search = json.loads(last_search_raw)
    except json.JSONDecodeError:
        logger.warning(f"Corrupted last_search cache for {customer['phone_number']}")
        return "I've lost the context of your last search. Could you please search again?"
    original_message = last_search["query"]
    
    config = SearchConfig()
    query_builder = QueryBuilder(config, customer=customer)
    text_query, _ = query_builder.build_query_parts(original_message)
    business_id = kwargs.get("business_id", "feelori")
    products, _ = await shopify_service.get_products(query=text_query, filters=None, limit=config.MAX_SEARCH_RESULTS, business_id=business_id)

    if not products: 
        return f"I'm sorry, I still couldn't find any results for '{original_message}'."
    return await _handle_standard_search(products, original_message, customer)

async def handle_contextual_product_question(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles questions asked in reply to a specific product message."""
    business_id = kwargs.get("business_id", "feelori")
    phone_number = customer["phone_number"]
    last_product_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=phone_number))
    if not last_product_raw: 
        return "I'm sorry, I've lost context. Could you search for the product again?"
    last_product = Product.parse_raw(last_product_raw)

    config = SearchConfig()
    query_builder = QueryBuilder(config, customer=customer)
    keywords = query_builder._extract_keywords(message)
    category = _identify_search_category(keywords)
    
    if category != "unknown" and category != "sets":
        return await handle_product_search(message, customer, business_id=business_id)

    if any(keyword in message.lower() for keyword in ["price", "cost", "how much", "rate"]):
        return f"The price for the *{last_product.title}* is ₹{last_product.price:,.2f}. ✨"
    if "available" in message.lower() or "stock" in message.lower():
        availability_text = last_product.availability.replace('_', ' ').title()
        return f"Yes, the *{last_product.title}* is currently {availability_text}!"

    prompt = ai_service.create_qa_prompt(last_product, message)
    try:
        ai_answer = await asyncio.wait_for(ai_service.generate_response(prompt, business_id=business_id), timeout=15.0)
        await whatsapp_service.send_product_detail_with_buttons(phone_number, last_product, business_id=business_id)
        return ai_answer
    except asyncio.TimeoutError:
        logger.warning(f"Contextual Q&A timed out for product {last_product.id}")
        return await _handle_error(customer)
    except Exception:
        logger.exception(f"Contextual Q&A failed for product {last_product.id}")
        return await _handle_error(customer)

async def handle_interactive_button_response(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles replies from interactive buttons on a product card."""
    business_id = kwargs.get("business_id", "feelori")
    
    if message.startswith("buy_"):
        product_id = message.replace("buy_", "")
        return await handle_buy_request(product_id, customer, business_id=business_id)
    elif message.startswith("more_"):
        product_id = message.replace("more_", "")
        product = await shopify_service.get_product_by_id(product_id, business_id=business_id)
        return product.description if product else "Details not found."
    elif message.startswith("similar_"):
        product_id = message.replace("similar_", "")
        product = await shopify_service.get_product_by_id(product_id, business_id=business_id)
        if product and product.tags: 
            return await handle_product_search(product.tags[0], customer, business_id=business_id)
        return "What kind of similar items are you looking for?"
    elif message.startswith("option_"):
        variant_id = message.replace("option_", "")
        cart_url = shopify_service.get_add_to_cart_url(variant_id, business_id=business_id)
        return f"Perfect! I've added that to your cart. Complete your purchase here:\n{cart_url}"
    
    return "I didn't understand that selection. How can I help?"


async def handle_buy_request(product_id: str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles a 'Buy Now' request, checking for product variants."""
    business_id = kwargs.get("business_id", "feelori")
    product = await shopify_service.get_product_by_id(product_id, business_id=business_id)
    if not product: 
        return "Sorry, that product is no longer available."

    variants = await shopify_service.get_product_variants(product.id, business_id=business_id)
    if len(variants) > 1:
        variant_options = {f"option_{v['id']}": v['title'] for v in variants[:3]}
        await whatsapp_service.send_quick_replies(
            customer["phone_number"],
            f"Please select an option for *{product.title}*:",
            variant_options,
            business_id=business_id
        )
        return "[Bot asked for variant selection]"
    elif variants:
        # --- THIS IS THE FIX ---
        # 1. Get the direct add-to-cart URL.
        cart_url = shopify_service.get_add_to_cart_url(variants[0]["id"], business_id=business_id)

        # 2. Create a simple text message.
        response_text = (
            f"Perfect! I've added the *{product.title}* to your cart. "
            f"Complete your purchase here:\n{cart_url}"
        )
        
        # 3. Send the plain text message instead of a template.
        await whatsapp_service.send_message(
            to_phone=customer["phone_number"],
            message=response_text,
            business_id=business_id
        )
        
        return f"[Sent plain text checkout link for {product.title}]"
        # --- END OF FIX ---
    else:
        product_url = shopify_service.get_product_page_url(product.handle, business_id=business_id)
        return f"This product is currently unavailable. You can view it here: {product_url}"

async def handle_price_inquiry(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles direct questions about price, considering context."""
    business_id = kwargs.get("business_id", "feelori")
    phone_number = customer["phone_number"]
    product_list_raw = await cache_service.redis.get(CacheKeys.LAST_PRODUCT_LIST.format(phone=phone_number))
    
    if product_list_raw:
        try:
            product_list = [Product(**p) for p in json.loads(product_list_raw)]
        except (json.JSONDecodeError, TypeError):
            logger.warning(f"Corrupted product_list cache for {phone_number}")
            product_list = []
        if len(product_list) > 1:
            return "I just showed you a few designs. Please tap 'View Details' on the one you're interested in for the price! 👍"

    product_to_price_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=phone_number))
    if product_to_price_raw:
        product_to_price = Product.parse_raw(product_to_price_raw)
        await whatsapp_service.send_product_detail_with_buttons(phone_number, product_to_price, business_id=business_id)
        return "[Bot sent product details]"
    
    return "I can help with prices! Which product are you interested in? Try searching for something like 'gold necklaces' first."

async def handle_product_detail(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Shows a detailed card for a specific product."""
    business_id = kwargs.get("business_id", "feelori")
    
    # Remove the 'product_' prefix from the button ID
    clean_id = message.replace("product_", "")

    # FIX: Check if it's already a full GID or just a numeric ID
    if clean_id.startswith("gid://"):
        graphql_gid = clean_id
    else:
        graphql_gid = f"gid://shopify/Product/{clean_id}"

    # Pass the correctly formatted GID to the service.
    product = await shopify_service.get_product_by_id(graphql_gid, business_id=business_id)
    
    if product:
        await cache_service.set(
            CacheKeys.LAST_SINGLE_PRODUCT.format(phone=customer['phone_number']),
            product.json(),
            ttl=900
        )
        await whatsapp_service.send_product_detail_with_buttons(customer["phone_number"], product, business_id=business_id)
        return "[Bot sent product details]"

    return "Sorry, I couldn't find details for that product."

async def handle_latest_arrivals(customer: Dict, **kwargs) -> Optional[str]:
    """Shows the newest products."""
    business_id = kwargs.get("business_id", "feelori")
    products, _ = await shopify_service.get_products(query="", limit=5, sort_key="CREATED_AT", business_id=business_id)
    if not products: 
        return "I couldn't fetch the latest arrivals right now. Please try again shortly."
    await _send_product_card(products=products, customer=customer, header_text="Here are our latest arrivals! ✨", body_text="Freshly added to our collection.", business_id=business_id)
    return "[Sent latest arrival recommendations]"

async def handle_bestsellers(customer: Dict, **kwargs) -> Optional[str]:
    """Shows the top-selling products."""
    business_id = kwargs.get("business_id", "feelori")
    products, _ = await shopify_service.get_products(query="", limit=5, sort_key="BEST_SELLING", business_id=business_id)
    if not products: 
        return "I couldn't fetch our bestsellers right now. Please try again shortly."
    await _send_product_card(products=products, customer=customer, header_text="Check out our bestsellers! 🌟", body_text="These are the items our customers love most.", business_id=business_id)
    return "[Sent bestseller recommendations]"

async def handle_more_results(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Shows more results based on the last search or viewed product."""
    phone_number = customer["phone_number"]
    search_query, price_filter, header_text, raw_query_for_display = None, None, "Here are some more designs ✨", ""

    last_product_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=phone_number))
    if last_product_raw:
        last_product = Product.parse_raw(last_product_raw)
        if last_product.tags:
            search_query, raw_query_for_display = last_product.tags[0], last_product.tags[0]
            header_text = f"More items similar to {last_product.title}"
    
    if not search_query:
        last_search_raw = await cache_service.redis.get(CacheKeys.LAST_SEARCH.format(phone=phone_number))
        if last_search_raw:
            try:
                raw_query = json.loads(last_search_raw)["query"]
            except (json.JSONDecodeError, KeyError):
                logger.warning(f"Corrupted last_search cache for {phone_number}")
                return "More of what? Please search for a product first (e.g., 'show me necklaces')."
            raw_query_for_display = raw_query
            config = SearchConfig()
            query_builder = QueryBuilder(config)
            search_query, price_filter = query_builder.build_query_parts(raw_query)

    if not search_query: 
        return "More of what? Please search for a product first (e.g., 'show me necklaces')."

    business_id = kwargs.get("business_id", "feelori")
    products, _ = await shopify_service.get_products(search_query, limit=5, filters=price_filter, business_id=business_id)
    if not products: 
        return f"I couldn't find any more designs for '{raw_query_for_display}'. Try something else."
    await _send_product_card(products=products, customer=customer, header_text=header_text, body_text="Here are a few more options.", business_id=business_id)
    return f"[Sent more results for '{raw_query_for_display}']"

async def handle_shipping_inquiry(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Provides shipping information and handles contextual delivery time questions."""
    business_id = kwargs.get("business_id", "feelori")
    message_lower = message.lower()
    if any(k in message_lower for k in {"policy", "cost", "charge", "fee"}):
        city_info = ""
        if "delhi" in message_lower: 
            city_info = "For Delhi, delivery is typically within **3-5 business days!** 🏙️\n\n"
        return string_service.get_formatted_string("SHIPPING_POLICY_INFO", business_id=business_id, city_info=city_info)

    cities = ["hyderabad", "delhi", "mumbai", "bangalore", "chennai", "kolkata"]
    found_city = next((city for city in cities if city in message_lower), None)
    await cache_service.set(
        CacheKeys.PENDING_QUESTION.format(phone=customer['phone_number']),
        json.dumps({
        "question_type": "delivery_time_inquiry", 
        "context": {"city": found_city}
        }), 
        ttl=900
    )
    return "To give you an accurate delivery estimate, I need to know which items you're interested in. Could you please search for a product?"

async def handle_visual_search(message: str, customer: Dict, **kwargs) -> Optional[str]:
    """Handles a visual search request using an uploaded image."""
    try:
        business_id = kwargs.get("business_id", "feelori")
        media_id = message.replace("visual_search_", "").strip().split("_caption_")[0]
        phone_number = customer["phone_number"]
        if not media_id: 
            return "I couldn't read the image. Please try uploading it again."

        await whatsapp_service.send_message(phone_number, "🔍 Analyzing your image and searching our catalog... ✨", business_id=business_id)
        image_bytes, mime_type = await whatsapp_service.get_media_content(media_id, business_id=business_id)
        if not image_bytes or not mime_type: 
            return "I had trouble downloading your image. Please try again."

        result = await asyncio.wait_for(
            ai_service.find_exact_product_by_image(image_bytes, mime_type),
            timeout=20.0 # Longer timeout for image processing
        )
        if not result or not result.get('success') or not result.get('products'):
            return "I couldn't find a good match for your image. 😔\n💡 **Tip:** Use clear, well-lit photos!"

        products = result.get('products', [])
        match_type = result.get('match_type', 'similar')
        header_text = f"✨ Found {len(products)} Similar Products"
        if match_type == 'exact': 
            header_text = "🎯 Perfect Match Found!"
        elif match_type == 'very_similar': 
            header_text = f"🌟 Found {len(products)} Excellent Matches"
        
        await _send_product_card(products=products, customer=customer, header_text=header_text, body_text="Here are some products matching your style!", business_id=business_id)
        return "[Sent visual search results]"
    except asyncio.TimeoutError:
        logger.warning(f"Visual search timed out for media ID {media_id}")
        return "My image search is taking a little too long right now. Please try again in a moment!"
    except Exception as e:
        logger.error(f"Critical error in visual search: {e}", exc_info=True)
        return "Something went wrong during the visual search. Please try again. 😔"

async def handle_order_inquiry(phone_number: str, customer: Dict, **kwargs) -> str:
    """
    Handles general order status inquiries by proactively searching for the
    customer's recent orders in the database.
    """
    business_id = kwargs.get("business_id", "feelori")
    # 1. Proactively search our database for recent orders
    recent_orders = await db_service.get_recent_orders_by_phone(phone_number, limit=3)

    if not recent_orders:
        # 2. NO ORDERS FOUND: Fall back to the original behavior
        logger.info(f"No orders found for {phone_number}. Asking for order number.")
        return string_service.get_formatted_string(
            "ORDER_INQUIRY_PROMPT",
            business_id=business_id
        )
    
    elif len(recent_orders) == 1:
        # 3. ONE ORDER FOUND: Give them the status directly
        logger.info(f"Found one recent order for {phone_number}. Displaying status.")
        order_data = recent_orders[0].get("raw", {}) # Get the raw payload
        return _format_single_order(order_data, detailed=False) # Use the existing formatter

    else:
        # 4. MULTIPLE ORDERS FOUND: List them and ask which one
        logger.info(f"Found multiple recent orders for {phone_number}. Asking to clarify.")
        order_list = []
        for order in recent_orders:
            order_num = order.get("order_number") or "N/A"
            order_date = "Unknown Date"
            try:
                raw_created_at = order.get("created_at") or order.get("raw", {}).get("created_at", "")
                order_date = datetime.fromisoformat(raw_created_at.replace("Z", "+00:00")).strftime("%d %b %Y")
            except Exception:
                pass # Keep default "Unknown Date"
            order_list.append(f"• {order_num} (from {order_date})")
        
        orders_text = "\n".join(order_list)
        return f"I found a few recent orders for this number:\n\n{orders_text}\n\nPlease reply with the order number (e.g., #FO1039) you'd like to check."

async def handle_support_request(message: str, customer: Dict, **kwargs) -> str:
    """Handles support requests for damaged items, returns, etc."""
    business_id = kwargs.get("business_id", "feelori")
    complaint_keywords = {"damaged", "broken", "defective", "wrong", "incorrect", "bad", "poor", "dull"}
    if any(keyword in message.lower() for keyword in complaint_keywords):
        return string_service.get_formatted_string("SUPPORT_COMPLAINT_RESPONSE", business_id=business_id)
    return string_service.get_formatted_string("SUPPORT_GENERAL_RESPONSE", business_id=business_id)

def get_last_conversation_date(history: list) -> Optional[datetime]:
    if not history: 
        return None
    try:
        ts = history[-1]["timestamp"]
        return datetime.fromisoformat(ts.replace('Z', '+00:00')) if isinstance(ts, str) else ts
    except (KeyError, ValueError, AttributeError): 
        return None

def get_previous_interest(history: list) -> Optional[str]:
    interest_keywords = {"earring", "necklace", "ring", "bracelet", "bangle", "pendant", "chain", "jhumka", "set"}
    try:
        for conv in reversed(history[-3:]):
            message = (conv.get("message") or "").lower()
            for keyword in interest_keywords:
                if keyword in message: 
                    return f"{keyword}s"
    except (KeyError, AttributeError): 
        pass
    return None

async def handle_greeting(phone_number: str, customer: Dict, **kwargs) -> str:
    """Generates a personalized greeting."""
    history = customer.get("conversation_history", [])
    name = customer.get("name", "").strip()
    name_greeting = f"{name}, " if name else ""

    if not history: 
        return f"Hello {name_greeting}welcome to FeelOri! 👋 I'm your AI assistant. What are you looking for today?"
    if len(history) <= 3: 
        return f"Hi {name_greeting}great to see you back! 👋 What can I help you discover?"
    
    last_convo_date = get_last_conversation_date(history)
    if last_convo_date and (datetime.utcnow() - last_convo_date.replace(tzinfo=None)).days <= 7:
        interest = get_previous_interest(history)
        interest_context = f" Still looking for {interest}?" if interest else ""
        return f"Welcome back, {name or 'there'}! 👋 Ready to continue our fashion journey?{interest_context}"
    
    return f"Hey {name_greeting}welcome back to FeelOri! 👋 What's catching your eye today?"

async def handle_general_inquiry(message: str, customer: Dict, **kwargs) -> str:
    """Handles general questions using the AI model."""
    try:
        business_id = kwargs.get("business_id", "feelori")
        context = {"conversation_history": customer.get("conversation_history", [])[-5:]}
        return await asyncio.wait_for(
            ai_service.generate_response(message, context, business_id=business_id),
            timeout=15.0
        )
    except asyncio.TimeoutError:
        logger.warning("General inquiry AI timed out.")
        return string_service.get_formatted_string("ERROR_AI_GENERAL", business_id=business_id)
    except Exception as e:
        logger.error(f"General inquiry AI error: {e}")
        return string_service.get_formatted_string("ERROR_AI_GENERAL", business_id=business_id)

# --- Handlers for string constants ---

async def handle_price_feedback(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("PRICE_FEEDBACK_RESPONSE", business_id=business_id)
async def handle_discount_inquiry(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("DISCOUNT_INFO", business_id=business_id)
async def handle_review_inquiry(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("REVIEW_INFO", business_id=business_id)
async def handle_bulk_order_inquiry(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("BULK_ORDER_INFO", business_id=business_id)
async def handle_reseller_inquiry(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("RESELLER_INFO", business_id=business_id)
async def handle_contact_inquiry(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("CONTACT_INFO", business_id=business_id)
async def handle_thank_you(**kwargs) -> str:
    business_id = kwargs.get("business_id", "feelori")
    return string_service.get_formatted_string("THANK_YOU_RESPONSE", business_id=business_id)


async def handle_human_escalation(phone_number: str, customer: Dict, **kwargs) -> str:
    """
    STARTS the automated triage flow instead of immediately escalating.
    It proactively finds the user's orders and asks them to confirm.
    """
    business_id = kwargs.get("business_id", "feelori")
    logger.info(f"Starting triage flow for {phone_number} instead of escalating.")
    
    # 1. Proactively search our database for recent orders
    recent_orders = await db_service.get_recent_orders_by_phone(phone_number, limit=3)

    if not recent_orders:
        # 2. NO ORDERS FOUND: Ask for the order number.
        logger.info(f"Triage: No orders found for {phone_number}. Asking for order number.")
        triage_state = {"state": TriageStates.AWAITING_ORDER_NUMBER}
        await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=phone_number), json.dumps(triage_state), ttl=900)
        return "I'm sorry to hear you're having an issue. To help, could you please reply with your order number (e.g., #FO1039)?"

    elif len(recent_orders) == 1:
        # 3. ONE ORDER FOUND: Ask to confirm.
        order_num = recent_orders[0].get("order_number")
        logger.info(f"Triage: Found one order ({order_num}) for {phone_number}. Asking for confirmation.")
        
        # Save the order number and state to Redis
        triage_state = {"state": TriageStates.AWAITING_ORDER_CONFIRM, "order_number": order_num}
        await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=phone_number), json.dumps(triage_state), ttl=900)
        
        options = {TriageButtons.CONFIRM_YES: "Yes, that's it", TriageButtons.CONFIRM_NO: "No, it's different"}
        await whatsapp_service.send_quick_replies(
            phone_number,
            f"I'm sorry to hear you're having an issue. I see your most recent order is **{order_num}**. Is this the one you need help with?",
            options,
            business_id=business_id
        )
        return "[Bot is asking to confirm order for triage]"

    else:
        # 4. MULTIPLE ORDERS FOUND: Ask to select.
        logger.info(f"Triage: Found multiple orders for {phone_number}. Asking to select.")
        
        options = {}
        for order in recent_orders:
            order_num = order.get("order_number")
            options[f"{TriageButtons.SELECT_ORDER_PREFIX}{order_num}"] = f"Order {order_num}"
        
        await whatsapp_service.send_quick_replies(
            phone_number,
            "I'm sorry to hear you're having an issue. I found a few of your recent orders. Which one do you need help with?",
            options
        )
        return "[Bot is asking to select order for triage]"

# --- Helper Functions for Handlers ---

async def _send_triage_issue_list(phone_number: str, order_number: str) -> None:
    """
    Sends the list of common issues for the user to select.
    This is Step 2 of the triage flow.
    """
    logger.info(f"Triage: Sending issue list for order {order_number} to {phone_number}.")
    
    # Set the new consolidated triage state
    triage_state = {"state": TriageStates.AWAITING_ISSUE_SELECTION, "order_number": order_number}
    await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=phone_number), json.dumps(triage_state), ttl=900)
    
    options = {
        TriageButtons.ISSUE_DAMAGED: "📦 Item is damaged",
        "triage_issue_wrong_item": "Wrong item received",
        "triage_issue_return": "I want to return it",
        "triage_issue_other": "Something else"
    }
    
    await whatsapp_service.send_quick_replies(
        phone_number,
        f"Thank you. What is the issue with order **{order_number}**?",
        options
    )

def _identify_search_category(keywords: List[str]) -> str:
    """Identifies the primary product category from a list of keywords."""
    specific = {
        "bangles": {"bangle", "bangles"}, "bracelets": {"bracelet", "bracelets"}, "rings": {"ring", "rings"},
        "necklaces": {"necklace", "necklaces", "haram", "choker"}, "earrings": {"earring", "earrings", "jhumka"},
        "hair_extensions": {"hair extension", "hair extensions"}
    }
    for cat, terms in specific.items():
        if any(w in terms for w in keywords): 
            return cat
    if any(w in {"set", "sets", "matching"} for w in keywords): 
        return "sets"
    return "unknown"

async def _handle_no_results(customer: Dict, original_query: str) -> str:
    """Provides intelligent responses when a search yields no results."""
    config = SearchConfig()
    query_builder = QueryBuilder(config)
    keywords = query_builder._extract_keywords(original_query)
    search_category = _identify_search_category(keywords)
    
    main_term = keywords[0] if keywords else "that item"
    response = f"I couldn't find any {main_term} in our collection right now. 🔍\n\nHowever, we have an amazing selection of handcrafted necklaces and earrings!\n\nWould you like to see our **bestsellers**? ⭐"
    
    if search_category in {"bangles", "rings", "bracelets"}:
        response = f"While we don't carry {search_category} right now, we have stunning **earrings and necklace sets** that would complement your look beautifully.\n\nWould you like me to show you some of our bestselling sets? ✨"
        
    await cache_service.set(CacheKeys.LAST_BOT_QUESTION.format(phone=customer['phone_number']), "offer_bestsellers", ttl=900)
    return response

async def _handle_unclear_request(customer: Dict, original_message: str) -> str:
    """Handles cases where the search intent is unclear."""
    return "I'd love to help! Could you tell me what type of jewelry you're looking for? (e.g., Necklaces, Earrings, Sets)"

async def _handle_standard_search(products: List[Product], message: str, customer: Dict, business_id: str = "feelori") -> str:
    """Handles standard product search results."""
    phone_number = customer["phone_number"]
    await cache_service.set(
        CacheKeys.LAST_SEARCH.format(phone=phone_number),
        json.dumps({"query": message, "page": 1}), 
        ttl=900
    )
    await cache_service.set(
        CacheKeys.LAST_PRODUCT_LIST.format(phone=phone_number),
        json.dumps([p.dict() for p in products]), 
        ttl=900
    )

    
    header_text = f"Found {len(products)} match{'es' if len(products) != 1 else ''} for you ✨"
    await _send_product_card(products=products, customer=customer, header_text=header_text, body_text="Tap any product for details!", business_id=business_id)

    pending_question_raw = await cache_service.redis.get(CacheKeys.PENDING_QUESTION.format(phone=phone_number))
    if pending_question_raw:
        await cache_service.redis.delete(CacheKeys.PENDING_QUESTION.format(phone=phone_number))
        try:
            pending_question = json.loads(pending_question_raw)
        except json.JSONDecodeError:
            logger.warning(f"Corrupted pending_question cache for {phone_number}")
            pending_question = {}
        if pending_question.get("question_type") == "delivery_time_inquiry":
            await asyncio.sleep(1.5)
            city = pending_question.get("context", {}).get("city")
            contextual_answer = f"Regarding your question about delivery to **{city.title()}**: it typically takes **3-5 business days**." if city else "Regarding delivery: it's typically **3-5 business days** for metro cities."
            await whatsapp_service.send_message(phone_number, contextual_answer)
    
    return f"[Sent {len(products)} product recommendations]"

async def _send_product_card(products: List[Product], customer: Dict, header_text: str, body_text: str, business_id: str = "feelori"):
    """Sends a rich multi-product message card."""
    catalog_id = await whatsapp_service.get_catalog_id(business_id=business_id)
    
    available_products = [p for p in products if p.availability == "in_stock" and p.id]

    # --- THIS IS THE FIX ---
    # Use the new helper function to ensure all IDs are numeric.
    product_items = [
        {"product_retailer_id": _get_whatsapp_product_id(p)}
        for p in available_products
    ]
    # --- END OF FIX ---

    await whatsapp_service.send_multi_product_message(
        to=customer["phone_number"], header_text=header_text, body_text=body_text,
        footer_text="Powered by FeelOri", catalog_id=catalog_id,
        section_title="Products", product_items=product_items, fallback_products=available_products,
        business_id=business_id
    )

async def _handle_error(customer: Dict) -> str:
    """Handles unexpected errors gracefully."""
    return "Sorry, I'm having trouble searching right now. 😔 Please try again in a moment."

def _perform_security_check(phone_number: str, customer: Dict) -> Optional[str]:
    """Checks if a user is asking for order details of another number."""
    latest_message = (customer.get("conversation_history", [{}])[-1] or {}).get("message", "")
    found_numbers = re.findall(r'\b\d{8,15}\b', latest_message)
    if not found_numbers: 
        return None
    
    sanitized_sender_phone = re.sub(r'\D', '', phone_number)
    if sanitized_sender_phone.startswith('91'): 
        sanitized_sender_phone = sanitized_sender_phone[2:]
    
    for number in found_numbers:
        if not re.sub(r'\D', '', number).endswith(sanitized_sender_phone):
            return "For your security, I can only check the order status for the phone number you're currently using."
    return None

def _format_orders_response(orders: List[Dict]) -> str:
    """Formats a list of orders into a single response string."""
    sorted_orders = sorted(orders, key=lambda o: o.get("created_at", ""), reverse=True)
    response_parts = [f"I found {len(orders)} order(s) for this number:\n"]
    for order in sorted_orders[:3]:
        response_parts.append(_format_single_order(order))
    response_parts.append("Reply with the order number for more details.")
    return "\n".join(response_parts)


# --- Webhook Processing Logic ---

async def process_webhook_message(message: Dict[str, Any], webhook_data: Dict[str, Any], business_id: str = "feelori", profile_name: str = None):
    """
    Main function to process an incoming webhook message from a user.
    """
    try:
        from_number = message.get("from")
        wamid = message.get("id")
        if not from_number or not wamid:
            logger.warning("Webhook message missing 'from' or 'id'.")
            return

        clean_phone = security_service.EnhancedSecurityService.sanitize_phone_number(from_number)
        
        # --- KILL SWITCH: Bot vs Human Mode with Auto-Release ---
        # Enforce auto-release for stale locks (>30 minutes)
        await db_service.enforce_auto_release(clean_phone)
        
        # Fetch conversation to check mode
        customer = await db_service.get_customer(clean_phone)
        if customer:
            conversation_mode = customer.get("conversation_mode", "bot")
            if conversation_mode == "human":
                logger.info(f"Skipping AI reply (Human Mode active) for {clean_phone[:4]}...")
                return
        # --- END OF KILL SWITCH ---

        # This duplicate check can be simplified now with a dedicated message log
        # but we'll leave it for now for extra safety.
        if await message_queue.is_duplicate_message(wamid, clean_phone):
            logger.info(f"Duplicate message {wamid} from {clean_phone} received, ignoring.")
            return

        if not await security_service.rate_limiter.check_phone_rate_limit(clean_phone):
            logger.warning(f"Rate limit exceeded for {clean_phone}.")
            return

        message_text = get_message_text(message)
        message_type = message.get("type", "unknown")
        # Use profile_name from parameter, fallback to webhook_data if not provided
        if not profile_name:
            profile_name = webhook_data.get("contacts", [{}])[0].get("profile", {}).get("name")
        quoted_wamid = message.get("context", {}).get("id")

        if message_type == "image":
            media_id = message.get("image", {}).get("id")
            caption = message.get("image", {}).get("caption", "")
            message_text = f"visual_search_{media_id}_caption_{caption}"

        if not message_text:
            logger.info(f"Ignoring empty message from {clean_phone}")
            return
            
        # --- THIS IS THE NEW LOGIC ---
        # Log the inbound message to the dedicated database collection

        log_data = {
            "wamid": wamid,
            "phone": clean_phone,
            "direction": "inbound",
            "message_type": message_type,
            "text": message_text,       # Frontend Source of Truth
            "content": message_text,    # Keep for legacy/backend compatibility
            "status": "received", # The initial status is 'received'
            "source": "customer",
            "timestamp": datetime.utcnow(),
            "business_id": business_id  # Add business_id for multi-tenancy
        }
        await db_service.log_message(log_data)
        # --- END OF NEW LOGIC ---

        message_data = {
            "from_number": clean_phone,
            "message_text": message_text,
            "message_type": message_type,
            "wamid": wamid,
            "profile_name": profile_name,
            "quoted_wamid": quoted_wamid,
            "business_id": business_id  # Pass business_id through message queue
        }
        await message_queue.add_message(message_data)

    except Exception as e:
        logger.error(f"Error in process_webhook_message: {e}", exc_info=True)


async def handle_status_update(status_data: Dict):
    """Processes a message status update from WhatsApp using the message_logs collection."""
    wamid, status = status_data.get("id"), status_data.get("status")
    if not wamid or not status:
        return

    try:
        from app.services.db_service import db_service

        # --- THIS IS THE FIX ---
        # Define a retryable update operation using tenacity
        @tenacity.retry(
            stop=tenacity.stop_after_delay(10),  # Stop after 10 seconds
            wait=tenacity.wait_exponential(multiplier=1, min=0.5, max=3),  # Wait 0.5s, 1s, 2s, 3s, 3s...
            retry=tenacity.retry_if_result(lambda result: result.modified_count == 0),  # Retry if it didn't update
            reraise=True  # Reraise the exception if it fails after 10s
        )
        async def _update_message_status():
            """Attempts to update the message status."""
            return await db_service.db.message_logs.update_one(
                {"wamid": wamid},
                {"$set": {"status": status}}
            )

        # Call the retryable function
        result = await _update_message_status()

        if result.modified_count > 0:
            logger.info(f"Message status updated for {wamid} to {status}")
        # --- END OF FIX ---

    except tenacity.RetryError:
        # This log happens *only* if it fails after all retries
        logger.warning(f"Could not find message {wamid} to update status in message_logs after 10 seconds.")
    except Exception as e:
        logger.error(f"Error handling status update: {e}", exc_info=True)


async def handle_abandoned_checkout(payload: dict, business_id: str = "feelori"):
    """
    Receives an abandoned checkout webhook and saves it to the database.
    The central scheduler will handle sending the reminder later.
    """
    checkout_id = payload.get("id")
    if not checkout_id:
        return # Ignore if there is no ID
    
    # Inject business_id into the payload for multi-tenant support
    payload["business_id"] = business_id
    
    # Just save the data. The scheduler will do the rest.
    await db_service.save_abandoned_checkout(payload)
    logger.info(f"Saved abandoned checkout {checkout_id} to database for future processing.")

async def send_packing_alert_background(order_payload: Dict):
    """Sends a notification to the packing department about a new order."""
    if not settings.packing_dept_whatsapp_number:
        logger.warning("Packing alert not sent: PACKING_DEPT_WHATSAPP_NUMBER is not set.")
        return
    message = f"🎉 New Order Received! #{order_payload.get('order_number')}\nView on the dashboard: {settings.dashboard_url}"
    await whatsapp_service.send_message(settings.packing_dept_whatsapp_number, message)