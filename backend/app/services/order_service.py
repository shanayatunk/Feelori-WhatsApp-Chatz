# /app/services/order_service.py

import re
import json
import asyncio
import logging
import tenacity
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Set, Dict, List, Tuple, Optional, Any
from pymongo import ReturnDocument


from rapidfuzz import process, fuzz

from app.config.settings import settings, get_business_config
from app.config import rules as default_rules  # Keep as fallback for constants not yet in BusinessConfig
from app.models.domain import Product
from app.services.security_service import EnhancedSecurityService
from app.services.ai_service import ai_service
from app.services.shopify_service import shopify_service
from app.services.whatsapp_service import whatsapp_service
from app.services.db_service import db_service
from app.services.cache_service import cache_service
from app.services import security_service
from app.services.security_service import rate_limiter
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


def normalize_price_text(text: str) -> str:
    """Normalizes '5k', '10k' to '5000', '10000' in any string."""
    return re.sub(
        r'\b(\d+)\s*k\b',
        lambda m: str(int(m.group(1)) * 1000),
        text.lower()
    )


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
        """
        Parses price filters like 'under 2000', 'above 5000', '2000 under', 'rs 5000 above'.
        Returns a Shopify-compatible filter dict and the list of words used for the price.
        """
        # --- IMPROVEMENT: Normalize Currency Symbols ---
        clean_msg = (
            message.lower()
            .replace("₹", "")
            .replace("rs", "")
            .replace("rupees", "")
            .replace(",", "")
            .strip()
        )

        # Regex for Prefix: "under 2000", "above 5k"
        prefix_lt = re.search(r'\b(under|below|less than|less|<)\s*(\d+k?)\b', clean_msg)
        prefix_gt = re.search(r'\b(over|above|more than|more|>)\s*(\d+k?)\b', clean_msg)

        # Regex for Suffix: "2000 under", "5k above"
        suffix_lt = re.search(r'\b(\d+k?)\s*(under|below|less)\b', clean_msg)
        suffix_gt = re.search(r'\b(\d+k?)\s*(over|above|more)\b', clean_msg)

        def normalize(val: str) -> float:
            return float(val.replace("k", "000"))

        price_filter = None
        used_words = []

        # Prioritize patterns (Suffix takes precedence if both exist to avoid overlap)
        if suffix_lt:
            price_filter = {"price": {"lessThan": normalize(suffix_lt.group(1))}}
            used_words = suffix_lt.group(0).split()
        elif suffix_gt:
            price_filter = {"price": {"greaterThan": normalize(suffix_gt.group(1))}}
            used_words = suffix_gt.group(0).split()
        elif prefix_lt:
            price_filter = {"price": {"lessThan": normalize(prefix_lt.group(2))}}
            used_words = prefix_lt.group(0).split()
        elif prefix_gt:
            price_filter = {"price": {"greaterThan": normalize(prefix_gt.group(2))}}
            used_words = prefix_gt.group(0).split()

        return price_filter, used_words

    def build_query_parts(self, message: str) -> Tuple[str, Optional[Dict]]:
        price_filter, price_words = self._parse_price_filter(message)
        message_for_text_search = message
        if price_words:
            for word in price_words:
                # Use \b to ensure we match whole words and ignore case
                message_for_text_search = re.sub(
                    r'\b' + re.escape(word) + r'\b',
                    '',
                    message_for_text_search,
                    flags=re.IGNORECASE
                )
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
    """
    Checks if the user is in an order verification state and handles their response.
    Includes brute-force protection (Max 3 attempts).
    """
    verification_state_raw = await cache_service.get(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
    if not verification_state_raw:
        return None

    # 1. ESCAPE HATCH: Check if user wants to cancel or switch topics
    # If the message is text (not digits) and looks like a command/intent
    clean_text = message_text.lower().strip()
    escape_words = {"cancel", "stop", "exit", "quit", "no", "nevermind", "show", "buy", "hi", "hello", "menu"}
    
    # If user says an escape word OR a sentence without digits (like "i dont know")
    if clean_text in escape_words or (len(clean_text) > 5 and not re.search(r'\d', clean_text)):
        await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
        # Returning None lets the message fall through to the main AI/Intent handler
        return None 

    try:
        verification_state = json.loads(verification_state_raw)
        expected_last_4 = verification_state["expected_last_4"]
        order_name = verification_state["order_name"]
        attempts = verification_state.get("attempts", 0)

        # 2. Proceed with digit verification...
        # Normalize input (remove spaces/dashes)
        input_digits = re.sub(r'\D', '', message_text)

        if input_digits and input_digits == expected_last_4:
            # SUCCESS: Verify and proceed
            await cache_service.set(CacheKeys.ORDER_VERIFIED.format(phone=clean_phone, order_name=order_name), "1", ttl=60)
            await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))

            response = await handle_order_detail_inquiry(order_name, customer)
            return response[:4096] if response else None
        
        else:
            # FAILURE LOGIC
            attempts += 1
            
            if attempts >= 3:
                # MAX ATTEMPTS REACHED: Lock out
                await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
                logger.warning(f"Security check failed 3 times for {clean_phone} on order {order_name}")
                return "⛔ Verification failed. For security reasons, I cannot show this order. Please contact our support team if you need help."
            
            # RETRY ALLOWED: Update state and warn user
            verification_state["attempts"] = attempts
            await cache_service.set(
                CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone),
                json.dumps(verification_state),
                ttl=300
            )
            
            remaining = 3 - attempts
            return f"That doesn't match our records. You have {remaining} attempt{'s' if remaining != 1 else ''} remaining. Please try again."

    except (json.JSONDecodeError, KeyError):
        logger.warning(f"Invalid verification state for {clean_phone}. Clearing state.")
        await cache_service.delete(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=clean_phone))
        return None

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
            return None
        else:
            new_state = {"state": TriageStates.AWAITING_ORDER_NUMBER}
            await cache_service.set(CacheKeys.TRIAGE_STATE.format(phone=clean_phone), json.dumps(new_state), ttl=900)
            return "No problem. Please reply with the correct order number (e.g., #FO1039)."

    elif current_state == TriageStates.AWAITING_ORDER_NUMBER:
        order_number = message_text.strip()
        if re.fullmatch(r'#?[A-Z]{0,3}\d{4,6}', order_number, re.IGNORECASE):
            await cache_service.delete(CacheKeys.TRIAGE_STATE.format(phone=clean_phone))
            await _send_triage_issue_list(clean_phone, order_number)
            return None
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
        return None

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


async def try_authoritative_answer(business_id: str, message: str) -> Optional[str]:
    """
    Checks if the message triggers a hard-coded Knowledge Base answer.
    Returns the answer string if found, else None.
    """
    try:
        # Fetch Config
        config = await db_service.db.business_configs.find_one({"business_id": business_id})
        if not config or "knowledge_base" not in config:
            return None

        kb = config.get("knowledge_base", {})
        message_lower = message.lower().strip()

        # Check all categories (social_media, policies, custom_faqs)
        for category_name, entries in kb.items():
            if not isinstance(entries, dict): 
                continue
            
            for key, entry in entries.items():
                # 🚨 SECURITY FIX: Skip if entry is just a string (e.g. "123 Main St")
                if not isinstance(entry, dict):
                    continue

                if not entry.get("enabled", True): 
                    continue
                
                triggers = entry.get("triggers", [])
                if triggers and any(t.lower() in message_lower for t in triggers):
                    logger.info(f"Authoritative Answer Triggered: {key} (Business: {business_id})")
                    return entry.get("value")
                    
    except Exception as e:
        logger.error(f"Error in authoritative answer check: {e}")
        return None
    return None


async def process_message(phone_number: str, message_text: str, message_type: str, quoted_wamid: str | None, business_id: str = "feelori", profile_name: str = None) -> str | None:
    """
    Processes an incoming message, handling triage states before
    routing to the AI-first intent model.
    """
    try:
        clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)

        # --- EARLY EXIT: Handle Cart/Order Messages Directly ---
        # MOVED TO TOP: We handle this BEFORE upserting the conversation to keep 
        # the "Last Message" in the dashboard clean (avoiding JSON blobs).
        if message_type == "order":
            logger.info(f"Received Order Event from {clean_phone}. Routing directly to Cart Handler.")
            
            # 1. Quick Customer Fetch (Read-Only)
            customer = await db_service.get_customer(clean_phone)
            if not customer:
                customer = {"phone_number": clean_phone}

            # 2. Generate Checkout Link
            response = await handle_cart_submission_direct(message_text, customer, business_id=business_id)
            
            # 3. Log the Interaction Manually (Since we skip the main logging flow)
            timestamp = datetime.utcnow()
            
            # Log Inbound (The Cart)
            await db_service.db.message_logs.insert_one({
                "tenant_id": business_id,
                "business_id": business_id,
                "phone": clean_phone,
                "direction": "inbound",
                "source": "customer",
                "message_type": "order", # Distinct type for analytics
                "text": "Cart Submission", # User-friendly text for UI
                "content": message_text, # Raw JSON payload
                "status": "received",
                "timestamp": timestamp
            })

            # Log Outbound (The Checkout Link)
            if response:
                await db_service.db.message_logs.insert_one({
                    "tenant_id": business_id,
                    "business_id": business_id,
                    "phone": clean_phone,
                    "direction": "outbound",
                    "source": "system",
                    "message_type": "text", # Keep 'text' so Frontend renders the link correctly
                    "text": response,
                    "status": "sending",
                    "timestamp": timestamp
                })
            return response
        # -------------------------------------------------------

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
        
        # Initialize flow_context for new conversations
        new_flow_context = {
            "intent": None,
            "step": None,
            "allowed_next_actions": [],
            "slots": {},
            "version": 1,
            "last_updated": now
        }
        
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
                    "assigned_to": None,
                    "flow_context": new_flow_context
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
        
        # --- ESCAPE HATCH: Force Re-enable AI on Keywords ---
        # If AI is disabled, check if user is trying to restart via "Start" or "Menu".
        # This fixes the "Stuck in Human Mode" loop.
        if conversation.get("ai_enabled", True) is False:
            escape_keywords = {"start", "menu", "restart", "reset", "bot", "talk to bot"}
            clean_text_check = (message_text or "").lower().strip()

            if clean_text_check in escape_keywords:
                logger.info(f"🚨 Escape Hatch Triggered by {clean_phone} with '{clean_text_check}'. Re-enabling AI.")

                # 1. Update DB: Set ai_enabled=True and status=open
                await db_service.db.conversations.update_one(
                    {"_id": conversation["_id"]},
                    {
                        "$set": {
                            "ai_enabled": True,
                            "status": "open",
                            "ai_paused_by": None,
                            "updated_at": datetime.utcnow()
                        }
                    }
                )

                # 2. Update Customer DB (Sync conversation_mode)
                await db_service.db.customers.update_one(
                    {"phone_number": clean_phone},
                    {"$set": {"conversation_mode": "bot"}}
                )

                # 3. Update local 'conversation' dict so we don't hit the suppression block below
                conversation["ai_enabled"] = True
                conversation["status"] = "open"
        # ----------------------------------------------------
        
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

        # ✅ MOVED HERE: Define customer before using it
        customer = await get_or_create_customer(clean_phone, profile_name=profile_name)

        # --- AUTHORITATIVE KNOWLEDGE CHECK ---
        # Facts (KB) → Explicit shortcuts → Workflow memory → Search → AI (LAST)
        if message_type == "text":
            exact_answer = await try_authoritative_answer(business_id, message_text)
            if exact_answer:
                await whatsapp_service.send_message(clean_phone, exact_answer, business_id=business_id)
                return None
        # -------------------------------------

        # --- GIFTING INTENT SHORTCUT ---
        if message_type == "text":
            gift_keywords = {"wife", "husband", "gift", "anniversary", "birthday", "suggest", "recommend"}
            if any(k in message_text.lower() for k in gift_keywords):
                await cache_service.set(
                    CacheKeys.LAST_BOT_QUESTION.format(phone=clean_phone),
                    "offer_bestsellers",
                    ttl=900
                )
                await whatsapp_service.send_message(
                    clean_phone,
                    "That's lovely! 💖 I'd be happy to help you find the perfect gift.\n\n"
                    "Would you like to explore:\n"
                    "✨ Necklaces\n✨ Earrings\n✨ Bangles\n\n"
                    "Or see our *Bestsellers*?",
                    business_id=business_id
                )
                return None
        # ---------------------------------

        # --- SHORTCUT HANDLER: Explicit Buying Intent ---
        if message_type == "text":
            normalized_msg = message_text.lower().strip()
            buying_intent_keywords = ["new order", "buy now", "place order", "i will order"]
            
            if any(keyword in normalized_msg for keyword in buying_intent_keywords):
                buying_intent_response = (
                    "Great! 🎉 I can help you with that. Are you looking for:\n\n"
                    "1️⃣ Necklaces\n"
                    "2️⃣ Earrings\n"
                    "3️⃣ Bangles\n"
                    "4️⃣ Something else?"
                )
                await whatsapp_service.send_message(clean_phone, buying_intent_response, business_id=business_id)
                return None
        # -------------------------------------------------

        # --- PHASE 4.1: Initialize workflow on first broadcast reply ---
        # Check if this is a broadcast reply that needs workflow initialization
        if (conversation and 
            conversation.get("campaign_context") and 
            conversation.get("campaign_context", {}).get("campaign_type") == "broadcast"):
            
            # Get flow_context from conversation
            flow_context_dict = conversation.get("flow_context")
            flow_context_intent = None
            if flow_context_dict:
                flow_context_intent = flow_context_dict.get("intent")
            
            # Only initialize if flow_context.intent is None (no workflow started yet)
            if flow_context_intent is None:
                # Check if this is the FIRST inbound message after the broadcast
                # by checking if there are previous inbound messages after the broadcast timestamp
                campaign_context = conversation.get("campaign_context", {})
                entry_timestamp = campaign_context.get("entry_timestamp")
                
                is_first_reply = True
                if entry_timestamp:
                    # Check if there are any inbound messages after the broadcast timestamp
                    # The current message may or may not be logged yet, so we check:
                    # - If count is 0: definitely first reply
                    # - If count is 1: check if it's very recent (likely the current message being processed)
                    # - If count > 1: definitely not first reply
                    previous_inbound_count = await db_service.db.message_logs.count_documents(
                        {
                            "phone": clean_phone,
                            "direction": "inbound",
                            "timestamp": {"$gt": entry_timestamp}
                        }
                    )
                    
                    if previous_inbound_count > 1:
                        # More than one message after broadcast, not the first reply
                        is_first_reply = False
                    elif previous_inbound_count == 1:
                        # One message found - check if it's very recent (likely current message)
                        recent_message = await db_service.db.message_logs.find_one(
                            {
                                "phone": clean_phone,
                                "direction": "inbound",
                                "timestamp": {"$gt": entry_timestamp}
                            },
                            sort=[("timestamp", -1)]
                        )
                        if recent_message:
                            # If the message is within the last 5 seconds, it's likely the current message
                            message_time = recent_message.get("timestamp")
                            if message_time:
                                time_diff = (now - message_time).total_seconds()
                                if time_diff > 5:
                                    # Message is older than 5 seconds, not the current one
                                    is_first_reply = False
                                # If time_diff <= 5, assume it's the current message (first reply)
                
                if is_first_reply:
                    # Map campaign_type "broadcast" → initial workflow intent "marketing_interest"
                    from app.models.conversation import Conversation
                    from app.workflows.engine import apply_workflow_proposal
                    
                    # Convert dict to Conversation model for engine
                    conversation_obj = Conversation(**conversation)
                    
                    # Propose workflow initialization
                    proposed_workflow = {
                        "intent": "marketing_interest",
                        "step": "capture_interest"
                    }
                    
                    engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)
                    
                    # Persist flow_context ONLY if engine.applied == True
                    if engine_result["applied"] and engine_result["updated_flow_context"]:
                        updated_fc = engine_result["updated_flow_context"]
                        # Get current version for optimistic locking
                        current_version = conversation_obj.flow_context.version if conversation_obj.flow_context else None
                        
                        # Build query with version check for optimistic locking
                        query = {"_id": conversation_id}
                        if current_version is not None:
                            query["flow_context.version"] = current_version
                        else:
                            # If no flow_context exists, match on absence
                            query["$or"] = [
                                {"flow_context": {"$exists": False}},
                                {"flow_context": None}
                            ]
                        
                        # Atomic update with optimistic locking
                        matched_count = await db_service.db.conversations.update_one(
                            query,
                            {
                                "$set": {
                                    "flow_context": updated_fc.model_dump(mode="json")
                                }
                            }
                        )
                        
                        if matched_count.matched_count == 0:
                            logger.warning(f"Optimistic lock failed for conversation {conversation_id}. Workflow initialization not applied.")
                        else:
                            logger.debug(f"Workflow initialized for broadcast reply: conversation {conversation_id}, intent: marketing_interest")
                            # Update conversation dict for subsequent use
                            conversation["flow_context"] = updated_fc.model_dump(mode="json")
        # --- END OF PHASE 4.1 ---

        # --- PHASE 4.1 PROMPT 5: Advance Marketing Workflow ---
        if conversation and conversation.get("flow_context", {}).get("intent") == "marketing_interest":
            flow_context = conversation["flow_context"]
            current_step = flow_context.get("step")

            # Step: capture_interest → identify_category
            if current_step == "capture_interest":
                normalized = message_text.lower().strip()

                category_map = {
                    "earring": "earrings",
                    "earrings": "earrings",
                    "necklace": "necklaces",
                    "necklaces": "necklaces",
                    "bangle": "bangles",
                    "bangles": "bangles",
                    "bracelet": "bangles",
                }

                matched_category = None
                for key, value in category_map.items():
                    if key in normalized:
                        matched_category = value
                        break

                if matched_category:
                    from app.models.conversation import Conversation
                    from app.workflows.engine import apply_workflow_proposal

                    conversation_obj = Conversation(**conversation)

                    proposed_workflow = {
                        "step": "identify_category",
                        "slots_to_update": {
                            "category": matched_category
                        }
                    }

                    engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)

                    if engine_result["applied"] and engine_result["updated_flow_context"]:
                        updated_fc = engine_result["updated_flow_context"]
                        current_version = conversation_obj.flow_context.version

                        await db_service.db.conversations.update_one(
                            {
                                "_id": conversation["_id"],
                                "flow_context.version": current_version
                            },
                            {
                                "$set": {
                                    "flow_context": updated_fc.model_dump(mode="json")
                                }
                            }
                        )

                        # Update local copy so response logic sees new step
                        conversation["flow_context"] = updated_fc.model_dump(mode="json")
        # --- END PROMPT 5 ---

        # --- PHASE 4.1 PROMPT 6A: Advance Marketing Workflow - Price Capture ---
        if conversation and conversation.get("flow_context", {}).get("intent") == "marketing_interest":
            flow_context = conversation["flow_context"]
            current_step = flow_context.get("step")

            # Step: identify_category → qualified
            if current_step == "identify_category":
                normalized = normalize_price_text(message_text.lower().strip())

                # --- Map Numeric Inputs & Button IDs to Ranges ---
                price_map = {
                    "1": "3000-5000",
                    "2": "5000-10000",
                    "3": "above 10000",
                    "price_3000_5000": "3000-5000",
                    "price_5000_10000": "5000-10000",
                    "price_above_10000": "above 10000"
                }
                if normalized in price_map:
                    normalized = price_map[normalized]
                # ------------------------------------

                # Detect price range intent
                price_range = None
                
                # under / below / less / cheaper → under_2000
                if any(keyword in normalized for keyword in ["under", "below", "less", "cheaper", "< 2k", "<2k", "under 2k", "under 2 k"]):
                    price_range = "under_2000"
                # above / over / more / expensive → above_5000
                elif any(keyword in normalized for keyword in ["above", "over", "more", "expensive", "> 5k", ">5k", "above 5k", "above 5 k"]):
                    price_range = "above_5000"
                # 2k-5k / between / 2000-5000 → 2000_5000
                elif any(keyword in normalized for keyword in ["2k-5k", "2k - 5k", "2k to 5k", "2000-5000", "2000 - 5000", "between", "2-5k", "2-5 k"]):
                    price_range = "2000_5000"
                # 3000-5000 / 3k-5k → 3000_5000
                elif any(keyword in normalized for keyword in ["3000-5000", "3000 - 5000", "3k-5k", "3k - 5k", "3k to 5k", "3-5k", "3-5 k"]):
                    price_range = "3000_5000"
                # 5000-10000 / 5k-10k → 5000_10000
                elif any(keyword in normalized for keyword in ["5000-10000", "5000 - 10000", "5k-10k", "5k - 10k", "5k to 10k", "5-10k", "5-10 k"]):
                    price_range = "5000_10000"
                # above 10000 / above 10k → above_10000
                elif any(keyword in normalized for keyword in ["above 10000", "above 10k", "above 10 k", "over 10000", "over 10k", "> 10k", ">10k"]):
                    price_range = "above_10000"

                if price_range:
                    from app.models.conversation import Conversation
                    from app.workflows.engine import apply_workflow_proposal

                    conversation_obj = Conversation(**conversation)

                    proposed_workflow = {
                        "step": "qualified",
                        "slots_to_update": {
                            "price_range": price_range
                        }
                    }

                    engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)

                    if engine_result["applied"] and engine_result["updated_flow_context"]:
                        updated_fc = engine_result["updated_flow_context"]
                        current_version = conversation_obj.flow_context.version

                        await db_service.db.conversations.update_one(
                            {
                                "_id": conversation["_id"],
                                "flow_context.version": current_version
                            },
                            {
                                "$set": {
                                    "flow_context": updated_fc.model_dump(mode="json")
                                }
                            }
                        )

                        # Update local copy so response logic sees new step
                        conversation["flow_context"] = updated_fc.model_dump(mode="json")
        # --- END PROMPT 6A ---

        # --- MARKETING WORKFLOW AUTOMATION ---
        flow_context = conversation.get("flow_context")
        if flow_context and flow_context.get("intent") == "marketing_interest":
            step = flow_context.get("step")
            slots = flow_context.get("slots", {})

            if step == "capture_interest":
                return string_service.get_formatted_string(
                    "MARKETING_CAPTURE_INTEREST", business_id=business_id
                )

            if step == "identify_category":
                category = slots.get("category", "these")
                options = {
                    "price_3000_5000": "₹3k – ₹5k",
                    "price_5000_10000": "₹5k – ₹10k",
                    "price_above_10000": "Above ₹10k"
                }
                
                await whatsapp_service.send_quick_replies(
                    clean_phone,
                    f"Great choice! ✨ We have beautiful *{category}* available.\n\nPlease choose your preferred price range:",
                    options,
                    business_id=business_id
                )
                return None

            if step == "qualified":
                # A) Idempotency Guard
                if flow_context.get("metadata", {}).get("products_sent"):
                    return None

                # B) Manual Acknowledgement Message
                await whatsapp_service.send_message(
                    clean_phone,
                    "Got it 👍 Let me find the best options for you.",
                    business_id=business_id
                )

                # C) Product Fetch & Carousel Send
                from app.services.product_selection_service import fetch_and_select_products

                products = await fetch_and_select_products(
                    category=flow_context["slots"].get("category"),
                    price_range=flow_context["slots"].get("price_range"),
                    business_id=business_id
                )

                await whatsapp_service.send_behavioral_product_carousel(
                    to_phone=clean_phone,
                    product_list=products,
                    business_id=business_id
                )

                # D) Workflow Completion & Persistence
                from app.models.conversation import Conversation

                conversation_obj = Conversation(**conversation)
                current_version = conversation_obj.flow_context.version
                
                # Work with dict representation to add metadata
                updated_fc_dict = conversation_obj.flow_context.model_dump(mode="json")
                updated_fc_dict["step"] = "completed"
                # Manual version increment (workflow engine not used for this terminal step)
                updated_fc_dict["version"] = current_version + 1
                updated_fc_dict["last_updated"] = datetime.utcnow().isoformat()
                
                # Initialize metadata if it doesn't exist
                if "metadata" not in updated_fc_dict:
                    updated_fc_dict["metadata"] = {}
                updated_fc_dict["metadata"]["products_sent"] = True
                updated_fc_dict["metadata"]["abandoned_cart"] = {
                    "status": "pending",
                    "first_shown_at": datetime.utcnow().isoformat(),
                    "last_nudge_at": None,
                    "nudge_count": 0
                }
                updated_fc_dict["metadata"]["shown_product_ids"] = [p["product_id"] for p in products]

                await db_service.db.conversations.update_one(
                    {
                        "_id": conversation["_id"],
                        "flow_context.version": current_version
                    },
                    {
                        "$set": {
                            "flow_context": updated_fc_dict
                        }
                    }
                )

                # Update in-memory copy to prevent re-entry
                conversation["flow_context"] = updated_fc_dict

                return None
        # --- END MARKETING WORKFLOW AUTOMATION ---

        # --- MODIFIED BROADCAST REPLY CHECK ---
        # Check if this is a reply to a broadcast message
        last_outbound = await db_service.get_last_outbound_message(clean_phone)
        if last_outbound and last_outbound.get("source") == "broadcast":
            logger.info(f"Detected reply to broadcast for {clean_phone}")
            
            # 1. CHECK FOR INTENT FIRST
            # If the user says a keyword, we want the AI/Workflow to handle it, NOT this generic block.
            bypass_keywords = {
                "hello", "hi", "hey", "start", "menu", "buy", "shop", 
                "earring", "necklace", "bangle", "ring", "show", "price", "cost"
            }
            message_lower = message_text.lower().strip()
            # Simple check: is it a keyword OR does it look like a price filter ("under 2000")?
            is_marketing_intent = (
                any(k in message_lower for k in bypass_keywords) or 
                "under" in message_lower or 
                "above" in message_lower
            )

            # 2. Check if marketing workflow is ALREADY active
            flow_context_dict = conversation.get("flow_context") if conversation else None
            marketing_workflow_active = (
                flow_context_dict and 
                flow_context_dict.get("intent") == "marketing_interest"
            )
            
            # 3. DECISION: Only trap if it's NOT marketing intent AND NOT active workflow
            if not marketing_workflow_active and not is_marketing_intent:
                # Marketing workflow is NOT active - create human ticket as before
                triage_ticket = {
                    "customer_phone": clean_phone,
                    "order_number": "N/A",
                    "issue_type": "broadcast_reply",
                    "status": "human_needed",
                    "business_id": "feelori",
                    "assigned_to": None,
                    "image_media_id": None,
                    "created_at": datetime.utcnow()
                }
                await db_service.db.triage_tickets.insert_one(triage_ticket)
                return "Thanks for replying to our update! A team member will be with you shortly."
            
            # If we reach here, we fall through to the rest of process_message (AI/Workflow)
            logger.info(f"Bypassing generic broadcast reply for intent: {message_text}")
        # --------------------------------------

        # --- PHASE 4.4: Abandoned Cart Recovery Handler ---
        abandoned_cart = conversation.get("flow_context", {}).get("metadata", {}).get("abandoned_cart", {})
        is_abandoned_pending = abandoned_cart.get("status") == "pending"
        nudge_count = abandoned_cart.get("nudge_count", 0)

        message_lower = message_text.lower().strip()
        recovery_keywords = {"yes", "show", "show again", "ok", "okay", "yep", "sure", "please"}
        is_recovery_reply = message_lower in recovery_keywords

        if is_abandoned_pending and nudge_count >= 1 and is_recovery_reply:
            if abandoned_cart.get("reshown"):
                logger.info(f"Skipping recovery for {clean_phone}: already reshown.")
                return None

            logger.info(f"User {clean_phone} recovered abandoned cart. Resending products.")

            shown_ids = conversation.get("flow_context", {}).get("metadata", {}).get("shown_product_ids", [])
            products_to_send = []

            if shown_ids:
                from app.services.shopify_service import shopify_service
                from app.services.product_selection_service import _extract_numeric_id

                for pid in shown_ids:
                    gid = f"gid://shopify/Product/{pid}" if not str(pid).startswith("gid://") else pid
                    product = await shopify_service.get_product_by_id(gid, business_id=business_id)

                    if product and product.availability == "in_stock":
                        product_url = shopify_service.get_product_page_url(product.handle, business_id=business_id)
                        product_dict = {
                            "product_id": _extract_numeric_id(product.id),
                            "title": product.title,
                            "price": product.price,
                            "currency": product.currency,
                            "image_url": product.image_url,
                            "product_url": product_url
                        }
                        products_to_send.append(product_dict)

            if products_to_send:
                await whatsapp_service.send_behavioral_product_carousel(
                    to_phone=clean_phone,
                    product_list=products_to_send,
                    business_id=business_id
                )

                now_iso = datetime.utcnow().isoformat()

                await db_service.db.conversations.update_one(
                    {
                        "_id": conversation["_id"],
                        "flow_context.metadata.abandoned_cart.nudge_count": nudge_count
                    },
                    {
                        "$set": {
                            "flow_context.metadata.abandoned_cart.status": "completed",
                            "flow_context.metadata.abandoned_cart.reshown": True,
                            "flow_context.metadata.abandoned_cart.recovered_at": now_iso
                        }
                    }
                )

                return None
            else:
                await whatsapp_service.send_message(
                    clean_phone,
                    "I checked, but those items are currently out of stock 😔 Would you like to see our latest arrivals?",
                    business_id=business_id
                )
                return None
        # --------------------------------------------------

        # --- Refactored State Handling ---
        if response := await _handle_security_verification(clean_phone, message_text, customer):
            return response
        if response := await _handle_triage_flow(clean_phone, message_text, message_type, business_id=business_id):
            return response
        # --- End of Refactored State Handling ---

        # --- Context-Aware Response Handling (LAST_BOT_QUESTION) ---
        last_question_raw = await cache_service.redis.get(CacheKeys.LAST_BOT_QUESTION.format(phone=clean_phone))
        if last_question_raw:
            # 1. Decode safely (handle bytes or string)
            if isinstance(last_question_raw, bytes):
                last_question = last_question_raw.decode('utf-8')
            else:
                last_question = str(last_question_raw)
            
            clean_msg = message_text.lower().strip()
            cache_key = CacheKeys.LAST_BOT_QUESTION.format(phone=clean_phone)
            
            # 2. Handle "offer_bestsellers"
            if last_question == "offer_bestsellers":
                if clean_msg in default_rules.AFFIRMATIVE_RESPONSES:
                    response = await handle_bestsellers(customer=customer, business_id=business_id)
                    await cache_service.redis.delete(cache_key)
                    return response
                elif clean_msg in default_rules.NEGATIVE_RESPONSES:
                    await cache_service.redis.delete(cache_key)
                    return "No problem! Let me know if there's anything else I can help you find. ✨"
            
            # --- HANDLE ESCALATION REASON (Sales vs Support) ---
            if last_question == "escalation_reason":
                # Clear the state immediately
                await cache_service.redis.delete(cache_key)
                
                normalized_msg = message_text.lower().strip()
                
                # Option 1: Existing Order -> Proceed to Order Lookup
                if normalized_msg == "1" or "order" in normalized_msg:
                    return await handle_human_escalation(
                        clean_phone, 
                        message_text, 
                        business_id, 
                        profile_name=profile_name, 
                        skip_menu=True  # <--- Bypass menu to run lookup
                    )
                    
                # Option 2: New Inquiry -> Smart Gatekeeper + Admin Alert
                elif normalized_msg == "2" or "new" in normalized_msg or "inquiry" in normalized_msg:
                    config = get_business_config(business_id)
                    
                    # --- 1. Create Dashboard Ticket (So it shows in "Needs Attention") ---
                    # We use status="pending" so it appears in the default Triage view
                    triage_ticket = {
                        "customer_phone": clean_phone,
                        "order_number": "N/A",
                        "issue_type": "sales_inquiry",
                        "status": "pending",
                        "business_id": business_id,
                        "assigned_to": None,
                        "created_at": datetime.utcnow()
                    }
                    await db_service.db.triage_tickets.insert_one(triage_ticket)

                    # --- 2. Check Time (IST) ---
                    # Simple UTC+5:30 conversion
                    now_utc = datetime.utcnow()
                    now_ist = now_utc + timedelta(hours=5, minutes=30)
                    current_hour = now_ist.hour
                    
                    # Open between 11 AM (11) and 10 PM (22)
                    is_open = 11 <= current_hour < 22

                    # --- 3. Send Admin Alert (WhatsApp) ---
                    if config.admin_phone:
                        # Dedup Check: Don't spam admin if already alerted in last 1 hour
                        alert_key = f"sales_alert_sent:{clean_phone}"
                        already_alerted = await cache_service.redis.get(alert_key)
                        
                        if not already_alerted:
                            alert_msg = (
                                f"🔔 *New Sales Lead ({business_id})*\n"
                                f"👤 Customer: +{clean_phone}\n"
                                f"⏰ Time: {now_ist.strftime('%I:%M %p')}\n"
                                f"📂 Status: {'✅ User told to call' if is_open else '💤 User told we are closed'}\n"
                                f"Ticket created in Dashboard."
                            )
                            # Send non-blocking alert
                            asyncio.create_task(
                                whatsapp_service.send_message(config.admin_phone, alert_msg, business_id=business_id)
                            )
                            # Mark as sent for 1 hour
                            await cache_service.redis.set(alert_key, "1", ex=3600)

                    # --- 4. Reply to User ---
                    if is_open:
                        await whatsapp_service.send_message(
                            clean_phone,
                            f"Perfect. 🌟 We are online!\n\n"
                            f"Please call or WhatsApp our team directly:\n"
                            f"📞 {config.support_phone}\n\n"
                            "Mention that you are looking for a *New Order*.",
                            business_id=business_id
                        )
                        return None
                    else:
                        await whatsapp_service.send_message(
                            clean_phone,
                            f"Thanks for reaching out! 🌙\n\n"
                            f"Our sales team is currently offline (Open 11 AM – 10 PM IST).\n\n"
                            f"✅ I have created a priority request for you.\n"
                            f"My team will contact you here as soon as we open tomorrow morning!",
                            business_id=business_id
                        )
                        return None
                
                # Invalid Input -> Default to Option 2 (Safer than looping)
                else:
                     config = get_business_config(business_id)
                     await whatsapp_service.send_message(
                        clean_phone,
                        f"Please reach out to us directly:\n📞 {config.support_phone}",
                        business_id=business_id
                    )
                     return None
            
            # 3. Handle "offer_unfiltered_products"
            elif last_question == "offer_unfiltered_products":
                if clean_msg in default_rules.AFFIRMATIVE_RESPONSES:
                    response = await handle_show_unfiltered_products(customer=customer, business_id=business_id)
                    await cache_service.redis.delete(cache_key)
                    return response
                elif clean_msg in default_rules.NEGATIVE_RESPONSES:
                    await cache_service.redis.delete(cache_key)
                    return "No problem! Let me know if there's anything else I can help you find. ✨"
            
            # 4. Handle "awaiting_order_number"
            elif last_question == "awaiting_order_number":
                # User is replying to "Please give me your order ID"
                # We extract the first sequence of 4 or more digits
                number_match = re.search(r'\d{4,}', message_text)
                if number_match:
                    order_number = number_match.group(0)
                    logger.info(f"Contextual Order Lookup: '{message_text}' -> detected '{order_number}'")
                    await cache_service.redis.delete(cache_key)
                    return await route_message("order_detail_inquiry", clean_phone, order_number, customer, quoted_wamid, business_id=business_id)
                # If no number match, don't delete cache yet (user might retry)
            
            # 5. If no match found (user changed topic), leave cache for now
            # It will be overwritten by new context or expire naturally
        # ---------------------------------------------------------


        if message_type == "interactive" or message_text.startswith("visual_search_"):
            intent = await analyze_intent(message_text, message_type, customer, quoted_wamid)
            route_result = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
            # Handle tuple return (text_response, proposed_workflow)
            if isinstance(route_result, tuple):
                response, proposed_workflow = route_result
            else:
                response = route_result
                proposed_workflow = None
            
            # CENTRALIZED WORKFLOW APPLICATION (before message sending)
            if proposed_workflow and conversation:
                from app.models.conversation import Conversation
                from app.workflows.engine import apply_workflow_proposal
                
                conversation_obj = Conversation(**conversation)
                engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)
                
                if engine_result["applied"] and engine_result["updated_flow_context"]:
                    updated_fc = engine_result["updated_flow_context"]
                    current_version = conversation_obj.flow_context.version if conversation_obj.flow_context else None
                    
                    query = {"_id": conversation_id}
                    if current_version is not None:
                        query["flow_context.version"] = current_version
                    else:
                        query["$or"] = [
                            {"flow_context": {"$exists": False}},
                            {"flow_context": None}
                        ]
                    
                    await db_service.db.conversations.update_one(
                        query,
                        {
                            "$set": {
                                "flow_context": updated_fc.model_dump(mode="json")
                            }
                        }
                    )
            
            return response[:4096] if response else None

        # --- TIGHTENED: Flexible Order Number Detection (No raw numbers) ---
        # Catches: "#1234", "Order 1234", "Its 1234", "No. 1234", "Is 1234"
        # Does NOT catch: "5000" (raw numbers are price filters, not orders)
        order_match = re.search(r'(?:#|order\s+|no\.?\s+|its\s+|is\s+)([A-Z]{0,3}\d{4,6})\b', message_text.strip(), re.IGNORECASE)

        if order_match:
            # We found an order number!
            clean_order_number = order_match.group(1)
            logger.info(f"Order number detected via Regex: {clean_order_number}. Routing directly.")

            # Route directly to the order detail handler
            response = await route_message("order_detail_inquiry", clean_phone, clean_order_number, customer, quoted_wamid, business_id=business_id)
            return response[:4096] if response else None
        # -------------------------------------------------

        if quoted_wamid:
            last_product_raw = await cache_service.redis.get(CacheKeys.LAST_SINGLE_PRODUCT.format(phone=clean_phone))
            if last_product_raw:
                logger.info(f"Detected contextual reply (quoted_wamid: {quoted_wamid}) about a product.")
                intent = "contextual_product_question"
                handler_result = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
                # Handle tuple return (text_response, proposed_workflow)
                if isinstance(handler_result, tuple):
                    response, proposed_workflow = handler_result
                else:
                    response = handler_result
                    proposed_workflow = None
                
                # CENTRALIZED WORKFLOW APPLICATION
                if proposed_workflow and conversation:
                    from app.models.conversation import Conversation
                    from app.workflows.engine import apply_workflow_proposal
                    
                    conversation_obj = Conversation(**conversation)
                    engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)
                    
                    if engine_result["applied"] and engine_result["updated_flow_context"]:
                        updated_fc = engine_result["updated_flow_context"]
                        current_version = conversation_obj.flow_context.version if conversation_obj.flow_context else None
                        
                        query = {"_id": conversation_id}
                        if current_version is not None:
                            query["flow_context.version"] = current_version
                        else:
                            query["$or"] = [
                                {"flow_context": {"$exists": False}},
                                {"flow_context": None}
                            ]
                        
                        await db_service.db.conversations.update_one(
                            query,
                            {
                                "$set": {
                                    "flow_context": updated_fc.model_dump(mode="json")
                                }
                            }
                        )
                
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

        proposed_workflow = None  # Initialize for centralized workflow application

        if ai_intent == "product_search":
            if conversation.get("flow_context", {}).get("intent") == "marketing_interest":
                logger.info("Skipping product_search because marketing workflow is active.")
                return None
            response = await handle_product_search(message=ai_keywords, customer=customer, phone_number=clean_phone, quoted_wamid=quoted_wamid, business_id=business_id)
        
        elif ai_intent == "human_escalation":
             response = await handle_human_escalation(phone_number=clean_phone, message_text=message_text, business_id=business_id, profile_name=profile_name, skip_menu=False)

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
            handler_result = await handle_general_inquiry(
                message=ai_keywords,
                customer=customer,
                phone_number=clean_phone,
                quoted_wamid=quoted_wamid,
                business_id=business_id  # <--- CRITICAL FIX
            )
            # Handle tuple return (text_response, proposed_workflow)
            if isinstance(handler_result, tuple):
                response, proposed_workflow = handler_result
            else:
                response = handler_result
                proposed_workflow = None

        else: 
            logger.debug("AI intent not definitive, running rule-based analyzer.")
            intent = await analyze_intent(message_text, message_type, customer, quoted_wamid)
            route_result = await route_message(intent, clean_phone, message_text, customer, quoted_wamid, business_id=business_id)
            # Handle tuple return (text_response, proposed_workflow)
            if isinstance(route_result, tuple):
                response, proposed_workflow = route_result
            else:
                response = route_result
                proposed_workflow = None
        
        # CENTRALIZED WORKFLOW APPLICATION
        # Apply workflow proposal if present, before message sending
        if proposed_workflow and conversation:
            from app.models.conversation import Conversation
            from app.workflows.engine import apply_workflow_proposal
            
            # Convert dict to Conversation model for engine
            conversation_obj = Conversation(**conversation)
            engine_result = apply_workflow_proposal(conversation_obj, proposed_workflow)
            
            # Persist if applied
            if engine_result["applied"] and engine_result["updated_flow_context"]:
                updated_fc = engine_result["updated_flow_context"]
                # Get current version for optimistic locking
                current_version = conversation_obj.flow_context.version if conversation_obj.flow_context else None
                
                # Build query with version check for optimistic locking
                query = {"_id": conversation_id}
                if current_version is not None:
                    query["flow_context.version"] = current_version
                else:
                    # If no flow_context exists, match on absence
                    query["$or"] = [
                        {"flow_context": {"$exists": False}},
                        {"flow_context": None}
                    ]
                
                # Atomic update with optimistic locking
                # Match on conversation._id AND flow_context.version
                await db_service.db.conversations.update_one(
                    query,
                    {
                        "$set": {
                            "flow_context": updated_fc.model_dump(mode="json")
                        }
                    }
                )
                # If version mismatch (matched_count == 0), reject safely (do nothing)
        
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

async def handle_cart_submission_direct(items_json: str, customer: Dict, business_id: str = "feelori") -> str:
    """
    Directly handles cart submission without AI/Intent analysis.
    Creates a Shopify Cart and returns the checkout URL.
    """
    try:
        items = json.loads(items_json)
        
        # Safety Check: Ensure items is actually a list
        if not isinstance(items, list):
            logger.error(f"Malformed order payload received: {items_json}")
            return "I had trouble reading your cart. Please try again."

        if not items:
            return "Your cart seems empty! Why not add some shiny things? ✨"

        # 1. Create Cart
        cart_id = await shopify_service.create_cart(business_id=business_id)
        if not cart_id:
            return "I had a little trouble preparing your cart. Please try again in a moment!"

        # 2. Add Items (Using strictly the IDs provided by WhatsApp)
        # Note: These are Variant IDs (Meta Content IDs), so we prefix them correctly.
        for item in items:
            variant_id = item.get("product_retailer_id")
            qty = item.get("quantity", 1)
            
            if variant_id:
                variant_gid = f"gid://shopify/ProductVariant/{variant_id}"
                await shopify_service.add_item_to_cart(cart_id, variant_gid, qty, business_id=business_id)

        # 3. Get Checkout URL
        checkout_url = await shopify_service.get_checkout_url(cart_id, business_id=business_id)
        
        if checkout_url:
            return (
                f"Great choice! 🛍️ I've prepared your secure checkout link.\n\n"
                f"👉 *Tap here to buy:* {checkout_url}\n\n"
                "Let me know if you need help with anything else!"
            )
        return "I couldn't generate a checkout link right now. Please try again."

    except Exception as e:
        logger.error(f"Error in direct cart handler: {e}", exc_info=True)
        return "Something went wrong creating your checkout. Please try again."

def _get_whatsapp_product_id(product: Product) -> Optional[str]:
    """
    Extracts the correct ID for WhatsApp Catalog (Meta).
    STRICT MODE: Meta ONLY accepts Variant IDs (Content IDs).
    """
    if not product:
        return None

    candidate_id = None

    # 1. Try explicit 'first_variant_id' field (if populated by service)
    if hasattr(product, "first_variant_id") and product.first_variant_id:
        candidate_id = product.first_variant_id

    # 2. Try fetching ID from the first variant in the list
    if not candidate_id and hasattr(product, "variants") and product.variants:
         v = product.variants[0]
         # Handle both dict and object access
         candidate_id = v.get("id") if isinstance(v, dict) else getattr(v, "id", None)

    if not candidate_id:
        # LOG WARNING: This is likely why your card failed!
        return None

    # 3. Clean up GID (gid://shopify/ProductVariant/12345 -> 12345)
    id_str = str(candidate_id)
    if 'gid://' in id_str:
        return id_str.rstrip('/').split('/')[-1]
    
    return id_str


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
    # --- NEW: Extract Order Data for Cart Checkout ---
    elif msg_type == "order":
        # Extract the list of items (Variant IDs and quantities)
        # We dump this to a JSON string so it can pass through the existing text-based pipeline
        items = message.get("order", {}).get("product_items", [])
        return json.dumps(items)
    # -------------------------------------------------
    return f"[Unsupported message type: '{msg_type}']"

def _analyze_interactive_intent(message: str) -> str:
    """Analyze intent for interactive messages based on their prefix."""
    for prefix, intent in default_rules.INTERACTIVE_PREFIXES.items():
        if message.startswith(prefix):
            return intent
    return "interactive_response"

def analyze_text_intent(message_lower: str) -> str:
    """Analyzes intent for text messages using rules from the database."""

    # --- TIGHTENED: Require prefix to avoid matching price filters ---
    # Only matches: "#1234", "Order 1234", "No. 1234" (not raw "5000")
    if re.fullmatch(r'(?:#|order\s*|no\.?\s*)[A-Z]{0,3}\d{4,6}', message_lower.strip(), re.IGNORECASE):
        return "order_detail_inquiry"

    # We apply the same fix to the search regex.
    if re.search(r'(?:#|order\s+|no\.?\s+)[A-Z]{0,3}\d{4,6}', message_lower, re.IGNORECASE):
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

async def route_message(intent: str, phone_number: str, message: str, customer: Dict, quoted_wamid: Optional[str] = None, business_id: str = "feelori") -> Optional[str] | tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Routes the message to the appropriate handler based on intent.
    
    Returns:
        str: Response text for most handlers
        tuple[str, Optional[Dict]]: (response_text, proposed_workflow) for handlers that call generate_response
    """
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
    result = await handler(phone_number=phone_number, message=message, customer=customer, quoted_wamid=quoted_wamid, business_id=business_id)
    # Return as-is (may be string or tuple)
    return result


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
        
        # --- CONTEXT INJECTION (Fix for Price-Only Searches) ---
        # Guard: Only check history if we have a price BUT no text query
        # AND message is a list/tuple (from AI Intent Classifier, not Visual Search captions)
        if not text_query and price_filter and isinstance(message, (list, tuple)):
            # User sent "6000 above" but no product name. Check history.
            last_search_raw = await cache_service.redis.get(CacheKeys.LAST_SEARCH.format(phone=customer['phone_number']))
            
            # Guard: If cache is expired (None), we skip this block (Stale Check)
            if last_search_raw:
                try:
                    last_search_data = json.loads(last_search_raw)
                    last_query = last_search_data.get("query", "")
                    
                    # Extract keywords from the PREVIOUS search
                    # We reuse the QueryBuilder to get just the keywords (ignoring previous price filters)
                    prev_keywords = query_builder._extract_keywords(last_query)
                    
                    if prev_keywords:
                        text_query = " AND ".join(prev_keywords)
                        # Update the display string so logs make sense
                        message_str = f"{text_query} ({original_message})"
                        logger.info(f"Inferred context '{text_query}' for price-only search '{original_message}'")
                        
                        # --- PERSISTENCE FIX ---
                        # Update LAST_SEARCH immediately so "Show more" works with this new context
                        await cache_service.set(
                            CacheKeys.LAST_SEARCH.format(phone=customer['phone_number']),
                            json.dumps({"query": f"{text_query} {original_message}", "page": 1}),
                            ttl=900
                        )
                        # -----------------------
                except Exception as e:
                    logger.warning(f"Failed to inject context for price search: {e}")
        # -------------------------------------------------------
        
        # 4. Use the clean keyword string for logging, display, and cache keys.
        # (Only set if context injection didn't set it)
        try:
            _ = message_str  # Check if already set by context injection
        except NameError:
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

        await _handle_standard_search(filtered_products, message_str, customer, business_id=business_id)
        return None
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
    """
    Handles a request for order details with robust lookup, JIT sync, and security.
    Implements 'Self-Healing' logic: If DB misses, fetch from Shopify, normalize to Integer, and upsert.
    """
    business_id = kwargs.get("business_id", "feelori")
    
    # Extract number (flexible regex)
    order_name_match = re.search(r'#?([a-zA-Z]*\d{4,})', message)
    if not order_name_match:
        return await _handle_unclear_request(customer, message)

    raw_number_str = order_name_match.group(1).upper() # e.g. "1067" (String)
    prefixed_number = f"#{raw_number_str}"             # e.g. "#1067" (String)
    
    logger.info(f"Looking up order: {prefixed_number} (raw: {raw_number_str})")

    # 1. SMART DB LOOKUP (Handle Int vs String mismatch efficiently)
    # Search for: 1067 (Int), "1067" (Str), "#1067" (Str)
    search_candidates = [prefixed_number, raw_number_str]
    if raw_number_str.isdigit():
        search_candidates.append(int(raw_number_str))

    order_from_db = await db_service.db.orders.find_one({
        "order_number": {"$in": search_candidates}
    })

    # --- JIT SYNC START (Self-Healing) ---
    if not order_from_db:
        logger.warning(f"Order {prefixed_number} not found in DB. Attempting JIT Sync from Shopify...")
        
        # A. Fetch from Shopify Real-time
        # Try fetching by name first (most reliable)
        shopify_order = await shopify_service.get_order_by_name(prefixed_number, business_id=business_id)
        
        # Fallback: Try raw number if prefixed failed
        if not shopify_order and raw_number_str.isdigit():
             shopify_order = await shopify_service.get_order_by_name(raw_number_str, business_id=business_id)

        if shopify_order:
            logger.info(f"JIT Sync: Found {prefixed_number} in Shopify. Syncing to DB.")
            
            # B. Extract Phones for Security (FIXED LINTING ERRORS HERE)
            phones = []
            if shopify_order.get("phone"):
                phones.append(shopify_order["phone"])
            if shopify_order.get("customer", {}).get("phone"):
                phones.append(shopify_order["customer"]["phone"])
            if shopify_order.get("billing_address", {}).get("phone"):
                phones.append(shopify_order["billing_address"]["phone"])
            if shopify_order.get("shipping_address", {}).get("phone"):
                phones.append(shopify_order["shipping_address"]["phone"])
            
            unique_phones = list(set([p for p in phones if p]))
            
            # C. Create DB Object (Normalize to Integer if possible, matching your existing schema)
            # Use the order number from Shopify. Try to cast to Int to match your DB history.
            try:
                final_order_num = int(shopify_order.get("order_number", raw_number_str))
            except (ValueError, TypeError):
                # Fallback to string if it contains letters (e.g. "ORD-101")
                final_order_num = str(shopify_order.get("order_number", raw_number_str))
            
            new_order_record = {
                "order_number": final_order_num, 
                "shopify_id": str(shopify_order.get("id")),
                "phone_numbers": unique_phones,
                "status": "synced_jit", # Metadata for auditing
                "synced_via": "jit",    # Explicit source tracking
                "created_at": datetime.utcnow()
            }
            
            # D. Save to Mongo (Idempotent Upsert)
            # Prevents duplicates if two requests come in at once
            await db_service.db.orders.update_one(
                {"order_number": final_order_num},
                {"$setOnInsert": new_order_record},
                upsert=True
            )
            
            # E. Retrieve the inserted document (ensure we have the _id and full structure)
            order_from_db = await db_service.db.orders.find_one({"order_number": final_order_num})
            
        else:
            # F. Truly Not Found
            logger.warning(f"Order {prefixed_number} does not exist in Shopify either.")
            return string_service.get_formatted_string('ORDER_NOT_FOUND_SPECIFIC', business_id=business_id, order_number=prefixed_number)
    # --- JIT SYNC END ---

    # 2. Proceed with Security Check
    # Normalize order name for display
    order_name_display = str(order_from_db.get("order_number", prefixed_number))
    if not order_name_display.startswith("#"):
        order_name_display = f"#{order_name_display}"

    # Check Verification Cache
    is_verified = await cache_service.get(CacheKeys.ORDER_VERIFIED.format(phone=customer['phone_number'], order_name=order_name_display))
    
    if not is_verified:
        order_phones = order_from_db.get("phone_numbers", [])
        
        # Validation: Ensure we have at least one phone number to verify against
        if not order_phones or not isinstance(order_phones[0], str) or len(order_phones[0]) < 4:
            # Fallback for JIT synced orders without phone data (rare but possible)
            logger.error(f"Invalid phone data for order {order_name_display}")
            return "For your security, I cannot display this order because the phone number on file is missing or invalid. Please contact support."

        expected_last_4 = order_phones[0][-4:]
        
        # Set Context for the next reply (Listening Mode)
        await cache_service.set(CacheKeys.AWAITING_ORDER_VERIFICATION.format(phone=customer['phone_number']), json.dumps({
            "order_name": order_name_display,
            "expected_last_4": expected_last_4,
            "attempts": 0  # Initialize counter
        }), ttl=300)

        # Return the challenge
        return f"For your security, please reply with the last 4 digits of the phone number used to place order *{order_name_display}*."

    # 3. Fetch Full Details (Shopify)
    order_to_display = await shopify_service.get_order_by_name(order_name_display, business_id=business_id)
    if not order_to_display:
        return string_service.get_formatted_string('ORDER_NOT_FOUND_SPECIFIC', business_id=business_id, order_number=order_name_display)
        
    await cache_service.delete(CacheKeys.ORDER_VERIFIED.format(phone=customer['phone_number'], order_name=order_name_display))
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

    prompt = ai_service.create_qa_prompt(last_product, message, business_id=business_id)
    try:
        # Fetch flow_context from conversation if available
        flow_context_dict = None
        try:
            clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
            conversation = await db_service.db.conversations.find_one(
                {"external_user_id": clean_phone, "tenant_id": business_id}
            )
            if conversation and conversation.get("flow_context"):
                # MongoDB returns flow_context as a dict
                flow_context_dict = conversation["flow_context"]
        except Exception as e:
            logger.debug(f"Could not fetch flow_context for {phone_number}: {e}")
        
        text_response = await asyncio.wait_for(ai_service.generate_response(prompt, business_id=business_id, flow_context=flow_context_dict), timeout=15.0)
        
        await whatsapp_service.send_product_detail_with_buttons(phone_number, last_product, business_id=business_id)
        return text_response
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
        return None
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
        
        return None
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
        return None
    
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
        return None

    return "Sorry, I couldn't find details for that product."

async def handle_latest_arrivals(customer: Dict, **kwargs) -> Optional[str]:
    """Shows the newest products."""
    business_id = kwargs.get("business_id", "feelori")
    products, _ = await shopify_service.get_products(query="", limit=5, sort_key="CREATED_AT", business_id=business_id)
    
    if not products: 
        return "I couldn't fetch the latest arrivals right now. Please try again shortly."
    
    # ✅ FIX: Send the card BEFORE returning None
    await _send_product_card(
        products=products, 
        customer=customer, 
        header_text="Here are our latest arrivals! ✨", 
        body_text="Freshly added to our collection.", 
        business_id=business_id
    )
    return None

async def handle_bestsellers(customer: Dict, **kwargs) -> Optional[str]:
    """Shows the top-selling products."""
    business_id = kwargs.get("business_id", "feelori")
    
    # 1. Log the business_id being used
    logger.info(f"Fetching bestsellers for business_id: {business_id}")
    
    try:
        # 2. Fetch products and log the result
        products, total_count = await shopify_service.get_products(
            query="", 
            limit=5, 
            sort_key="BEST_SELLING", 
            business_id=business_id
        )
        
        # 3. Log the raw result (count of products)
        logger.info(f"Bestsellers fetch result for {business_id}: {len(products) if products else 0} products returned (total_count: {total_count})")
        
        if not products:
            # 4. If products is empty, log a WARNING with the specific reason
            logger.warning(
                f"Bestsellers fetch returned empty list for business_id={business_id}. "
                f"Total count from API: {total_count}. "
                f"This could indicate: (1) No products in store, (2) API error, (3) Sort key 'BEST_SELLING' not supported."
            )
            return "I couldn't fetch our bestsellers right now. Please try again shortly."
        
        await _send_product_card(
            products=products, 
            customer=customer, 
            header_text="Check out our bestsellers! 🌟", 
            body_text="These are the items our customers love most.", 
            business_id=business_id
        )
        return None
        
    except Exception as e:
        logger.error(f"Error fetching bestsellers for business_id={business_id}: {e}", exc_info=True)
        return "I couldn't fetch our bestsellers right now. Please try again shortly."

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
    return None

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
    """
    Handles visual search with Enterprise Guardrails (Size, Type, Rate Limit).
    """
    try:
        business_id = kwargs.get("business_id", "feelori")
        phone_number = customer["phone_number"]

        # 1. PARSE PAYLOAD
        clean_payload = message.replace("visual_search_", "")
        parts = clean_payload.split("_caption_")
        media_id = parts[0].strip()
        caption = parts[1].strip() if len(parts) > 1 else ""
        
        # 2. RATE LIMIT CHECK (Prevent Abuse: 5 searches / 10 mins)
        is_allowed = await rate_limiter.check_rate_limit(f"visual_search:{phone_number}", limit=5, window=600)
        if not is_allowed:
            return "You've sent a lot of images! Please wait a few minutes before searching again. ⏳"

        # Acknowledge
        await whatsapp_service.send_message(phone_number, "🔍 Analyzing your photo... ✨", business_id=business_id)
        
        # 3. DOWNLOAD & GUARDRAILS
        image_bytes, mime_type = await whatsapp_service.get_media_content(media_id, business_id=business_id)
        
        if not image_bytes:
            return "I couldn't download the image. Please try again."
            
        allowed_mimes = ["image/jpeg", "image/png", "image/webp"]
        if mime_type not in allowed_mimes:
            return "I can only analyze standard images (JPG, PNG). Please try another format."
        
        if len(image_bytes) > 5 * 1024 * 1024: # 5MB limit
            return "That image is a bit too large for me to process. Please try a smaller one."

        # 4. EXECUTE SEARCH
        result = await ai_service.find_products_by_image_content(image_bytes, mime_type, caption, business_id=business_id)
        
        if not result.get('success'):
            # Fallback to text search if caption is strong
            if caption and len(caption) > 4:
                return await handle_product_search(caption, customer, business_id=business_id)
            return result.get('message', "No matches found.")

        # 5. SEND RESULTS
        # Convert dicts back to objects if needed (handled by ai_service now, but good to be safe)
        raw_products = result.get('products', [])
        products = raw_products  # Assuming ai_service returns objects now
        
        # --- LOGIC START ---
        direct_answer = result.get('direct_answer', '')
        confidence = result.get('confidence', 0.0)
        
        # 1. CLEAN UP AI ANSWER (The Muzzle)
        lazy_phrases = [
            "cannot determine the price", "check with the seller", 
            "refer to the product listing", "price of this item from the image",
            "i don't have access to real-time"
        ]
        if any(phrase in direct_answer.lower() for phrase in lazy_phrases):
            logger.info(f"Suppressing lazy AI answer: '{direct_answer}'")
            direct_answer = ""

        # 2. DETERMINE FINAL REPLY
        final_reply = ""
        caption_lower = caption.lower()
        asking_price = any(x in caption_lower for x in ['price', 'cost', 'how much', 'rate', 'rs', 'rupees'])

        if products:
            # SCENARIO A: Products Found (Success)
            top_product = products[0]
            
            # A1. Construct Header
            raw_query = result.get('search_query', '')
            safe_query = (raw_query[:30] + '...') if len(raw_query) > 33 else raw_query
            
            if confidence > 0.85:
                safe_header = f"✨ Matches: '{safe_query}'"
            else:
                safe_header = f"✨ Similar to: '{safe_query}'"
            safe_header = safe_header[:60]  # Hard Guardrail

            # A2. Answer Price Question (DEFENSIVE CHECK)
            if asking_price:
                # Safely get price/currency to prevent crashes if missing
                price = getattr(top_product, "price", None)
                currency = getattr(top_product, "currency", "INR")
                
                if price:
                    final_reply = (
                        f"The item in the first image matches our *{top_product.title}*.\n"
                        f"💰 Price: *{currency} {price}*\n"
                        f"Tap 'View' above to see more details! 🛍️"
                    )
                else:
                    # Fallback if price is hidden/missing
                    final_reply = (
                        f"The item in the first image matches our *{top_product.title}*.\n"
                        f"Please tap 'View' above to check the latest price and availability! 🛍️"
                    )
            else:
                # Use AI's answer if it's NOT lazy, otherwise generic helpful text
                final_reply = direct_answer if direct_answer else "Here are the closest items I found in our collection:"

            # A3. Send Product Card
            await _send_product_card(
                products=products, 
                customer=customer, 
                header_text=safe_header,
                body_text="Tap 'View Items' to shop 👇", 
                business_id=business_id
            )

        else:
            # SCENARIO B: No Products Found (Fallback)
            
            # B1. Determine Message based on Confidence
            if confidence < 0.4:
                # AI was confused
                final_reply = "I couldn't quite identify the jewelry in that image. 🧐\nCould you try a clearer photo, or tell me what you're looking for?"
            else:
                # AI knew it, but we don't have it
                clean_query = result.get('search_query', 'that item').replace('"', '')
                final_reply = (
                    f"I see you're looking for *{clean_query}*, but I couldn't find a close match in our collection right now. 😔\n\n"
                    "Would you like to see our **Bestsellers** instead?"
                )
                await cache_service.set(CacheKeys.LAST_BOT_QUESTION.format(phone=phone_number), "offer_bestsellers", ttl=300)

        # 3. SEND FINAL REPLY (If exists)
        if final_reply:
            await asyncio.sleep(1)  # Small delay for UX pacing
            await whatsapp_service.send_message(phone_number, final_reply, business_id=business_id)
                
        return None

    except Exception as e:
        logger.error(f"Visual search handler error: {e}", exc_info=True)
        return "Something went wrong while searching. Please try again."

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
        
        # --- ADD THIS LINE: Enable Context Memory ---
        await cache_service.set(CacheKeys.LAST_BOT_QUESTION.format(phone=phone_number), "awaiting_order_number", ttl=300)
        # --------------------------------------------
        
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
        phone_number = customer.get("phone_number") or kwargs.get("phone_number")
        context = {"conversation_history": customer.get("conversation_history", [])[-5:]}
        
        # Fetch conversation and flow_context if available
        flow_context_dict = None
        conversation_dict = None
        if phone_number:
            try:
                clean_phone = EnhancedSecurityService.sanitize_phone_number(phone_number)
                conversation_dict = await db_service.db.conversations.find_one(
                    {"external_user_id": clean_phone, "tenant_id": business_id}
                )
                if conversation_dict and conversation_dict.get("flow_context"):
                    # MongoDB returns flow_context as a dict
                    flow_context_dict = conversation_dict["flow_context"]
            except Exception as e:
                logger.debug(f"Could not fetch flow_context for {phone_number}: {e}")
        
        text_response = await asyncio.wait_for(
            ai_service.generate_response(message, context, business_id=business_id, flow_context=flow_context_dict),
            timeout=15.0
        )
        
        return text_response
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


async def handle_human_escalation(phone_number: str, message_text: str, business_id: str, profile_name: str = None, skip_menu: bool = False) -> str:
    """
    STARTS the automated triage flow instead of immediately escalating.
    It proactively finds the user's orders and asks them to confirm.
    """
    # --- ENTERPRISE FORK: Sales vs Support ---
    # If we haven't asked yet, and the user didn't explicitly trigger a specific flow
    if not skip_menu:
        # Set a memory marker so we know to handle the answer next
        await cache_service.set(
            CacheKeys.LAST_BOT_QUESTION.format(phone=phone_number),
            "escalation_reason",
            ttl=300
        )
        
        # Send the Menu
        msg = (
            "To connect you with the right person, is this regarding:\n\n"
            "1️⃣ *An Existing Order* (Status, Issues, Returns)\n"
            "2️⃣ *New Inquiry* (Bulk, Custom, General)\n\n"
            "Please reply with *1* or *2*."
        )
        await whatsapp_service.send_message(phone_number, msg, business_id=business_id)
        return None  # 🚨 CRITICAL: Return immediately to stop order lookup

    # ---------------------------------------------------------
    
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
        return None

    else:
        # 4. MULTIPLE ORDERS FOUND: Ask to select.
        logger.info(f"Triage: Found multiple orders for {phone_number}. Asking to select.")
        
        options = {}
        # FIX: Slice to [:3] to prevent WhatsApp API error (Max 3 buttons)
        for order in recent_orders[:3]: 
            order_num = order.get("order_number")
            options[f"{TriageButtons.SELECT_ORDER_PREFIX}{order_num}"] = f"Order {order_num}"
        
        await whatsapp_service.send_quick_replies(
            phone_number,
            "I'm sorry to hear you're having an issue. I found a few of your recent orders. Which one do you need help with?",
            options,
            business_id=business_id
        )
        return None

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

async def _handle_standard_search(products: List[Product], message: str, customer: Dict, business_id: str = "feelori") -> None:
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
    
    return None

async def _send_product_card(products: List[Product], customer: Dict, header_text: str, body_text: str, business_id: str = "feelori"):
    """
    Sends a rich multi-product message card with X-RAY LOGGING.
    """
    catalog_id = await whatsapp_service.get_catalog_id(business_id=business_id)

    valid_items = []
    filtered_products_for_fallback = []

    logger.info(f"--- X-RAY: Validating {len(products)} products for WhatsApp ---")

    for p in products:
        w_id = _get_whatsapp_product_id(p)
        is_in_stock = (p.availability == "in_stock")
        
        # LOGGING EVERY DECISION
        if is_in_stock and w_id:
            valid_items.append({"product_retailer_id": w_id})
            filtered_products_for_fallback.append(p)
            logger.info(f"✅ ACCEPTED: {p.title} | Meta ID: {w_id}")
        else:
            logger.warning(
                f"⚠️ REJECTED: {p.title}\n"
                f"   - Availability: {p.availability} (Need 'in_stock')\n"
                f"   - Meta ID Found: {w_id} (Need valid ID)"
            )

    # 2. Check: Do we have any valid items to send?
    if valid_items:
        # Success Path: Send the Carousel
        await whatsapp_service.send_multi_product_message(
            to=customer["phone_number"], header_text=header_text, body_text=body_text,
            footer_text="Powered by FeelOri", catalog_id=catalog_id,
            section_title="Products", product_items=valid_items, 
            fallback_products=filtered_products_for_fallback,
            business_id=business_id
        )
    else:
        # SMART FALLBACK: Generate a link based on the products we TRIED to send.
        logger.warning(f"All {len(products)} products failed validation. Sending smart fallback.")

        fallback_url = "https://feelori.com/collections/all"  # Default safety net
        
        if products:
            top_match = products[0]
            
            # PRIORITY 1: Direct Product Link (Best Experience) 🥇
            # If we have a handle, send them straight to the item!
            if hasattr(top_match, 'handle') and top_match.handle:
                fallback_url = f"https://feelori.com/products/{top_match.handle}"
                
            # PRIORITY 2: Fallback to Title Search (Better than Tag Search) 🥈
            # If handle is missing, search for the specific title
            else:
                encoded_title = top_match.title.replace(' ', '+')
                fallback_url = f"https://feelori.com/search?q={encoded_title}"

        fallback_msg = (
            f"{header_text}\n\n"
            f"I found this exact design matching your photo, but I can't display the card here right now. 😔\n\n"
            f"👇 *Tap to view details & price on our website:*\n{fallback_url}"
        )

        await whatsapp_service.send_message(
            to_phone=customer["phone_number"],
            message=fallback_msg,
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
        
        # --- MOVED UP: Extract Message Details Early ---
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
        # -----------------------------------------------

        # --- KILL SWITCH: Bot vs Human Mode with Auto-Release ---
        # Enforce auto-release for stale locks (>30 minutes)
        await db_service.enforce_auto_release(clean_phone)
        
        # Fetch conversation to check mode
        customer = await db_service.get_customer(clean_phone)
        # --- HUMAN MODE CHECK WITH ESCAPE HATCH ---
        if customer and customer.get("conversation_mode") == "human":
            # Check for EXPLICIT "Escape Hatch" keywords to let user self-unlock.
            # NOTE: We exclude generic greetings like "hi"/"hello" so polite users 
            # don't accidentally kick out the human agent.
            escape_keywords = {"start", "menu", "restart", "reset", "bot", "talk to bot"}
            clean_text = message_text.lower().strip()
            
            if clean_text in escape_keywords:
                logger.info(f"User {clean_phone} triggered Escape Hatch with '{clean_text}'. Reverting to Bot Mode.")
                
                # 1. Force-unlock the user immediately in DB
                await db_service.db.customers.update_one(
                    {"phone_number": clean_phone},
                    {
                        "$set": {
                            "conversation_mode": "bot",
                            "conversation_last_mode_change_at": datetime.utcnow()
                        },
                        "$unset": {"conversation_locked_by": ""}
                    }
                )
                
                # 2. Update local state so THIS message gets processed by the bot immediately
                customer["conversation_mode"] = "bot"
                
                # Optional: We could log a system note here if needed
                
            else:
                # Still in human mode and didn't say a magic word -> Suppress Bot
                logger.info(f"Bot suppressed for {clean_phone}: AI is explicitly disabled.")
                return
        # ---------------------------------------------
        # --- END OF KILL SWITCH ---

        # This duplicate check can be simplified now with a dedicated message log
        # but we'll leave it for now for extra safety.
        if await message_queue.is_duplicate_message(wamid, clean_phone):
            logger.info(f"Duplicate message {wamid} from {clean_phone} received, ignoring.")
            return

        if not await security_service.rate_limiter.check_phone_rate_limit(clean_phone):
            logger.warning(f"Rate limit exceeded for {clean_phone}.")
            return

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