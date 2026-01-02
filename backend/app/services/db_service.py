# /app/services/db_service.py

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum

from app.config.settings import settings
from app.models.api import BroadcastGroupCreate, Rule, StringResource
from app.utils.circuit_breaker import RedisCircuitBreaker
from app.utils.metrics import database_operations_counter
from app.services.cache_service import cache_service
from app.services import security_service
from app.services.shopify_service import shopify_service

logger = logging.getLogger(__name__)

# Constants
DEFAULT_QUERY_LIMIT = 100
RECENT_ORDERS_LIMIT = 3
PAGINATION_DEFAULT_LIMIT = 20
CONVERSATION_HISTORY_MAX = 100  # Limit conversation history growth


class OrderStatus(str, Enum):
    """Order fulfillment statuses"""
    PENDING = "Pending"
    NEEDS_STOCK_CHECK = "Needs Stock Check"
    IN_PROGRESS = "In Progress"
    ON_HOLD = "On Hold"
    COMPLETED = "Completed"


class DatabaseService:
    """
    Manages all interactions with MongoDB database including CRUD operations,
    aggregation queries, and webhook processing.
    """

    def __init__(self, mongo_uri: str):
        try:
            self.client = AsyncIOMotorClient(
                mongo_uri,
                maxPoolSize=settings.max_pool_size,
                minPoolSize=settings.min_pool_size,
                tls=settings.mongo_ssl,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            self.db = self.client.get_default_database()
            self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "database")
            logger.info("MongoDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MongoDB client: {e}")
            raise

    # ==================== Helper Methods ====================

    def _sanitize_phone(self, phone_number: str) -> str:
        """
        Centralized phone number sanitization.
        
        Args:
            phone_number: Raw phone number input
            
        Returns:
            Sanitized phone number or empty string if invalid
        """
        if not phone_number:
            return ""
        
        cleaned = security_service.EnhancedSecurityService.sanitize_phone_number(phone_number)
        if not cleaned:
            logger.debug(f"Invalid phone number sanitized to empty: {phone_number[:4]}...")
        return cleaned

    def _extract_phones_from_payload(self, payload: Dict[str, Any]) -> List[str]:
        """
        Extract and sanitize all phone numbers from a Shopify order payload.
        
        Args:
            payload: Shopify order data
            
        Returns:
            List of sanitized phone numbers
        """
        phones = {
            (payload.get("customer") or {}).get("phone"),
            (payload.get("shipping_address") or {}).get("phone"),
            (payload.get("billing_address") or {}).get("phone")
        }
        return [self._sanitize_phone(p) for p in phones if p]

    def _validate_object_id(self, obj_id: str) -> bool:
        """
        Validate MongoDB ObjectId format.
        
        Args:
            obj_id: String to validate
            
        Returns:
            True if valid ObjectId format
        """
        if not obj_id or not isinstance(obj_id, str):
            return False
        return ObjectId.is_valid(obj_id)

    def _serialize_id(self, document: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Convert ObjectId to string for JSON serialization.
        
        Args:
            document: MongoDB document
            
        Returns:
            Document with _id converted to string, or None if input is None
        """
        if document and "_id" in document:
            document["_id"] = str(document["_id"])
        return document

    def _serialize_ids(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Serialize multiple documents."""
        return [self._serialize_id(doc) for doc in documents]

    async def _safe_db_operation(
        self, 
        operation, 
        use_circuit_breaker: bool = True,
        default_return: Any = None
    ) -> Any:
        """
        Execute database operation with consistent error handling.
        
        Args:
            operation: Async callable to execute
            use_circuit_breaker: Whether to use circuit breaker
            default_return: Value to return on failure
            
        Returns:
            Operation result or default_return on failure
        """
        try:
            if use_circuit_breaker:
                return await self.circuit_breaker.call(operation)
            return await operation()
        except Exception as e:
            logger.exception(f"Database operation failed: {type(e).__name__}")
            database_operations_counter.labels(operation="db_error", status="failed").inc()
            return default_return

    def _now_utc(self) -> datetime:
        """Get current UTC timestamp (Python 3.12+ compatible)."""
        return datetime.now(timezone.utc)
    
    def _parse_iso_utc(self, value: Optional[str]) -> Optional[datetime]:
        """Parse Shopify ISO8601 strings to aware UTC datetimes."""
        if not value or not isinstance(value, str):
            return None
        try:
            # Handle Z suffix and ensure timezone awareness
            return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
        except Exception:
            logger.warning(f"Failed to parse timestamp: {value}")
            return None
    
    def _map_shop_domain_to_business_id(self, shop_domain: str) -> str:
        """
        Map Shopify shop domain to business_id for multi-tenancy.
        
        Args:
            shop_domain: Shopify shop domain (e.g., "feelori.myshopify.com")
            
        Returns:
            business_id string (defaults to "feelori" if mapping not found)
        """
        if not shop_domain:
            return "feelori"  # Default fallback
        
        # Extract shop name from domain (e.g., "feelori" from "feelori.myshopify.com")
        shop_name = shop_domain.split(".")[0].lower() if "." in shop_domain else shop_domain.lower()
        
        # Map known shop domains to business_ids
        domain_to_business = {
            "feelori": "feelori",
            "goldencollections": "goldencollections",
            "godjewellery9": "godjewellery9"
        }
        
        return domain_to_business.get(shop_name, "feelori")  # Default to feelori if unknown
    
    # ==================== Index Management ====================

    async def create_indexes(self) -> None:
        """Create all necessary database indexes on startup."""
        indexes = [
            ("orders", [("id", 1)], {"unique": True}),
            ("orders", [("order_number", 1)], {}),
            ("orders", [("phone_numbers", 1)], {}),
            ("orders", [("fulfillment_status_internal", 1), ("created_at", 1)], {}),
            ("orders", [("updated_at", -1)], {}),  # For performance metrics
            ("orders", [("business_id", 1), ("fulfillment_status_internal", 1), ("fulfilled_at", 1), ("packed_by", 1)], {}),  # For KPI cards and leaderboard queries
            ("customers", [("phone_number", 1)], {"unique": True}),
            ("customers", [("last_interaction", -1)], {}),
            ("security_events", [("timestamp", -1)], {}),
            ("rules", [("name", 1)], {"unique": True}),
            ("strings", [("key", 1)], {"unique": True}),
            ("message_logs", [("wamid", 1)], {"unique": True}),
            ("message_logs", [("metadata.broadcast_id", 1), ("status", 1)], {}),
            ("abandoned_checkouts", [("updated_at", 1), ("reminder_sent", 1), ("completed_at", 1)], {}),
            ("triage_tickets", [("status", 1), ("_id", 1)], {}),
            ("triage_tickets", [("business_id", 1), ("status", 1), ("created_at", 1)], {}),
            ("human_escalation_analytics", [("timestamp", -1)], {}),
        ]

        for collection, keys, options in indexes:
            try:
                await self.db[collection].create_index(keys, **options)
                logger.debug(f"Created index on {collection}: {keys}")
            except Exception as e:
                logger.error(f"Failed to create index on {collection} {keys}: {e}")

        logger.info("Database indexes created successfully.")

    async def health_check(self) -> bool:
        """
        Check MongoDB connection health.
        
        Returns:
            True if connection is healthy
        """
        try:
            await self.client.admin.command('ping')
            return True
        except Exception as e:
            logger.error(f"MongoDB health check failed: {e}")
            return False

    # ==================== Customer Operations ====================

    async def get_customer(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve customer by phone number.
        
        Args:
            phone_number: Customer's phone number
            
        Returns:
            Customer document or None
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return None
        
        database_operations_counter.labels(operation="get_customer", status="attempted").inc()
        customer = await self._safe_db_operation(
            lambda: self.db.customers.find_one({"phone_number": cleaned_phone})
        )
        
        if customer:
            database_operations_counter.labels(operation="get_customer", status="success").inc()
        return customer
    
    async def toggle_opt_out(self, phone: str, status: bool) -> bool:
        """
        Toggle opt-out status for a customer.
        
        Args:
            phone: Customer's phone number
            status: True to opt-out, False to opt-in
            
        Returns:
            True if update was successful, False otherwise
        """
        cleaned_phone = self._sanitize_phone(phone)
        if not cleaned_phone:
            logger.warning(f"Invalid phone number for opt-out toggle: {phone}")
            return False
        
        try:
            result = await self.db.customers.update_one(
                {"phone_number": cleaned_phone},
                {"$set": {"opted_out": status, "opt_out_updated_at": self._now_utc()}},
                upsert=False
            )
            if result.modified_count > 0:
                logger.info(f"Opt-out status updated for {cleaned_phone[:4]}...: opted_out={status}")
                return True
            else:
                logger.warning(f"Customer not found for opt-out toggle: {cleaned_phone[:4]}...")
                return False
        except Exception:
            logger.exception(f"Failed to toggle opt-out for {cleaned_phone[:4]}...")
            return False

    async def create_customer(self, customer_data: Dict[str, Any]) -> Optional[str]:
        """
        Create a new customer record.
        
        Args:
            customer_data: Customer information
            
        Returns:
            Inserted customer ID or None on failure
        """
        # Sanitize phone number in customer data
        if "phone_number" in customer_data:
            customer_data["phone_number"] = self._sanitize_phone(customer_data["phone_number"])
            if not customer_data["phone_number"]:
                logger.warning("Attempted to create customer with invalid phone number")
                return None

        customer_data.setdefault("created_at", self._now_utc())
        customer_data.setdefault("conversation_history", [])
        
        try:
            result = await self._safe_db_operation(
                lambda: self.db.customers.insert_one(customer_data)
            )
            if result:
                database_operations_counter.labels(operation="create_customer", status="success").inc()
                return str(result.inserted_id)
        except Exception as e:
            if "duplicate key" in str(e).lower():
                logger.warning(f"Duplicate customer creation attempt for phone: {customer_data.get('phone_number', 'N/A')[:4]}...")
            else:
                logger.exception("Failed to create customer")
            database_operations_counter.labels(operation="create_customer", status="failed").inc()
        
        return None

    async def takeover_conversation(self, phone_number: str, user_id: str) -> bool:
        """
        Take over a conversation for human handling.
        
        Args:
            phone_number: Customer's phone number
            user_id: ID of the user taking over the conversation
            
        Returns:
            True if successful
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return False
        
        now = self._now_utc()
        result = await self._safe_db_operation(
            lambda: self.db.customers.update_one(
                {"phone_number": cleaned_phone},
                {
                    "$set": {
                        "conversation_mode": "human",
                        "conversation_locked_by": user_id,
                        "conversation_last_mode_change_at": now
                    }
                }
            )
        )
        
        return result and result.modified_count > 0
    
    async def release_conversation(self, phone_number: str, user_id: str) -> bool:
        """
        Release a conversation back to bot mode.
        
        Args:
            phone_number: Customer's phone number
            user_id: ID of the user releasing the conversation (must match locked_by)
            
        Returns:
            True if successful
            
        Raises:
            ValueError: If locked_by does not match user_id
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return False
        
        # First, check if the conversation is locked by this user
        customer = await self.get_customer(cleaned_phone)
        if not customer:
            return False
        
        locked_by = customer.get("conversation_locked_by")
        if locked_by and locked_by != user_id:
            raise ValueError(f"Conversation is locked by a different user (locked_by={locked_by}, user_id={user_id})")
        
        result = await self._safe_db_operation(
            lambda: self.db.customers.update_one(
                {"phone_number": cleaned_phone},
                {
                    "$set": {
                        "conversation_mode": "bot",
                        "conversation_last_mode_change_at": self._now_utc()
                    },
                    "$unset": {
                        "conversation_locked_by": ""
                    }
                }
            )
        )
        
        return result and result.modified_count > 0
    
    async def enforce_auto_release(self, phone_number: str) -> bool:
        """
        Automatically release a conversation if it's been in human mode for >30 minutes.
        
        Args:
            phone_number: Customer's phone number
            
        Returns:
            True if the conversation was auto-released, False otherwise
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return False
        
        customer = await self.get_customer(cleaned_phone)
        if not customer:
            return False
        
        mode = customer.get("conversation_mode", "bot")
        if mode != "human":
            return False
        
        last_mode_change = customer.get("conversation_last_mode_change_at")
        if not last_mode_change:
            # If there's no timestamp, assume it's stale and release it
            logger.warning(f"Auto-releasing stale lock for {cleaned_phone[:4]}... (no timestamp)")
            await self._force_release_conversation(cleaned_phone)
            return True
        
        # Check if last_mode_change_at is a datetime or needs parsing
        if isinstance(last_mode_change, str):
            try:
                last_mode_change = datetime.fromisoformat(last_mode_change.replace("Z", "+00:00"))
            except Exception:
                logger.warning(f"Could not parse last_mode_change_at for {cleaned_phone[:4]}...")
                await self._force_release_conversation(cleaned_phone)
                return True
        
        # Ensure timezone awareness
        if last_mode_change.tzinfo is None:
            last_mode_change = last_mode_change.replace(tzinfo=timezone.utc)
        
        now = self._now_utc()
        time_diff = now - last_mode_change
        
        if time_diff.total_seconds() > 30 * 60:  # 30 minutes
            logger.warning(f"Auto-releasing stale lock for {cleaned_phone[:4]}... (locked for {int(time_diff.total_seconds() / 60)} minutes)")
            await self._force_release_conversation(cleaned_phone)
            return True
        
        return False
    
    async def _force_release_conversation(self, cleaned_phone: str) -> None:
        """Internal helper to force release a conversation."""
        await self._safe_db_operation(
            lambda: self.db.customers.update_one(
                {"phone_number": cleaned_phone},
                {
                    "$set": {
                        "conversation_mode": "bot",
                        "conversation_last_mode_change_at": self._now_utc()
                    },
                    "$unset": {
                        "conversation_locked_by": ""
                    }
                }
            )
        )

    async def update_conversation_history(
        self, 
        phone_number: str, 
        message: str, 
        response: str, 
        wamid: Optional[str] = None
    ) -> bool:
        """
        Add entry to customer's conversation history with size limit.
        
        Args:
            phone_number: Customer's phone number
            message: User's message
            response: Bot's response
            wamid: WhatsApp message ID
            
        Returns:
            True if successful
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return False

        entry = {
            "timestamp": self._now_utc(),
            "message": message,
            "response": response,
            "wamid": wamid,
            "status": "sent" if wamid else "internal"
        }
        
        result = await self._safe_db_operation(
            lambda: self.db.customers.update_one(
                {"phone_number": cleaned_phone},
                {
                    "$push": {
                        "conversation_history": {
                            "$each": [entry],
                            "$slice": -CONVERSATION_HISTORY_MAX  # Keep only last N entries
                        }
                    },
                    "$set": {"last_interaction": self._now_utc()}
                }
            )
        )
        
        return result and result.modified_count > 0

    async def update_customer_name(self, phone_number: str, name: str) -> bool:
        """
        Update customer's name.
        
        Args:
            phone_number: Customer's phone number
            name: New name
            
        Returns:
            True if successful
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return False

        result = await self._safe_db_operation(
            lambda: self.db.customers.update_one(
            {"phone_number": cleaned_phone},
            {"$set": {"name": name}}
            )
        )
        
        if result.modified_count > 0:
            await cache_service.redis.delete(f"customer:v2:{cleaned_phone}")
            return True
        return False

    async def get_customer_by_id(self, customer_id: str) -> Optional[Dict[str, Any]]:
        """
        Find customer by MongoDB ObjectId.
        
        Args:
            customer_id: Customer's ObjectId as string
            
        Returns:
            Customer document or None
        """
        if not self._validate_object_id(customer_id):
            logger.warning(f"Invalid customer_id format: {customer_id}")
            return None

        try:
            customer = await self.db.customers.find_one({"_id": ObjectId(customer_id)})
            return self._serialize_id(customer)
        except Exception:
            logger.exception(f"Failed to get customer by ID: {customer_id}")
            return None

    # ==================== Order Operations ====================

    async def get_recent_orders_by_phone(
        self, 
        phone_number: str, 
        limit: int = RECENT_ORDERS_LIMIT
    ) -> List[Dict[str, Any]]:
        """
        Get recent orders for a customer.
        
        Args:
            phone_number: Customer's phone number
            limit: Maximum number of orders to return
            
        Returns:
            List of order documents
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            logger.debug("Invalid phone number in get_recent_orders_by_phone")
            return []

        try:
            # Only fetch necessary fields from raw
            projection = {
                "order_number": 1,
                "created_at": 1,
                "raw.total_price": 1,
                "raw.currency": 1,
                "raw.line_items": 1,
                "raw.financial_status": 1
            }
            
            cursor = self.db.orders.find(
                {"phone_numbers": cleaned_phone},
                projection
            ).sort("created_at", -1).limit(limit)
            
            orders = await cursor.to_list(length=limit)
            database_operations_counter.labels(operation="get_recent_orders", status="success").inc()
            return orders
        except Exception:
            logger.exception("Failed to get recent orders")
            database_operations_counter.labels(operation="get_recent_orders", status="failed").inc()
            return []

    async def get_order_by_id(self, order_id: int) -> Optional[Dict[str, Any]]:
        """
        Get order by Shopify order ID.
        
        Args:
            order_id: Shopify order ID
            
        Returns:
            Order document or None
        """
        return await self.db.orders.find_one({"id": order_id})

    # ==================== Triage Tickets ====================

    async def resolve_triage_ticket(self, ticket_id: str) -> bool:
        """
        Mark triage ticket as resolved.
        
        Args:
            ticket_id: Ticket's ObjectId as string
            
        Returns:
            True if ticket was resolved
        """
        if not self._validate_object_id(ticket_id):
            logger.warning(f"Invalid ticket_id format: {ticket_id}")
            return False

        try:
            result = await self.db.triage_tickets.update_one(
                {"_id": ObjectId(ticket_id), "status": "pending"},
                {"$set": {
                    "status": "resolved",
                    "resolved_at": self._now_utc()
                }}
            )
            return result.modified_count > 0
        except Exception:
            logger.exception(f"Failed to resolve triage ticket: {ticket_id}")
            return False
    
    async def get_chat_history(self, phone_number: str, limit: int = 50) -> List[Dict[str, Any]]:
        """
        Get chat history for a phone number from message_logs collection.
        
        Args:
            phone_number: Customer's phone number
            limit: Maximum number of messages to return
            
        Returns:
            List of message documents sorted by timestamp (oldest to newest)
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return []
        
        try:
            cursor = self.db.message_logs.find(
                {"phone": cleaned_phone}
            ).sort("timestamp", -1).limit(limit)
            
            messages = await cursor.to_list(length=limit)
            # Convert ObjectId to string if present
            for msg in messages:
                if "_id" in msg:
                    msg["_id"] = str(msg["_id"])
            # Reverse to return in chronological order (oldest to newest)
            return list(reversed(messages))
        except Exception:
            logger.exception(f"Failed to get chat history for {cleaned_phone[:4]}...")
            return []
    
    async def get_last_outbound_message(self, phone_number: str) -> Optional[Dict[str, Any]]:
        """
        Get the last outbound message sent to a phone number.
        
        Args:
            phone_number: Customer's phone number
            
        Returns:
            Last outbound message document or None if not found
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            return None
        
        try:
            message = await self.db.message_logs.find_one(
                {"phone": cleaned_phone, "direction": "outbound"},
                sort=[("timestamp", -1)]
            )
            if message and "_id" in message:
                message["_id"] = str(message["_id"])
            return message
        except Exception:
            logger.exception(f"Failed to get last outbound message for {cleaned_phone[:4]}...")
            return None
    
    async def is_within_24h_window(self, phone: str) -> bool:
        """
        Checks if the customer has messaged us within the last 24 hours.
        Returns True if within window, False otherwise.
        """
        cleaned_phone = self._sanitize_phone(phone)
        if not cleaned_phone:
            return False
        
        try:
            # 1. Find the last INBOUND message from this customer
            last_msg = await self.db.message_logs.find_one(
                {"phone": cleaned_phone, "direction": "inbound"},
                sort=[("timestamp", -1)]
            )

            if not last_msg:
                return False

            last_ts = last_msg.get("timestamp")
            if not last_ts:
                return False

            # 2. Fix Timezones (The Critical Fix)
            from datetime import datetime, timezone
            
            # Get 'now' as UTC Aware
            now = datetime.now(timezone.utc)

            # Ensure 'last_ts' is UTC Aware
            # If it's a string, parse it first (just in case)
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            
            # If it's a naive datetime object, force it to UTC
            if last_ts.tzinfo is None:
                last_ts = last_ts.replace(tzinfo=timezone.utc)
            
            # 3. Calculate difference
            diff = now - last_ts
            
            # Check if less than 24 hours (86400 seconds)
            return diff.total_seconds() < 86400

        except Exception as e:
            logger.error(f"Failed to check 24h window for {cleaned_phone[:4]}...: {e}")
            # If check fails, default to FALSE (Safety first compliance)
            return False
    
    async def get_ticket_by_id(self, ticket_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a ticket by its ID.
        
        Args:
            ticket_id: Ticket's ObjectId as string
            
        Returns:
            Ticket document or None if not found
        """
        if not self._validate_object_id(ticket_id):
            logger.warning(f"Invalid ticket_id format: {ticket_id}")
            return None
        
        try:
            ticket = await self.db.triage_tickets.find_one({"_id": ObjectId(ticket_id)})
            if ticket and "_id" in ticket:
                ticket["_id"] = str(ticket["_id"])
            return ticket
        except Exception:
            logger.exception(f"Failed to get ticket by ID: {ticket_id}")
            return None
    
    async def assign_ticket(self, ticket_id: str, user_id: str) -> bool:
        """
        Assign a ticket to a user and set status to human_needed.
        
        Args:
            ticket_id: Ticket's ObjectId as string
            user_id: User ID to assign the ticket to
            
        Returns:
            True if ticket was assigned successfully
        """
        if not self._validate_object_id(ticket_id):
            logger.warning(f"Invalid ticket_id format: {ticket_id}")
            return False
        
        try:
            result = await self.db.triage_tickets.update_one(
                {"_id": ObjectId(ticket_id)},
                {"$set": {
                    "assigned_to": user_id,
                    "status": "human_needed",
                    "updated_at": self._now_utc()
                }}
            )
            return result.modified_count > 0
        except Exception:
            logger.exception(f"Failed to assign ticket {ticket_id} to user {user_id}")
            return False
    
    async def handoff_to_bot(self, ticket_id: str) -> bool:
        """
        Release a ticket back to the bot (handoff from human to bot).
        
        Args:
            ticket_id: Ticket's ObjectId as string
            
        Returns:
            True if ticket was successfully released to bot, False if status is invalid
        """
        if not self._validate_object_id(ticket_id):
            logger.warning(f"Invalid ticket_id format: {ticket_id}")
            return False
        
        try:
            # Retrieve the ticket first to check status
            ticket = await self.get_ticket_by_id(ticket_id)
            if not ticket:
                logger.warning(f"Ticket not found: {ticket_id}")
                return False
            
            # Guard clause: Only allow release if status is "human_needed"
            if ticket.get("status") != "human_needed":
                logger.warning(f"Ticket {ticket_id} status is {ticket.get('status')}, cannot release to bot")
                return False
            
            # Update operation: Set status to pending, clear assignment
            result = await self.db.triage_tickets.update_one(
                {"_id": ObjectId(ticket_id)},
                {"$set": {
                    "status": "pending",
                    "assigned_to": None,
                    "updated_at": self._now_utc()
                }}
            )
            return result.modified_count > 0
        except Exception:
            logger.exception(f"Failed to handoff ticket {ticket_id} to bot")
            return False
    
    async def log_manual_message(self, phone_number: str, text: str, user_id: str, source: str = "agent") -> None:
        """
        Log a manual message sent by a human agent.
        
        Args:
            phone_number: Customer's phone number
            text: Message text content
            user_id: User ID of the agent who sent the message
            source: Message source ("agent", "system", etc.) - defaults to "agent"
        """
        cleaned_phone = self._sanitize_phone(phone_number)
        if not cleaned_phone:
            logger.warning(f"Invalid phone number for manual message logging: {phone_number}")
            return
        
        message_data = {
            "phone": cleaned_phone,
            "text": text,
            "direction": "outbound",
            "status": "sent",
            "timestamp": self._now_utc(),
            "source": source,
            "user_id": user_id
        }
        
        await self._safe_db_operation(
            lambda: self.db.message_logs.insert_one(message_data)
        )
    
    async def get_all_agents(self) -> List[Dict[str, Any]]:
        """
        Get all available agents (admin and agent roles) for ticket assignment.
        
        Returns:
            List of agent dictionaries with id, username, and role
        """
        try:
            cursor = self.db.users.find(
                {"role": {"$in": ["admin", "agent"]}},
                {"_id": 1, "username": 1, "role": 1}
            )
            agents = await cursor.to_list(length=100)
            
            # Transform: Convert _id to string id and remove _id
            result = []
            for agent in agents:
                agent_dict = {
                    "id": str(agent["_id"]),
                    "username": agent.get("username", ""),
                    "role": agent.get("role", "")
                }
                result.append(agent_dict)
            
            return result
        except Exception:
            logger.exception("Failed to get all agents")
            return []

    # ==================== Abandoned Checkouts ====================

    async def save_abandoned_checkout(self, checkout_data: Dict[str, Any]) -> None:
        """
        Save or update abandoned checkout record.
        
        Args:
            checkout_data: Checkout information from Shopify
        """
        checkout_id = checkout_data.get("id")
        if not checkout_id:
            logger.warning("Attempted to save checkout without ID")
            return

        # Normalize time fields before saving
        normalized_data = dict(checkout_data)
        if "updated_at" in normalized_data:
            normalized_data["updated_at"] = self._parse_iso_utc(normalized_data.get("updated_at"))
        if "completed_at" in normalized_data:
            normalized_data["completed_at"] = self._parse_iso_utc(normalized_data.get("completed_at"))

        await self.db.abandoned_checkouts.update_one(
            {"id": checkout_id},
            {
                "$set": normalized_data, # <-- Use the normalized data
                "$setOnInsert": {"reminder_sent": False}
            },
            upsert=True
        )

    async def get_pending_abandoned_checkouts(self) -> List[Dict[str, Any]]:
        """
        Find checkouts ready for reminder (between 45 minutes and 3 hours old).
        
        Returns:
            List of checkout documents
        """
        now = self._now_utc()
        min_time = now - timedelta(minutes=45)
        max_time = now - timedelta(hours=3)
        
        cursor = self.db.abandoned_checkouts.find({
            "updated_at": {"$gte": max_time, "$lt": min_time},
            "reminder_sent": False,
            "completed_at": None
        })
        
        return await cursor.to_list(length=DEFAULT_QUERY_LIMIT)

    async def mark_reminder_as_sent(self, checkout_id: int) -> bool:
        """
        Mark abandoned checkout reminder as sent.
        
        Args:
            checkout_id: Checkout ID
            
        Returns:
            True if marked successfully
        """
        result = await self.db.abandoned_checkouts.update_one(
            {"id": checkout_id, "reminder_sent": False},
            {"$set": {
                "reminder_sent": True,
                "reminder_sent_at": self._now_utc()
            }}
        )
        return result.modified_count > 0

    # ==================== Security Operations ====================

    async def log_security_event(
        self, 
        event_type: str, 
        ip_address: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Log security event.
        
        Args:
            event_type: Type of security event
            ip_address: Source IP address
            details: Additional event details
        """
        event_data = {
            "event_type": event_type,
            "ip_address": ip_address,
            "timestamp": self._now_utc(),
            "details": details
        }
        await self._safe_db_operation(
            lambda: self.db.security_events.insert_one(event_data)
        )

    async def get_security_events(
        self, 
        limit: int = 100, 
        event_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get recent security events.
        
        Args:
            limit: Maximum number of events to return
            event_type: Filter by event type
            
        Returns:
            List of security event documents
        """
        query = {}
        if event_type:
            query["event_type"] = event_type
        
        cursor = self.db.security_events.find(query).sort("timestamp", -1).limit(limit)
        events = await cursor.to_list(length=limit)
        return self._serialize_ids(events)

    # ==================== Admin Dashboard Operations ====================

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get system statistics with optimized aggregation.
        
        Returns:
            Dictionary containing system metrics
        """
        now = self._now_utc()
        last_24_hours = now - timedelta(hours=24)
        last_7_days = now - timedelta(days=7)

        pipeline = [
            {"$facet": {
                "customer_stats": [
                    {"$group": {
                        "_id": None,
                        "total_customers": {"$sum": 1},
                        "active_24h": {
                            "$sum": {
                                "$cond": [
                                    {"$gte": ["$last_interaction", last_24_hours]},
                                    1,
                                    0
                                ]
                            }
                        }
                    }}
                ],
                "conversation_volume": [
                    {"$unwind": "$conversation_history"},
                    {
                        "$match": {
                            "conversation_history.timestamp": {"$gte": last_7_days}
                        }
                    },
                    {
                        "$group": {
                            "_id": {
                                "$dateToString": {
                                    "format": "%Y-%m-%d",
                                    "date": "$conversation_history.timestamp"
                                }
                            },
                            "count": {"$sum": 1}
                        }
                    },
                    {"$sort": {"_id": 1}}
                ]
            }}
        ]
        
        escalation_count = await self.db.human_escalation_analytics.count_documents(
            {"timestamp": {"$gte": last_24_hours}}
        )
        
        main_results = await self.db.customers.aggregate(pipeline).to_list(length=1)
        data = main_results[0] if main_results else {}
        customer_stats = data.get("customer_stats", [{}])[0]
        conversation_volume = data.get("conversation_volume", [])

        # Calculate abandoned cart stats for the last 24 hours
        abandoned_stats = await self.db.abandoned_checkouts.aggregate([
            {"$match": {"updated_at": {"$gte": last_24_hours}}},
            {"$group": {
                "_id": None,
                "total": {"$sum": 1},
                "recovered": {"$sum": {"$cond": [{"$ifNull": ["$completed_at", False]}, 1, 0]}},
                "revenue": {"$sum": {"$cond": [{"$ifNull": ["$completed_at", False]}, {"$toDouble": "$total_price"}, 0]}}
            }}
        ]).to_list(length=1)
        
        abandoned_data = abandoned_stats[0] if abandoned_stats else {"total": 0, "recovered": 0, "revenue": 0}

        return {
            "customers": {
                "total": customer_stats.get("total_customers", 0),
                "active_24h": customer_stats.get("active_24h", 0)
            },
            "escalations": {"count": escalation_count},
            "messages": {"avg_response_time_minutes": "N/A"},
            "conversation_volume": conversation_volume,
            "abandoned_carts": {
                "total": abandoned_data.get("total", 0),
                "recovered": abandoned_data.get("recovered", 0),
                "revenue": abandoned_data.get("revenue", 0)
            }
        }

    async def get_paginated_customers(
        self, 
        page: int = 1, 
        limit: int = PAGINATION_DEFAULT_LIMIT
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Get paginated customer list.
        
        Args:
            page: Page number (1-indexed)
            limit: Items per page
            
        Returns:
            Tuple of (customers list, pagination info)
        """
        skip = (page - 1) * limit
        
        # Use $facet to get both results and count in one query
        pipeline = [
            {
                "$facet": {
                    "customers": [
                        {"$sort": {"last_interaction": -1}},
                        {"$skip": skip},
                        {"$limit": limit},
                        {"$project": {"conversation_history": 0}}
                    ],
                    "total": [{"$count": "count"}]
                }
            }
        ]
        
        result = await self.db.customers.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return [], {"page": page, "limit": limit, "total": 0, "pages": 0}
        
        customers = result[0].get("customers", [])
        total_list = result[0].get("total", [])
        total_count = total_list[0].get("count", 0) if total_list else 0
        
        customers = self._serialize_ids(customers)
        pagination = {
            "page": page,
            "limit": limit,
            "total": total_count,
            "pages": (total_count + limit - 1) // limit if total_count > 0 else 0
        }
        
        return customers, pagination

    async def get_human_escalation_requests(self, limit: int = 5) -> List[Dict[str, Any]]:
        """
        Get recent human escalation requests from analytics collection.
        
        Args:
            limit: Maximum number of requests to return
            
        Returns:
            List of escalation request documents
        """
        try:
            cursor = self.db.human_escalation_analytics.find().sort("timestamp", -1).limit(limit)
            requests = await cursor.to_list(length=limit)
            return self._serialize_ids(requests)
        except Exception as e:
            logger.warning(f"Failed to fetch escalation requests: {e}")
            return []

    # ==================== Broadcast Operations ====================

    async def create_broadcast_group(self, group_data: BroadcastGroupCreate) -> Optional[Dict[str, Any]]:
        """
        Create a new broadcast group.

        Args:
            group_data: Broadcast group data

        Returns:
            The created broadcast group document or None on failure
        """
        sanitized_phones = [self._sanitize_phone(p) for p in group_data.phone_numbers]

        group_doc = {
            "name": group_data.name,
            "phone_numbers": [p for p in sanitized_phones if p],
            "created_at": self._now_utc()
        }

        try:
            result = await self.db.broadcast_groups.insert_one(group_doc)
            new_group = await self.db.broadcast_groups.find_one({"_id": result.inserted_id})
            return self._serialize_id(new_group)
        except Exception as e:
            logger.exception(f"Failed to create broadcast group: {e}")
            return None

    async def get_broadcast_groups(self) -> List[Dict[str, Any]]:
        """
        Get all broadcast groups.

        Returns:
            List of broadcast group documents
        """
        groups = await self.db.broadcast_groups.find({}).to_list(length=None)
        return self._serialize_ids(groups)

    async def _get_engaged_phones(self, days: int = 30) -> List[str]:
        """
        Fetch distinct phone numbers that have READ a message OR REPLIED in the last X days.
        This captures 'Hidden Readers' who have read receipts disabled but reply.
        """
        cutoff = self._now_utc() - timedelta(days=days)
        try:
            pipeline = [
                {
                    "$match": {
                        "timestamp": {"$gte": cutoff},
                        "$or": [
                            {"status": "read"},       # Explicit Read
                            {"direction": "inbound"}  # Implicit Read (Reply)
                        ]
                    }
                },
                {"$group": {"_id": "$phone"}},
            ]
            result = await self.db.message_logs.aggregate(pipeline).to_list(length=None)
            return [doc["_id"] for doc in result if doc.get("_id")]
        except Exception as e:
            logger.error(f"Failed to fetch engaged phones: {e}")
            return []

    async def count_audience(
        self,
        audience_type: str,
        target_phones: Optional[List[str]] = None,
        target_group_id: Optional[str] = None
    ) -> int:
        """
        Count customers for broadcast audience, excluding opted-out users.
        Must match get_customers_for_broadcast logic exactly.
        
        Args:
            audience_type: One of 'all', 'active', 'recent', 'inactive', 'engaged', 'custom', 'custom_group'
            target_phones: Specific phone numbers for 'custom' targeting
            target_group_id: Group ID for 'custom_group' targeting
            
        Returns:
            Count of eligible customers (excluding opted-out)
        """
        # Build query matching get_customers_for_broadcast logic
        if audience_type == "custom_group":
            if not target_group_id or not self._validate_object_id(target_group_id):
                return 0
            
            group = await self.db.broadcast_groups.find_one({"_id": ObjectId(target_group_id)})
            if not group:
                return 0
            
            group_phones = group.get("phone_numbers", [])
            sanitized_phones = [self._sanitize_phone(p) for p in group_phones]
            sanitized_phones = [p for p in sanitized_phones if p]
            
            # Count excluding opted-out users
            return await self.db.customers.count_documents({
                "phone_number": {"$in": sanitized_phones},
                "opted_out": {"$ne": True}
            })
        
        if target_phones:
            sanitized_phones = [self._sanitize_phone(p) for p in target_phones]
            sanitized_phones = [p for p in sanitized_phones if p]
            
            # Count excluding opted-out users
            return await self.db.customers.count_documents({
                "phone_number": {"$in": sanitized_phones},
                "opted_out": {"$ne": True}
            })
        
        # Build query based on audience_type
        if audience_type == "engaged":
            engaged_phones = await self._get_engaged_phones(30)
            if not engaged_phones:
                return 0
            return await self.db.customers.count_documents({
                "phone_number": {"$in": engaged_phones},
                "opted_out": {"$ne": True}
            })
        
        query = {"opted_out": {"$ne": True}}  # Always exclude opted-out
        now = self._now_utc()
        
        if audience_type == "active":
            query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
        elif audience_type == "recent":
            query["last_interaction"] = {"$gte": now - timedelta(days=7)}
        elif audience_type == "inactive":
            query["last_interaction"] = {"$lt": now - timedelta(days=30)}
        # 'all' uses query with only opted_out filter
        
        return await self.db.customers.count_documents(query)
    
    async def get_customers_for_broadcast(
        self, 
        target_type: str, 
        target_phones: Optional[List[str]] = None,
        target_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get customers for broadcast based on targeting criteria.
        
        Args:
            target_type: One of 'all', 'active', 'recent', 'inactive', 'engaged', 'custom', 'custom_group'
            target_phones: Specific phone numbers for 'custom' targeting
            
        Returns:
            List of customer documents with phone numbers
        """
        if target_type == "custom_group":
            if not target_group_id or not self._validate_object_id(target_group_id):
                return []

            group = await self.db.broadcast_groups.find_one({"_id": ObjectId(target_group_id)})
            if not group:
                return []

            # For custom groups, we need to fetch customer records to get opted_out status
            group_phones = group.get("phone_numbers", [])
            sanitized_phones = [self._sanitize_phone(p) for p in group_phones]
            sanitized_phones = [p for p in sanitized_phones if p]
            customers = await self.db.customers.find(
                {"phone_number": {"$in": sanitized_phones}},
                {"phone_number": 1, "opted_out": 1}
            ).to_list(length=None)
            # Ensure opted_out field exists (default to False if missing)
            for customer in customers:
                if "opted_out" not in customer:
                    customer["opted_out"] = False
            return customers

        if target_phones:
            sanitized_phones = [self._sanitize_phone(p) for p in target_phones]
            sanitized_phones = [p for p in sanitized_phones if p]
            customers = await self.db.customers.find(
                {"phone_number": {"$in": sanitized_phones}},
                {"phone_number": 1, "opted_out": 1}
            ).to_list(length=None)
            # Ensure opted_out field exists (default to False if missing)
            for customer in customers:
                if "opted_out" not in customer:
                    customer["opted_out"] = False
            return customers
        
        if target_type == "engaged":
            engaged_phones = await self._get_engaged_phones(30)
            if not engaged_phones:
                return []
                
            customers = await self.db.customers.find(
                {"phone_number": {"$in": engaged_phones}, "opted_out": {"$ne": True}},
                {"phone_number": 1, "first_name": 1, "opted_out": 1}
            ).to_list(length=None)
            
            # Ensure opted_out default
            for customer in customers:
                if "opted_out" not in customer:
                    customer["opted_out"] = False
            return customers
        
        query = {}
        now = self._now_utc()
        
        if target_type == "active":
            query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
        elif target_type == "recent":
            query["last_interaction"] = {"$gte": now - timedelta(days=7)}
        elif target_type == "inactive":
            query["last_interaction"] = {"$lt": now - timedelta(days=30)}
        # 'all' uses empty query
        
        customers = await self.db.customers.find(query, {"phone_number": 1, "opted_out": 1}).to_list(length=None)
        # Ensure opted_out field exists (default to False if missing)
        for customer in customers:
            if "opted_out" not in customer:
                customer["opted_out"] = False
        return customers

    async def create_broadcast_job(
        self, 
        message: str, 
        image_url: Optional[str], 
        target_type: str, 
        total_recipients: int
    ) -> str:
        """
        Create new broadcast job record.
        
        Args:
            message: Broadcast message content
            image_url: Optional image URL
            target_type: Targeting criteria
            total_recipients: Number of recipients
            
        Returns:
            Broadcast job ID
        """
        job_doc = {
            "created_at": datetime.utcnow(),
            "message": message,
            "image_url": image_url,
            "target_type": target_type,
            "status": "queued",
            "stats": {
                "total_recipients": total_recipients,
                "sent": 0,
                "delivered": 0,
                "read": 0,
                "failed": 0,
            },
            # Phase 6C: Flat stats fields for frontend compatibility
            "sent_count": 0,
            "delivered_count": 0,
            "read_count": 0,
            "failed_count": 0,
            "total_recipients": total_recipients
        }
        result = await self.db.broadcasts.insert_one(job_doc)
        return str(result.inserted_id)
    
    async def update_broadcast_job(self, job_id: str, updates: dict) -> bool:
        """
        Update a broadcast job status and stats.
        
        Args:
            job_id: Broadcast job ID (string or ObjectId)
            updates: Dictionary of fields to update (e.g., {"status": "completed", "stats.sent": 10})
            
        Returns:
            True if update was successful, False otherwise
        """
        try:
            # Convert string ID to ObjectId if needed
            if isinstance(job_id, str):
                if not self._validate_object_id(job_id):
                    logger.warning(f"Invalid job_id format: {job_id}")
                    return False
                _id = ObjectId(job_id)
            else:
                _id = job_id
            
            # Use the same collection as create_broadcast_job: 'broadcasts'
            result = await self.db.broadcasts.update_one(
                {"_id": _id},
                {"$set": updates}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to update broadcast job {job_id}: {e}")
            return False
    
    async def link_message_to_job(self, wamid: str, job_id: str):
        """
        Tag a message log with a job_id so we can track it later.
        
        Args:
            wamid: WhatsApp message ID
            job_id: Broadcast job ID
        """
        try:
            await self.db.message_logs.update_one(
                {"wamid": wamid},
                {"$set": {"metadata.job_id": job_id}}
            )
        except Exception as e:
            logger.error(f"Failed to link message {wamid} to job {job_id}: {e}")
    
    async def increment_job_stats(self, job_id: str, status: str):
        """Increment the counters for a job."""
        try:
            # Robust ID conversion
            try:
                _id = ObjectId(job_id) if isinstance(job_id, str) else job_id
            except Exception:
                _id = job_id # Fallback if it's not a valid ObjectId string
            
            field_map = {
                "sent": "sent_count",
                "delivered": "delivered_count",
                "read": "read_count",
                "failed": "failed_count"
            }
            
            if status in field_map:
                # DEBUG LOG: Remove this after it works
                logger.info(f"Incrementing {field_map[status]} for Job {_id}")
                
                # USE THE CORRECT COLLECTION NAME HERE (Check create_broadcast_job)
                # create_broadcast_job uses 'broadcasts', so we use 'broadcasts'
                result = await self.db.broadcasts.update_one( 
                    {"_id": _id},
                    {"$inc": {field_map[status]: 1}}
                )
                
                if result.modified_count == 0:
                    logger.warning(f"Failed to increment stats: Job {_id} not found in DB")
                    
        except Exception as e:
            logger.error(f"Failed to increment stats for job {job_id}: {e}")
    
    async def update_message_status(self, wamid: str, status: str):
        """
        Update message status and increment job stats.
        IMPORTANT: Uses find_one_and_update to ensure we only increment ONCE per status change.
        
        Args:
            wamid: WhatsApp message ID
            status: New status ("sent", "delivered", "read", "failed")
        """
        try:
            # 1. Try to find the message AND ensure the status is actually new.
            # This prevents double-counting if WhatsApp sends 'delivered' twice.
            updated_doc = await self.db.message_logs.find_one_and_update(
                {"wamid": wamid, "status": {"$ne": status}},  # Query: ID match AND Status is different
                {"$set": {"status": status, "updated_at": datetime.utcnow()}},
                return_document=True
            )
            
            # 2. If no document was returned, it means the status was ALREADY set. Do nothing.
            if not updated_doc:
                return

            # 3. If we updated it, check if it belongs to a job and increment stats.
            metadata = updated_doc.get("metadata", {})
            job_id = metadata.get("job_id")
            
            if job_id:
                await self.increment_job_stats(job_id, status)
                
        except Exception as e:
            logger.error(f"Failed to update status for {wamid}: {e}")

    async def get_broadcast_jobs(
        self, 
        page: int = 1, 
        limit: int = PAGINATION_DEFAULT_LIMIT
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Get paginated list of broadcast jobs.
        
        Args:
            page: Page number
            limit: Items per page
            
        Returns:
            Tuple of (jobs list, pagination info)
        """
        skip = (page - 1) * limit
        
        pipeline = [
            {
                "$facet": {
                    "jobs": [
                        {"$sort": {"created_at": -1}},
                        {"$skip": skip},
                        {"$limit": limit}
                    ],
                    "total": [{"$count": "count"}]
                }
            }
        ]
        
        result = await self.db.broadcasts.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return [], {"page": page, "limit": limit, "total": 0, "pages": 0}
        
        jobs = result[0].get("jobs", [])
        total_list = result[0].get("total", [])
        total_count = total_list[0].get("count", 0) if total_list else 0
        
        jobs = self._serialize_ids(jobs)
        pagination = {
            "page": page,
            "limit": limit,
            "total": total_count,
            "pages": (total_count + limit - 1) // limit if total_count > 0 else 0
        }
        
        return jobs, pagination

    async def get_broadcast_job_details(self, job_id: str) -> Dict[str, Any]:
        """
        Get broadcast job details with cumulative message statistics.
        
        Args:
            job_id: Broadcast job ObjectId
            
        Returns:
            Job document with updated stats
        """
        if not self._validate_object_id(job_id):
            return {}

        # Cumulative status calculation pipeline
        pipeline = [
            {"$match": {"metadata.broadcast_id": job_id}},
            {
                "$group": {
                    "_id": None,
                    "sent": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$status", ["sent", "delivered", "read", "failed"]]},
                                1,
                                0
                            ]
                        }
                    },
                    "delivered": {
                        "$sum": {
                            "$cond": [
                                {"$in": ["$status", ["delivered", "read"]]},
                                1,
                                0
                            ]
                        }
                    },
                    "read": {
                        "$sum": {"$cond": [{"$eq": ["$status", "read"]}, 1, 0]}
                    },
                    "failed": {
                        "$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}
                    }
                }
            }
        ]
        
        status_results = await self.db.message_logs.aggregate(pipeline).to_list(length=1)
        stats = status_results[0] if status_results else {}
        
        try:
            job_doc = await self.db.broadcasts.find_one({"_id": ObjectId(job_id)})
            if job_doc:
                job_doc = self._serialize_id(job_doc)
                job_doc["stats"] = {
                    "total_recipients": job_doc.get("stats", {}).get("total_recipients", 0),
                    "sent": stats.get("sent", 0),
                    "delivered": stats.get("delivered", 0),
                    "read": stats.get("read", 0),
                    "failed": stats.get("failed", 0),
                }
                return job_doc
        except Exception:
            logger.exception(f"Failed to get broadcast job details: {job_id}")
        
        return {}

    async def get_broadcast_recipients(
        self, 
        job_id: str, 
        page: int = 1, 
        limit: int = PAGINATION_DEFAULT_LIMIT,
        search: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
        """
        Get paginated broadcast recipients with optional search.
        
        Args:
            job_id: Broadcast job ID
            page: Page number
            limit: Items per page
            search: Optional search query for phone/name
            
        Returns:
            Tuple of (recipients list, pagination info)
        """
        skip = (page - 1) * limit
        match_query = {"metadata.broadcast_id": job_id}

        pipeline = [
            {"$match": match_query},
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "phone",
                    "foreignField": "phone_number",
                    "as": "customer_info"
                }
            },
            {"$unwind": {"path": "$customer_info", "preserveNullAndEmptyArrays": True}},
        ]

        # Add search filter if provided
        if search:
            search_regex = {"$regex": search, "$options": "i"}
            pipeline.append({
                "$match": {
                    "$or": [
                        {"phone": search_regex},
                        {"customer_info.name": search_regex}
                    ]
                }
            })

        # Add pagination facet
        pipeline.append({
            "$facet": {
                "recipients": [
                    {"$skip": skip},
                    {"$limit": limit}
                ],
                "total_count": [{"$count": "count"}]
            }
        })
        
        result = await self.db.message_logs.aggregate(pipeline).to_list(length=1)
        
        recipients = result[0]['recipients'] if result and result[0].get('recipients') else []
        total_count = result[0]['total_count'][0]['count'] if result and result[0].get('total_count') else 0

        pagination = {
            "page": page,
            "limit": limit,
            "total": total_count,
            "pages": (total_count + limit - 1) // limit if total_count > 0 else 0
        }
        
        return recipients, pagination

    async def get_all_broadcast_recipients_for_csv(self, job_id: str) -> List[Dict[str, Any]]:
        """
        Get all recipients for CSV export.
        
        Args:
            job_id: Broadcast job ID
            
        Returns:
            List of recipient records formatted for CSV
        """
        pipeline = [
            {"$match": {"metadata.broadcast_id": job_id}},
            {
                "$lookup": {
                    "from": "customers",
                    "localField": "phone",
                    "foreignField": "phone_number",
                    "as": "customer_info"
                }
            },
            {"$unwind": {"path": "$customer_info", "preserveNullAndEmptyArrays": True}},
            {
                "$project": {
                    "Name": "$customer_info.name",
                    "Phone Number": "$phone",
                    "Status": "$status",
                    "Timestamp": "$timestamp"
                }
            }
        ]
        return await self.db.message_logs.aggregate(pipeline).to_list(length=None)

    # ==================== Rules Engine ====================

    async def get_all_rules(self) -> List[Dict[str, Any]]:
        """
        Get all conversation rules.
        
        Returns:
            List of rule documents
        """
        rules = await self.db.rules.find({}).to_list(length=DEFAULT_QUERY_LIMIT)
        return self._serialize_ids(rules)

    async def create_rule(self, rule: Rule) -> Optional[Dict[str, Any]]:
        """
        Create new conversation rule.
        
        Args:
            rule: Rule model instance
            
        Returns:
            Created rule document or None on failure
        """
        try:
            result = await self.db.rules.insert_one(rule.dict())
            new_rule = await self.db.rules.find_one({"_id": result.inserted_id})
            return self._serialize_id(new_rule)
        except Exception as e:
            if "duplicate key" in str(e).lower():
                logger.warning(f"Duplicate rule name: {rule.name}")
            else:
                logger.exception("Failed to create rule")
            return None

    async def update_rule(self, rule_id: str, rule: Rule) -> Optional[Dict[str, Any]]:
        """
        Update existing rule.
        
        Args:
            rule_id: Rule's ObjectId
            rule: Updated rule data
            
        Returns:
            Updated rule document or None
        """
        if not self._validate_object_id(rule_id):
            return None

        try:
            result = await self.db.rules.update_one(
                {"_id": ObjectId(rule_id)},
                {"$set": rule.dict()}
            )
            if result.modified_count == 0:
                return None
            
            updated_rule = await self.db.rules.find_one({"_id": ObjectId(rule_id)})
            return self._serialize_id(updated_rule)
        except Exception:
            logger.exception(f"Failed to update rule: {rule_id}")
            return None

    # ==================== Strings Manager ====================

    async def get_all_strings(self) -> List[Dict[str, Any]]:
        """
        Get all localized strings.
        
        Returns:
            List of string resource documents
        """
        strings = await self.db.strings.find({}).to_list(length=DEFAULT_QUERY_LIMIT)
        return self._serialize_ids(strings)

    async def update_strings(self, strings: List[StringResource]) -> None:
        """
        Bulk update string resources.
        
        Args:
            strings: List of string resources to update
        """
        for s in strings:
            await self.db.strings.update_one(
                {"key": s.key},
                {"$set": {"value": s.value}},
                upsert=True
            )

    # ==================== Packing Dashboard ====================

    async def get_global_packing_config(self) -> Dict[str, Any]:
        """
        Get shared configuration for the operations team.
        
        Returns:
            Dictionary with packers and carriers lists
        """
        # Query active packer users from the users collection
        packer_users = await self.db.users.find(
            {"role": "packer", "disabled": {"$ne": True}},
            {"username": 1}
        ).to_list(length=None)
        
        packer_names = [user.get("username", "") for user in packer_users if user.get("username")]
        
        # Fallback to default packers if none found
        if not packer_names:
            packer_names = ["Swathi", "Dharam", "Pushpa"]
        
        return {
            "packers": packer_names,
            "carriers": ["India Post", "Delhivery", "Blue Dart", "DTDC", "FedEx"]
        }

    async def create_packer_user(self, username: str) -> bool:
        """
        Create a new packer user with default password 'packer123'.
        """
        # 1. Check if username exists (case insensitive)
        existing = await self.db.users.find_one({"username": {"$regex": f"^{username}$", "$options": "i"}})
        if existing:
            # If disabled, re-enable
            if existing.get("disabled"):
                await self.db.users.update_one({"_id": existing["_id"]}, {"$set": {"disabled": False, "role": "packer"}})
                return True
            return False

        # 2. Hash default password
        from app.services import security_service
        hashed_password = security_service.SecurityService.hash_password("packer123")

        # 3. Create User
        user_doc = {
            "username": username.lower(),
            "hashed_password": hashed_password,
            "role": "packer",
            "disabled": False,
            "created_at": self._now_utc()
        }
        await self.db.users.insert_one(user_doc)
        return True

    async def remove_packer_user(self, username: str) -> bool:
        """Soft-delete a packer."""
        result = await self.db.users.update_one(
            {"username": username, "role": "packer"},
            {"$set": {"disabled": True}}
        )
        return result.modified_count > 0

    async def get_all_packing_orders(self, business_id: str) -> List[Dict[str, Any]]:
        """
        Get all orders for packing dashboard.
        
        Args:
            business_id: Business ID to filter orders
        
        Returns:
            List of formatted order documents
        """
        statuses = [status.value for status in OrderStatus]
        
        orders_cursor = self.db.orders.find(
            {"fulfillment_status_internal": {"$in": statuses}, "business_id": business_id}
        ).sort("created_at", 1)
        
        orders_list = await orders_cursor.to_list(length=200)
        
        formatted_orders = []
        for order in orders_list:
            raw_order = order.get("raw", {})
            customer = raw_order.get("customer", {})
            shipping_address = raw_order.get("shipping_address", {})
            
            line_items = [
                {
                    "quantity": item.get("quantity"),
                    "title": item.get("title"),
                    "sku": item.get("sku"),
                    "image_url": item.get("image_url", "https://placehold.co/80")
                }
                for item in order.get("line_items_with_images", [])
            ]

            customer_name = f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip()
            customer_phone = shipping_address.get("phone") or customer.get("phone")

            # Get Shopify order ID (primary id)
            shopify_id = order.get("id") or raw_order.get("id")
            
            # Get order name (e.g., "FO1067") - fallback to order_number if missing
            order_name = order.get("name") or raw_order.get("name") or str(order.get("order_number", ""))
            
            formatted_orders.append({
                "id": shopify_id,  # Primary ID field
                "order_id": str(shopify_id) if shopify_id else None,  # String alias for frontend
                "order_number": order.get("order_number"),
                "name": order_name,  # Critical for "FO1067"
                "business_id": order.get("business_id"),
                "status": order.get("fulfillment_status_internal"),
                "created_at": order.get("created_at"),
                "packer_name": order.get("packed_by"),
                "customer": {
                    "name": customer_name or "N/A",
                    "phone": customer_phone
                },
                "items": line_items,
                "notes": order.get("notes"),
                "hold_reason": order.get("hold_reason"),
                "problem_item_skus": order.get("problem_item_skus", []),
                "previously_on_hold_reason": order.get("previously_on_hold_reason"),
                "previously_problem_skus": order.get("previously_problem_skus", [])
            })
        
        return formatted_orders

    async def update_order_status(self, order_id: int, business_id: str, new_status: str) -> bool:
        """
        Update order status from Pending/Needs Stock Check to In Progress.
        
        Args:
            order_id: Shopify order ID
            business_id: Business ID for isolation
            new_status: New status value
            
        Returns:
            True if updated successfully
        """
        update_data = {
            "fulfillment_status_internal": new_status,
            "updated_at": self._now_utc()
        }
        
        if new_status == OrderStatus.IN_PROGRESS.value:
            update_data["in_progress_at"] = self._now_utc()
        
        result = await self.db.orders.update_one(
            {
                "id": order_id,
                "business_id": business_id,
                "fulfillment_status_internal": {
                    "$in": [OrderStatus.PENDING.value, OrderStatus.NEEDS_STOCK_CHECK.value]
                }
            },
            {"$set": update_data}
        )
        
        return result.modified_count > 0

    async def hold_order(
        self, 
        order_id: int, 
        business_id: str,
        reason: str, 
        notes: Optional[str] = None,
        skus: Optional[List[str]] = None
    ) -> None:
        """
        Put order on hold with reason.
        
        Args:
            order_id: Shopify order ID
            business_id: Business ID for isolation
            reason: Hold reason
            notes: Optional additional notes
            skus: Optional list of problematic SKUs
        """
        await self.db.orders.update_one(
            {"id": order_id, "business_id": business_id},
            {"$set": {
                "fulfillment_status_internal": OrderStatus.ON_HOLD.value,
                "hold_reason": reason,
                "notes": notes,
                "problem_item_skus": skus or [],
                "updated_at": self._now_utc()
            }}
        )

    async def requeue_held_order(self, order_id: int, business_id: str) -> bool:
        """
        Move order from On Hold back to Pending.
        
        Args:
            order_id: Shopify order ID
            business_id: Business ID for isolation
            
        Returns:
            True if requeued successfully
        """
        order_on_hold = await self.db.orders.find_one(
            {"id": order_id, "business_id": business_id, "fulfillment_status_internal": OrderStatus.ON_HOLD.value}
        )
        
        if not order_on_hold:
            return False

        previous_reason = order_on_hold.get("hold_reason", "Unknown reason")
        previous_skus = order_on_hold.get("problem_item_skus", [])

        await self.db.orders.update_one(
            {"id": order_id, "business_id": business_id},
            {
                "$set": {
                    "fulfillment_status_internal": OrderStatus.PENDING.value,
                    "previously_on_hold_reason": previous_reason,
                    "previously_problem_skus": previous_skus,
                    "updated_at": self._now_utc()
                },
                "$unset": {
                    "hold_reason": "",
                    "problem_item_skus": "",
                    "notes": ""
                }
            }
        )
        
        return True

    async def complete_order_fulfillment(
        self, 
        order_id: int, 
        business_id: str,
        packer_name: str, 
        fulfillment_id: int
    ) -> None:
        """
        Mark order as completed.
        
        Args:
            order_id: Shopify order ID
            business_id: Business ID for isolation
            packer_name: Name of person who packed the order
            fulfillment_id: Shopify fulfillment ID
        """
        await self.db.orders.update_one(
            {"id": order_id, "business_id": business_id},
            {"$set": {
                "fulfillment_status_internal": OrderStatus.COMPLETED.value,
                "business_id": business_id,
                "packed_by": packer_name,
                "fulfillment_id": fulfillment_id,
                "fulfilled_at": self._now_utc(),
                "updated_at": self._now_utc()
            }}
        )

    async def update_order_packing_status(
        self, 
        order_id: int, 
        business_id: str,
        new_status: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Update order packing status with additional details.
        
        Args:
            order_id: Shopify order ID
            business_id: Business ID for isolation
            new_status: New status value
            details: Additional fields to update
        """
        update_doc = {
            "fulfillment_status_internal": new_status,
            "updated_at": self._now_utc(),
            **details
        }
        
        if new_status == OrderStatus.IN_PROGRESS.value:
            update_doc["in_progress_at"] = self._now_utc()
        
        await self.db.orders.update_one(
            {"id": order_id, "business_id": business_id},
            {"$set": update_doc}
        )

    async def get_packing_dashboard_metrics(self, business_id: str) -> Dict[str, Any]:
        """Get live status counts and today's completion stats."""
        # 1. Current Queue Status (Pending, In Progress, On Hold)
        pipeline = [
            {"$match": {"business_id": business_id}},
            {"$group": {"_id": "$fulfillment_status_internal", "count": {"$sum": 1}}}
        ]
        status_counts = await self.db.orders.aggregate(pipeline).to_list(length=None)
        stats = {doc["_id"]: doc["count"] for doc in status_counts if doc.get("_id")}
        
        # 2. Completed TODAY count (Uses the new index efficiently)
        start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        completed_today = await self.db.orders.count_documents({
            "business_id": business_id,
            "fulfillment_status_internal": "Completed",
            "fulfilled_at": {"$gte": start_of_day}
        })
        
        return {
            "pending": stats.get("Pending", 0),
            "in_progress": stats.get("In Progress", 0),
            "on_hold": stats.get("On Hold", 0),
            "completed_today": completed_today
        }

    async def get_packer_performance_metrics(self, days: int = 7) -> List[Dict[str, Any]]:
        """
        Calculate per-packer throughput across ALL businesses (Global Operational View).
        """
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "fulfillment_status_internal": "Completed",
                    "fulfilled_at": {"$gte": cutoff},
                    "packed_by": {"$ne": None}
                }
            },
                        {
                            "$group": {
                    "_id": "$packed_by",
                                "total_orders": {"$sum": 1},
                    "last_active": {"$max": "$fulfilled_at"}
                }
            },
            {"$sort": {"total_orders": -1}}
        ]
        
        results = await self.db.orders.aggregate(pipeline).to_list(length=None)
        
        return [
            {
                "name": r["_id"],
                "count": r["total_orders"],
                "last_active": r["last_active"]
            }
            for r in results
        ]

    def _empty_performance_metrics(self) -> Dict[str, Any]:
        """Return empty metrics structure."""
        return {
            "kpis": {
                "total_orders": 0,
                "completed_orders": 0,
                "on_hold_orders": 0,
                "avg_time_to_pack_minutes": 0,
                "hold_rate": 0
            },
            "packer_leaderboard": [],
            "hold_analysis": {
                "by_reason": [],
                "top_problem_skus": []
            }
        }


    # ==================== Message Logging ====================

    async def log_message(self, message_data: Dict[str, Any]) -> None:
        """
        Log inbound or outbound message.
        
        Args:
            message_data: Message information to log (must include 'source' field:
                         "customer", "bot", "agent", "system", or "broadcast")
        """
        message_data.setdefault("timestamp", self._now_utc())
        # Ensure source field is present (required for message source hardening)
        if "source" not in message_data:
            logger.warning(f"Message logged without 'source' field: {message_data.get('wamid', 'unknown')}")
            message_data["source"] = "unknown"
        # Ensure metadata field is present (default to empty dict if not provided)
        message_data.setdefault("metadata", {})
        await self.db.message_logs.insert_one(message_data)

    # ==================== Webhook Processing ====================

    async def process_new_order_webhook(self, order_data: Dict[str, Any], shop_domain: str = "") -> None:
        """
        Ingest a new order from Shopify.
        Maps the Shopify Domain to our internal business_id.
        """
        try:
            # 1. Map Shop Domain to Business ID
            business_id = "feelori" # Default
            if "goldencollections" in shop_domain:
                business_id = "goldencollections"
            elif "godjewellery" in shop_domain:
                business_id = "godjewellery9"
            
            # 2. Extract Customer Info
            customer = order_data.get("customer", {})
            phone = customer.get("phone") or order_data.get("phone")
            
            # 3. Prepare the Document
            order_doc = {
                "id": order_data["id"],
                "order_number": str(order_data.get("order_number", "")),
                "name": order_data.get("name", ""),
                "business_id": business_id, # <--- CRITICAL for Dashboard
                
                # Packing Dashboard Statuses
                "fulfillment_status_internal": "Pending", 
                "fulfillment_status": order_data.get("fulfillment_status"),
                
                # Financials
                "financial_status": order_data.get("financial_status"),
                "total_price": float(order_data.get("total_price", 0)),
                
                # Items & Customer
                "items": order_data.get("line_items", []),
                "customer": {
                    "id": customer.get("id"),
                    "name": f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip(),
                    "phone": phone
                },
                
                "created_at": datetime.now(timezone.utc),
                "updated_at": datetime.now(timezone.utc)
            }
            
            # 4. Upsert (Insert if new, Update if exists)
            await self.db.orders.update_one(
                {"id": order_data["id"]},
                {"$set": order_doc},
                upsert=True
            )
            logger.info(f"Order {order_doc['name']} ingested for {business_id} from {shop_domain}")
            
            # 5. Track conversion from abandoned checkout if checkout_id exists
            checkout_id = order_data.get("checkout_id") or order_data.get("checkout_token")
            if checkout_id:
                await self.db.abandoned_checkouts.update_one(
                    {"id": checkout_id},
                    {
                        "$set": {
                            "completed_at": self._now_utc(),
                            "converted_order_id": order_data["id"]
                        }
                    }
                )
                logger.info(f"Marked abandoned checkout {checkout_id} as converted for order {order_data['id']}")
            
        except Exception as e:
            logger.error(f"Failed to process order webhook: {e}", exc_info=True)

    async def _check_inventory_availability(self, payload: Dict[str, Any]) -> bool:
        """Check if all line items have sufficient inventory."""
        for item in payload.get("line_items", []):
            variant_id = item.get("variant_id")
            if not variant_id:
                continue
            
            try:
                available_qty = await shopify_service.get_inventory_for_variant(variant_id)
                if available_qty is None or available_qty < item.get("quantity", 1):
                    return True
            except Exception:
                logger.exception(f"Failed to check inventory for variant {variant_id}")
                return True
        
        return False

    async def _send_order_confirmation(self, payload: Dict[str, Any]) -> None:
        """Send order confirmation WhatsApp message to customer."""
        customer_phone = (
            (payload.get("customer") or {}).get("phone")
            or (payload.get("shipping_address") or {}).get("phone")
            or (payload.get("billing_address") or {}).get("phone")
        )
        
        if not customer_phone:
            return

        try:
            customer_name = (payload.get("customer") or {}).get("first_name", "there")
            order_number = payload.get("order_number")
            order_url = payload.get("order_status_url")

            # Extract order token from URL
            order_token = None
            if order_url and "/orders/" in order_url and "/authenticate" in order_url:
                try:
                    order_token = order_url.split("/orders/")[1].split("/authenticate")[0]
                except IndexError:
                    logger.warning(f"Failed to parse order token from URL: {order_url}")
            
            if not order_token:
                logger.warning(f"Could not extract order token for order {order_number}")
                return

            body_params = [customer_name, order_number]
            button_param = order_token
            
            from app.services.whatsapp_service import whatsapp_service
            wamid = await whatsapp_service.send_template_message(
                to=customer_phone,
                template_name="order_confirmation_v2",
                body_params=body_params,
                button_url_param=button_param
            )
            
            if wamid:
                await self.update_conversation_history(
                    phone_number=customer_phone,
                    message="[Auto-reply: Order Confirmation]",
                    response="Order confirmation template sent.",
                    wamid=wamid
                )
                logger.info(f"Sent order confirmation for {order_number} to {customer_phone[:4]}...")
        except Exception:
            logger.exception(f"Failed to send order confirmation for order {payload.get('order_number')}")

    async def process_updated_order_webhook(self, payload: Dict[str, Any]) -> None:
        """
        Process order update webhook from Shopify.
        
        Args:
            payload: Shopify order webhook payload
        """
        order_id = payload.get("id")
        if not order_id:
            return
        
        patch = {
            "raw": payload,
            "last_synced": self._now_utc(),
            "updated_at": self._now_utc()
        }
        
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": patch}
        )

    async def process_fulfillment_webhook(self, payload: Dict[str, Any]) -> None:
        """
        Process fulfillment webhook from Shopify.
        
        Args:
            payload: Shopify fulfillment webhook payload
        """
        fulfillment = payload.get("fulfillment") or payload
        order_id = fulfillment.get("order_id")
        
        if not order_id:
            logger.warning("Received fulfillment webhook without order_id")
            return
        
        tracking_numbers = fulfillment.get("tracking_numbers", [])
        
        await self.db.orders.update_one(
            {"id": order_id},
            {
                "$addToSet": {"tracking_numbers": {"$each": tracking_numbers}},
                "$set": {
                    "fulfillment_status": fulfillment.get("status", "fulfilled"),
                    "last_fulfillment": fulfillment,
                    "updated_at": self._now_utc()
                }
            },
            upsert=True
        )

        # Send shipping update to customer
        await self._send_shipping_update(order_id, fulfillment)

    async def _send_shipping_update(self, order_id: int, fulfillment: Dict[str, Any]) -> None:
        """Send shipping update WhatsApp message to customer."""
        try:
            order_doc = await self.db.orders.find_one({"id": order_id})
            if not order_doc:
                logger.warning(f"Order {order_id} not found for shipping update")
                return

            customer_phone = (order_doc.get("raw", {}).get("customer") or {}).get("phone")
            if not customer_phone:
                return

            customer_name = (order_doc.get("raw", {}).get("customer") or {}).get("first_name", "there")
            order_number = order_doc.get("order_number")
            
            tracking_url = (fulfillment.get("tracking_urls") or [""])[0]
            if not tracking_url:
                logger.warning(f"No tracking URL for order {order_id}")
                return

            carrier_name = fulfillment.get("tracking_company", "our shipping partner")
            body_params = [customer_name, order_number, carrier_name]
            button_param = tracking_url
            
            from app.services.whatsapp_service import whatsapp_service
            wamid = await whatsapp_service.send_template_message(
                to=customer_phone,
                template_name="shipping_update_v1",
                body_params=body_params,
                button_url_param=button_param
            )

            if wamid:
                await self.update_conversation_history(
                    phone_number=customer_phone,
                    message="[Auto-reply: Shipping Update]",
                    response="Shipping update template sent.",
                    wamid=wamid
                )
                logger.info(f"Sent shipping update for order {order_id}")
        except Exception:
            logger.exception(f"Failed to send shipping update for order {order_id}")


# Globally accessible instance
db_service = DatabaseService(settings.mongo_atlas_uri)