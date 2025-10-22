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
ABANDONED_CHECKOUT_WINDOW_HOURS = (1, 2)  # (min, max) hours


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
    # ==================== Index Management ====================

    async def create_indexes(self) -> None:
        """Create all necessary database indexes on startup."""
        indexes = [
            ("orders", [("id", 1)], {"unique": True}),
            ("orders", [("order_number", 1)], {}),
            ("orders", [("phone_numbers", 1)], {}),
            ("orders", [("fulfillment_status_internal", 1), ("created_at", 1)], {}),
            ("orders", [("updated_at", -1)], {}),  # For performance metrics
            ("customers", [("phone_number", 1)], {"unique": True}),
            ("customers", [("last_interaction", -1)], {}),
            ("security_events", [("timestamp", -1)], {}),
            ("rules", [("name", 1)], {"unique": True}),
            ("strings", [("key", 1)], {"unique": True}),
            ("message_logs", [("wamid", 1)], {"unique": True}),
            ("message_logs", [("metadata.broadcast_id", 1), ("status", 1)], {}),
            ("abandoned_checkouts", [("updated_at", 1), ("reminder_sent", 1), ("completed_at", 1)], {}),
            ("triage_tickets", [("status", 1), ("_id", 1)], {}),
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
        Find checkouts ready for reminder (between 1-2 hours old).
        
        Returns:
            List of checkout documents
        """
        now = self._now_utc()
        min_hours, max_hours = ABANDONED_CHECKOUT_WINDOW_HOURS
        min_time = now - timedelta(hours=min_hours)
        max_time = now - timedelta(hours=max_hours)
        
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

        return {
            "customers": {
                "total": customer_stats.get("total_customers", 0),
                "active_24h": customer_stats.get("active_24h", 0)
            },
            "escalations": {"count": escalation_count},
            "messages": {"avg_response_time_minutes": "N/A"},
            "conversation_volume": conversation_volume
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
        total_count = result[0].get("total", [{}])[0].get("count", 0)
        
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

    async def get_customers_for_broadcast(
        self, 
        target_type: str, 
        target_phones: Optional[List[str]] = None,
        target_group_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get customers for broadcast based on targeting criteria.
        
        Args:
            target_type: One of 'all', 'active', 'recent', 'inactive', 'custom'
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

            return [{"phone_number": p} for p in group.get("phone_numbers", [])]

        if target_phones:
            sanitized_phones = [self._sanitize_phone(p) for p in target_phones]
            sanitized_phones = [p for p in sanitized_phones if p]
            return await self.db.customers.find(
                {"phone_number": {"$in": sanitized_phones}},
                {"phone_number": 1}
            ).to_list(length=None)
        
        query = {}
        now = self._now_utc()
        
        if target_type == "active":
            query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
        elif target_type == "recent":
            query["last_interaction"] = {"$gte": now - timedelta(days=7)}
        elif target_type == "inactive":
            query["last_interaction"] = {"$lt": now - timedelta(days=30)}
        # 'all' uses empty query
        
        return await self.db.customers.find(query, {"phone_number": 1}).to_list(length=None)

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
            "created_at": self._now_utc(),
            "message": message,
            "image_url": image_url,
            "target_type": target_type,
            "status": "pending",
            "stats": {
                "total_recipients": total_recipients,
                "sent": 0,
                "delivered": 0,
                "read": 0,
                "failed": 0,
            }
        }
        result = await self.db.broadcasts.insert_one(job_doc)
        return str(result.inserted_id)

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
        total_count = result[0].get("total", [{}])[0].get("count", 0)
        
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
                    # Ensure _id is projected if needed elsewhere, maybe add specific projection if not
                    # {"$project": {"_id": 1, ... other fields ...}},
                    {"$skip": skip},
                    {"$limit": limit}
                ],
                "total_count": [{"$count": "count"}]
            }
        })

        result = await self.db.message_logs.aggregate(pipeline).to_list(length=1)

        recipients = result[0]['recipients'] if result and result[0].get('recipients') else []
        total_count = result[0]['total_count'][0]['count'] if result and result[0].get('total_count') else 0

        # --- THIS IS THE FIX ---
        # Explicitly serialize the ObjectId to string for each recipient document
        recipients = self._serialize_ids(recipients)
        # --- END OF FIX ---

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

    async def get_all_packing_orders(self) -> List[Dict[str, Any]]:
        """
        Get all orders for packing dashboard.
        
        Returns:
            List of formatted order documents
        """
        statuses = [status.value for status in OrderStatus]
        
        orders_cursor = self.db.orders.find(
            {"fulfillment_status_internal": {"$in": statuses}}
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

            formatted_orders.append({
                "order_id": raw_order.get("id"),
                "order_number": order.get("order_number"),
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

    async def update_order_status(self, order_id: int, new_status: str) -> bool:
        """
        Update order status from Pending/Needs Stock Check to In Progress.
        
        Args:
            order_id: Shopify order ID
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
        reason: str, 
        notes: Optional[str] = None,
        skus: Optional[List[str]] = None
    ) -> None:
        """
        Put order on hold with reason.
        
        Args:
            order_id: Shopify order ID
            reason: Hold reason
            notes: Optional additional notes
            skus: Optional list of problematic SKUs
        """
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": OrderStatus.ON_HOLD.value,
                "hold_reason": reason,
                "notes": notes,
                "problem_item_skus": skus or [],
                "updated_at": self._now_utc()
            }}
        )

    async def requeue_held_order(self, order_id: int) -> bool:
        """
        Move order from On Hold back to Pending.
        
        Args:
            order_id: Shopify order ID
            
        Returns:
            True if requeued successfully
        """
        order_on_hold = await self.db.orders.find_one(
            {"id": order_id, "fulfillment_status_internal": OrderStatus.ON_HOLD.value}
        )
        
        if not order_on_hold:
            return False

        previous_reason = order_on_hold.get("hold_reason", "Unknown reason")
        previous_skus = order_on_hold.get("problem_item_skus", [])

        await self.db.orders.update_one(
            {"id": order_id},
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
        packer_name: str, 
        fulfillment_id: int
    ) -> None:
        """
        Mark order as completed.
        
        Args:
            order_id: Shopify order ID
            packer_name: Name of person who packed the order
            fulfillment_id: Shopify fulfillment ID
        """
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": OrderStatus.COMPLETED.value,
                "packed_by": packer_name,
                "fulfillment_id": fulfillment_id,
                "fulfilled_at": self._now_utc(),
                "updated_at": self._now_utc()
            }}
        )

    async def update_order_packing_status(
        self, 
        order_id: int, 
        new_status: str, 
        details: Dict[str, Any]
    ) -> None:
        """
        Update order packing status with additional details.
        
        Args:
            order_id: Shopify order ID
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
            {"id": order_id},
            {"$set": update_doc}
        )

    async def get_packing_dashboard_metrics(self) -> Dict[str, Any]:
        """
        Get simple packing dashboard metrics.
        
        Returns:
            Dictionary with status counts
        """
        pipeline = [
            {"$group": {"_id": "$fulfillment_status_internal", "count": {"$sum": 1}}}
        ]
        results = await self.db.orders.aggregate(pipeline).to_list(length=None)
        status_counts = {item['_id']: item['count'] for item in results if item['_id']}
        return {"status_counts": status_counts}

    async def get_packer_performance_metrics(self, days: int = 7) -> Dict[str, Any]:
        """
        Calculate advanced packer performance metrics.
        
        Args:
            days: Number of days to include in analysis
            
        Returns:
            Dictionary with KPIs, leaderboard, and hold analysis
        """
        start_date = self._now_utc() - timedelta(days=days)
        
        pipeline = [
            {
                "$match": {
                    "$or": [
                        {"updated_at": {"$gte": start_date}},
                        {"fulfilled_at": {"$gte": start_date}},
                        {"created_at": {"$gte": start_date}}
                    ]
                }
            },
            {
                "$facet": {
                    "kpi_metrics": [
                        {
                            "$group": {
                                "_id": None,
                                "total_orders": {"$sum": 1},
                                "completed_orders": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$fulfillment_status_internal", OrderStatus.COMPLETED.value]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                "on_hold_orders": {
                                    "$sum": {
                                        "$cond": [
                                            {"$eq": ["$fulfillment_status_internal", OrderStatus.ON_HOLD.value]},
                                            1,
                                            0
                                        ]
                                    }
                                },
                                "avg_time_to_pack_ms": {
                                    "$avg": {
                                        "$cond": {
                                            "if": {
                                                "$and": [
                                                    {"$ne": ["$in_progress_at", None]},
                                                    {"$ne": ["$fulfilled_at", None]},
                                                    {"$gt": ["$fulfilled_at", "$in_progress_at"]}
                                                ]
                                            },
                                            "then": {"$subtract": ["$fulfilled_at", "$in_progress_at"]},
                                            "else": None
                                        }
                                    }
                                }
                            }
                        }
                    ],
                    "packer_leaderboard": [
                        {
                            "$match": {
                                "fulfillment_status_internal": OrderStatus.COMPLETED.value,
                                "packed_by": {"$ne": None}
                            }
                        },
                        {"$group": {"_id": "$packed_by", "orders_packed": {"$sum": 1}}},
                        {"$sort": {"orders_packed": -1}},
                        {"$limit": 10}
                    ],
                    "hold_reasons": [
                        {
                            "$match": {
                                "fulfillment_status_internal": OrderStatus.ON_HOLD.value,
                                "hold_reason": {"$ne": None}
                            }
                        },
                        {"$group": {"_id": "$hold_reason", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ],
                    "problem_skus": [
                        {
                            "$match": {
                                "fulfillment_status_internal": OrderStatus.ON_HOLD.value,
                                "problem_item_skus": {"$exists": True, "$nin": [[], None]}
                            }
                        },
                        {"$unwind": "$problem_item_skus"},
                        {"$group": {"_id": "$problem_item_skus", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}},
                        {"$limit": 5}
                    ]
                }
            }
        ]
        
        result = await self.db.orders.aggregate(pipeline).to_list(length=1)
        
        if not result:
            return self._empty_performance_metrics()

        data = result[0]
        
        # --- FIXED: Properly handle empty kpi_metrics ---
        kpi_list = data.get("kpi_metrics", [])
        if kpi_list and len(kpi_list) > 0:
            kpis = kpi_list[0]
        else:
            kpis = {}
        # --- END OF FIX ---
        
        # Convert milliseconds to minutes
        avg_time_ms = kpis.get("avg_time_to_pack_ms")
        avg_time_min = round(avg_time_ms / 60000, 2) if avg_time_ms else 0
        
        total = max(kpis.get("total_orders", 1), 1)  # Prevent division by zero
        on_hold = kpis.get("on_hold_orders", 0)

        return {
            "kpis": {
                "total_orders": kpis.get("total_orders", 0),
                "completed_orders": kpis.get("completed_orders", 0),
                "on_hold_orders": on_hold,
                "avg_time_to_pack_minutes": avg_time_min,
                "hold_rate": round(on_hold / total * 100, 2)
            },
            "packer_leaderboard": data.get("packer_leaderboard", []),
            "hold_analysis": {
                "by_reason": data.get("hold_reasons", []),
                "top_problem_skus": data.get("problem_skus", [])
            }
        }

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
            message_data: Message information to log
        """
        message_data.setdefault("timestamp", self._now_utc())
        await self.db.message_logs.insert_one(message_data)

    # ==================== Webhook Processing ====================

    async def process_new_order_webhook(self, payload: Dict[str, Any]) -> None:
        """
        Process new order webhook from Shopify.
        
        Args:
            payload: Shopify order webhook payload
        """
        order_id = payload.get("id")
        if not order_id:
            logger.warning("Received order webhook without ID")
            return

        # Get product images for line items
        line_items_with_images = []
        for item in payload.get("line_items", []):
            image_url = None
            if item.get("product_id"):
                image_url = await shopify_service.get_product_image_url(item["product_id"])
            
            line_items_with_images.append({
                "title": item.get("title"),
                "quantity": item.get("quantity"),
                "sku": item.get("sku"),
                "image_url": image_url
            })

        # Check if inventory is sufficient
        needs_stock_check = await self._check_inventory_availability(payload)
        initial_status = OrderStatus.NEEDS_STOCK_CHECK.value if needs_stock_check else OrderStatus.PENDING.value
        
        # Extract and sanitize phone numbers
        clean_phones = self._extract_phones_from_payload(payload)

        order_doc = {
            "id": order_id,
            "order_number": payload.get("order_number"),
            "created_at": self._parse_iso_utc(payload.get("created_at")), # <-- Use the new helper
            "raw": payload,
            "line_items_with_images": line_items_with_images,
            "phone_numbers": clean_phones,
            "fulfillment_status_internal": initial_status,
            "last_synced": self._now_utc(),
            "updated_at": self._now_utc()
        }
        
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": order_doc},
            upsert=True
        )
        
        # Send packing alert
        try:
            from app.services.order_service import send_packing_alert_background
            await send_packing_alert_background(payload)
        except Exception:
            logger.exception("Failed to send packing alert")
        
        # Send customer confirmation
        await self._send_order_confirmation(payload)

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