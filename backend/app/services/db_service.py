# /app/services/db_service.py

import logging
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId
from datetime import datetime, timedelta
from typing import List

from app.config.settings import settings
from app.models.api import Rule, StringResource
from app.utils.circuit_breaker import CircuitBreaker, RedisCircuitBreaker
from app.utils.metrics import database_operations_counter
from app.services.cache_service import cache_service
from app.services import security_service
from app.services.shopify_service import shopify_service
from app.config import strings # Make sure this import is present

# This service manages all interactions with the MongoDB database, including
# creating indexes, CRUD operations, and complex aggregation queries for stats.

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, mongo_uri: str):
        try:
            self.client = AsyncIOMotorClient(
                mongo_uri,
                maxPoolSize=settings.max_pool_size,
                minPoolSize=settings.min_pool_size,
                tls=True,
                tlsAllowInvalidCertificates=False
            )
            self.db = self.client.get_default_database()
            self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "database")
            logger.info("MongoDB client initialized successfully.")
        except Exception as e:
            logger.error(f"Error initializing MongoDB client: {e}")
            raise

    async def create_indexes(self):
        """Create all necessary database indexes on startup."""
        try:
            await self.db.orders.create_index([("id", 1)], unique=True)
            await self.db.orders.create_index([("order_number", 1)])
            await self.db.orders.create_index([("phone_numbers", 1)])
            await self.db.customers.create_index("phone_number", unique=True)
            await self.db.security_events.create_index([("timestamp", -1)])
            await self.db.rules.create_index("name", unique=True)
            await self.db.strings.create_index("key", unique=True)
            await self.db.message_logs.create_index([("wamid", 1)], unique=True)
            logger.info("Database indexes created successfully.")
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")


    async def get_recent_orders_by_phone(self, phone_number: str, limit: int = 3) -> List[dict]:
        """
        Gets a list of the most recent orders for a customer by querying
        our own database, which has the phone_numbers array.
        """
        try:
            # Find orders where the 'phone_numbers' array contains the customer's phone
            # Fetch the 'raw' payload which _format_single_order needs
            cursor = self.db.orders.find(
                {"phone_numbers": phone_number},
                {"order_number": 1, "created_at": 1, "raw": 1} # Get the raw payload
            ).sort("created_at", -1).limit(limit)
            
            orders = await cursor.to_list(length=limit)
            return orders
        except Exception as e:
            logger.error(f"Failed to get_recent_orders_by_phone for {phone_number}: {e}")
            return []

    async def resolve_triage_ticket(self, ticket_id: str) -> bool:
        """Updates a triage ticket's status to 'resolved'."""
        try:
            result = await self.db.triage_tickets.update_one(
                {"_id": ObjectId(ticket_id), "status": "pending"},
                {"$set": {"status": "resolved"}}
            )
            # Return True if a document was actually modified
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Failed to resolve triage ticket {ticket_id}: {e}")
            return False


    # Scheduler

   
    async def save_abandoned_checkout(self, checkout_data: dict):
        """Saves or updates an abandoned checkout record."""
        checkout_id = checkout_data.get("id")
        # Use upsert=True to create a new record or update an existing one
        await self.db.abandoned_checkouts.update_one(
            {"id": checkout_id},
            {"$set": checkout_data, "$setOnInsert": {"reminder_sent": False}},
            upsert=True
        )

  
    async def get_pending_abandoned_checkouts(self) -> List[dict]:
        """Finds checkouts that are ready for a reminder."""
        one_hour_ago = datetime.utcnow() - timedelta(hours=1)
        two_hours_ago = datetime.utcnow() - timedelta(hours=2)
        
        # Find checkouts updated between 1 and 2 hours ago that haven't had a reminder sent
        cursor = self.db.abandoned_checkouts.find({
            "updated_at": {"$gte": two_hours_ago, "$lt": one_hour_ago},
            "reminder_sent": False,
            "completed_at": None  # Ensure it wasn't completed
        })
        return await cursor.to_list(length=100)


    async def mark_reminder_as_sent(self, checkout_id: int):
        """Marks an abandoned checkout reminder as sent to prevent duplicates."""
        await self.db.abandoned_checkouts.update_one(
            {"id": checkout_id},
            {"$set": {"reminder_sent": True, "reminder_sent_at": datetime.utcnow()}}
        )            
    # --- Customer Operations ---
    async def get_customer(self, phone_number: str):
        return await self.circuit_breaker.call(self.db.customers.find_one, {"phone_number": phone_number})

    async def create_customer(self, customer_data: dict):
        return await self.circuit_breaker.call(self.db.customers.insert_one, customer_data)

    async def update_conversation_history(self, phone_number: str, message: str, response: str, wamid: str | None = None):
        entry = {"timestamp": datetime.utcnow(), "message": message, "response": response, "wamid": wamid, "status": "sent" if wamid else "internal"}
        await self.circuit_breaker.call(
            self.db.customers.update_one,
            {"phone_number": phone_number},
            {"$push": {"conversation_history": entry}, "$set": {"last_interaction": datetime.utcnow()}}
        )

    async def update_customer_name(self, phone_number: str, name: str):
        await self.db.customers.update_one({"phone_number": phone_number}, {"$set": {"name": name}})
        await cache_service.redis.delete(f"customer:v2:{phone_number}")

    async def get_customer_by_id(self, customer_id: str):
        """Finds a single customer by their MongoDB ObjectId."""
        try:
            customer = await self.db.customers.find_one({"_id": ObjectId(customer_id)})
            if customer:
                customer["_id"] = str(customer["_id"])
            return customer
        except Exception:
            return None

    # --- Security Operations ---
    async def log_security_event(self, event_type: str, ip_address: str, details: dict):
        event_data = {"event_type": event_type, "ip_address": ip_address, "timestamp": datetime.utcnow(), "details": details}
        await self.circuit_breaker.call(self.db.security_events.insert_one, event_data)

    # --- Admin & Dashboard Operations ---
    async def get_system_stats(self) -> dict:
        """
        Get system statistics with an optimized aggregation pipeline that includes
        escalation counts, average response time, and daily conversation volume.
        """
        now = datetime.utcnow()
        last_24_hours = now - timedelta(hours=24)
        last_7_days = now - timedelta(days=7)

        # Main pipeline to facet all our analytics at once
        pipeline = [
            {"$facet": {
                "customer_stats": [
                    {"$group": {
                        "_id": None,
                        "total_customers": {"$sum": 1},
                        "active_24h": {"$sum": {"$cond": [{"$gte": ["$last_interaction", last_24_hours]}, 1, 0]}}
                    }}
                ],
                "conversation_volume": [
                    {"$unwind": "$conversation_history"},
                    {"$match": {"conversation_history.timestamp": {"$gte": last_7_days}}},
                    {"$group": {
                        "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$conversation_history.timestamp"}},
                        "count": {"$sum": 1}
                    }},
                    {"$sort": {"_id": 1}}
                ]
            }}
        ]
        
        # We still run this separately as it's on a different collection
        escalation_results = await self.db.human_escalation_analytics.count_documents({"timestamp": {"$gte": last_24_hours}})
        
        # Execute the main pipeline
        main_results = await self.db.customers.aggregate(pipeline).to_list(length=1)

        # Safely extract all the data
        data = main_results[0] if main_results else {}
        customer_stats = data.get("customer_stats", [{}])[0]
        conversation_volume = data.get("conversation_volume", [])

        return {
            "customers": {
                "total": customer_stats.get("total_customers", 0),
                "active_24h": customer_stats.get("active_24h", 0)
            },
            "escalations": {"count": escalation_results},
            # This can be enhanced later to calculate avg response time
            "messages": {"avg_response_time_minutes": "N/A"},
            "conversation_volume": conversation_volume
        }
    async def get_paginated_customers(self, page: int, limit: int) -> tuple[list, dict]:
        skip = (page - 1) * limit
        cursor = self.db.customers.find({}, {"conversation_history": 0}).sort("last_interaction", -1).skip(skip).limit(limit)
        customers = await cursor.to_list(length=limit)
        total_count = await self.db.customers.count_documents({})
        for customer in customers:
            customer["_id"] = str(customer["_id"])
        pagination = {"page": page, "limit": limit, "total": total_count, "pages": (total_count + limit - 1) // limit}
        return customers, pagination

    async def get_security_events(self, limit: int, event_type: str | None) -> list:
        query = {}
        if event_type: query["event_type"] = event_type
        cursor = self.db.security_events.find(query).sort("timestamp", -1).limit(limit)
        events = await cursor.to_list(length=limit)
        for event in events: event["_id"] = str(event["_id"])
        return events

    async def get_customers_for_broadcast(self, target_type: str, target_phones: list | None) -> list:
        if target_phones:
            return await self.db.customers.find({"phone_number": {"$in": target_phones}}, {"phone_number": 1}).to_list(length=None)
        
        query = {}
        now = datetime.utcnow()
        if target_type == "active":
            query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
        elif target_type == "recent": # 'recent' was not in the frontend but is supported here
            query["last_interaction"] = {"$gte": now - timedelta(days=7)}
        elif target_type == "inactive": # --- ADDED THIS LOGIC ---
            query["last_interaction"] = {"$lt": now - timedelta(days=30)}
        
        # if target_type is 'all', the query remains {} which correctly gets all customers.
        return await self.db.customers.find(query, {"phone_number": 1}).to_list(length=None)

    async def create_broadcast_job(self, message: str, image_url: str | None, target_type: str, total_recipients: int) -> str:
        """Creates a new record for a broadcast job."""
        job_doc = {
            "created_at": datetime.utcnow(),
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

    async def get_broadcast_recipients(self, job_id: str, page: int, limit: int, search: str | None) -> tuple[list, dict]:
        """Gets a paginated and searchable list of recipients for a specific broadcast job."""
        skip = (page - 1) * limit
        match_query = {"metadata.broadcast_id": job_id}

        # Main pipeline to join message logs with customer names
        pipeline = [
            {"$match": match_query},
            {"$lookup": {
                "from": "customers",
                "localField": "phone",
                "foreignField": "phone_number",
                "as": "customer_info"
            }},
            {"$unwind": {"path": "$customer_info", "preserveNullAndEmptyArrays": True}},
        ]

        # Add search stage if a query is provided
        if search:
            search_regex = {"$regex": search, "$options": "i"}
            pipeline.append({"$match": {"$or": [
                {"phone": search_regex},
                {"customer_info.name": search_regex}
            ]}})

        # Facet for pagination and total count
        facet_pipeline = pipeline + [
            {"$facet": {
                "recipients": [{"$skip": skip}, {"$limit": limit}],
                "total_count": [{"$count": "count"}]
            }}
        ]
        
        result = await self.db.message_logs.aggregate(facet_pipeline).to_list(length=1)
        
        recipients = result[0]['recipients'] if result and result[0]['recipients'] else []
        total_count = result[0]['total_count'][0]['count'] if result and result[0]['total_count'] else 0

        pagination = {"page": page, "limit": limit, "total": total_count, "pages": (total_count + limit - 1) // limit}
        return recipients, pagination

    async def get_all_broadcast_recipients_for_csv(self, job_id: str) -> list:
        """Gets all recipients for a broadcast job for CSV export."""
        pipeline = [
            {"$match": {"metadata.broadcast_id": job_id}},
            {"$lookup": {
                "from": "customers",
                "localField": "phone",
                "foreignField": "phone_number",
                "as": "customer_info"
            }},
            {"$unwind": {"path": "$customer_info", "preserveNullAndEmptyArrays": True}},
            {"$project": {
                "Name": "$customer_info.name",
                "Phone Number": "$phone",
                "Status": "$status",
                "Timestamp": "$timestamp"
            }}
        ]
        return await self.db.message_logs.aggregate(pipeline).to_list(length=None)

    async def get_broadcast_jobs(self, page: int = 1, limit: int = 20) -> tuple[list, dict]:
        """Gets a paginated list of all broadcast jobs."""
        skip = (page - 1) * limit
        cursor = self.db.broadcasts.find({}).sort("created_at", -1).skip(skip).limit(limit)
        jobs = await cursor.to_list(length=limit)
        total_count = await self.db.broadcasts.count_documents({})
        for job in jobs:
            job["_id"] = str(job["_id"])
        pagination = {"page": page, "limit": limit, "total": total_count, "pages": (total_count + limit - 1) // limit}
        return jobs, pagination

    async def get_broadcast_job_details(self, job_id: str) -> dict:
        """Gets details and aggregated message statuses for a single broadcast job."""
        
        # --- THIS IS THE FIX ---
        # This new pipeline correctly calculates cumulative statuses.
        pipeline = [
            {"$match": {"metadata.broadcast_id": job_id}},
            {
                "$group": {
                    "_id": None,
                    "sent": {"$sum": {"$cond": [{"$in": ["$status", ["sent", "delivered", "read", "failed"]]}, 1, 0]}},
                    "delivered": {"$sum": {"$cond": [{"$in": ["$status", ["delivered", "read"]]}, 1, 0]}},
                    "read": {"$sum": {"$cond": [{"$eq": ["$status", "read"]}, 1, 0]}},
                    "failed": {"$sum": {"$cond": [{"$eq": ["$status", "failed"]}, 1, 0]}}
                }
            }
        ]
        status_results = await self.db.message_logs.aggregate(pipeline).to_list(length=1)
        stats = status_results[0] if status_results else {}
        # --- END OF FIX ---
        
        job_doc = await self.db.broadcasts.find_one({"_id": ObjectId(job_id)})
        if job_doc:
            job_doc["_id"] = str(job_doc["_id"])
            job_doc["stats"] = {
                "total_recipients": job_doc.get("stats", {}).get("total_recipients", 0),
                "sent": stats.get("sent", 0),
                "delivered": stats.get("delivered", 0),
                "read": stats.get("read", 0),
                "failed": stats.get("failed", 0),
            }
            return job_doc
        return {}

    # --- THIS IS THE FIX ---
    async def get_human_escalation_requests(self, limit: int = 5) -> List:
        """
        Finds the most recent human escalation requests from the pre-aggregated
        analytics collection for fast dashboard performance.
        """
        try:
            # This is now a simple, fast find() query on the new collection
            cursor = self.db.human_escalation_analytics.find().sort("timestamp", -1).limit(limit)
            requests = await cursor.to_list(length=limit)
            for req in requests:
                req["_id"] = str(req["_id"])
            return requests
        except Exception:
            # If the analytics collection doesn't exist yet, return an empty list gracefully
            return []
    # --- END OF FIX ---

    # --- Rules Engine ---
    async def get_all_rules(self) -> list:
        rules = await self.db.rules.find({}).to_list(length=100)
        for rule in rules:
            rule["_id"] = str(rule["_id"])
        return rules

    async def create_rule(self, rule: Rule) -> dict:
        result = await self.db.rules.insert_one(rule.dict())
        new_rule = await self.db.rules.find_one({"_id": result.inserted_id})
        new_rule["_id"] = str(new_rule["_id"])
        return new_rule

    async def update_rule(self, rule_id: str, rule: Rule) -> dict:
        result = await self.db.rules.update_one({"_id": ObjectId(rule_id)}, {"$set": rule.dict()})
        if result.modified_count == 0:
            return None
        updated_rule = await self.db.rules.find_one({"_id": ObjectId(rule_id)})
        updated_rule["_id"] = str(updated_rule["_id"])
        return updated_rule

    # --- Strings Manager ---
    async def get_all_strings(self) -> list:
        strings = await self.db.strings.find({}).to_list(length=100)
        for s in strings:
            s["_id"] = str(s["_id"])
        return strings

    async def update_strings(self, strings: List[StringResource]):
        for s in strings:
            await self.db.strings.update_one({"key": s.key}, {"$set": {"value": s.value}}, upsert=True)
        
    # --- Packing Dashboard ---
    async def get_all_packing_orders(self) -> list:
        statuses = ["Pending", "Needs Stock Check", "In Progress", "On Hold", "Completed"]
        orders_cursor = self.db.orders.find({"fulfillment_status_internal": {"$in": statuses}}).sort("created_at", 1)
        orders_list = await orders_cursor.to_list(length=200)
        
        formatted_orders = []
        for order in orders_list:
            raw_order = order.get("raw", {})
            customer = raw_order.get("customer", {})
            shipping_address = raw_order.get("shipping_address", {})
            
            line_items = [{"quantity": item.get("quantity"), "title": item.get("title"), "sku": item.get("sku"), "image_url": item.get("image_url", "https://placehold.co/80")} for item in order.get("line_items_with_images", [])]

            formatted_orders.append({
                "order_id": raw_order.get("id"), "order_number": order.get("order_number"),
                "status": order.get("fulfillment_status_internal"), "created_at": order.get("created_at"),
                "packer_name": order.get("packed_by"),
                "customer": {"name": f"{customer.get('first_name', '')} {customer.get('last_name', '')}".strip(), "phone": shipping_address.get("phone") or customer.get("phone")},
                "items": line_items, "notes": order.get("notes"),
                "hold_reason": order.get("hold_reason"), "problem_item_skus": order.get("problem_item_skus", []),
                "previously_on_hold_reason": order.get("previously_on_hold_reason"), "previously_problem_skus": order.get("previously_problem_skus", [])
            })
        return formatted_orders

    async def get_order_by_id(self, order_id: int):
        return await self.db.orders.find_one({"id": order_id})
        
    async def update_order_status(self, order_id: int, new_status: str) -> bool:
        update_data = {"fulfillment_status_internal": new_status}
        if new_status == "In Progress":
            update_data["in_progress_at"] = datetime.utcnow()
        
        result = await self.db.orders.update_one(
            {"id": order_id, "fulfillment_status_internal": {"$in": ["Pending", "Needs Stock Check"]}},
            {"$set": update_data}
        )
        return result.modified_count > 0

    async def requeue_held_order(self, order_id: int) -> bool:
        order_on_hold = await self.db.orders.find_one(
            {"id": order_id, "fulfillment_status_internal": "On Hold"}
        )
        if not order_on_hold:
            return False

        previous_reason = order_on_hold.get("hold_reason", "Unknown reason")
        previous_skus = order_on_hold.get("problem_item_skus", [])

        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "Pending",
                "previously_on_hold_reason": previous_reason,
                "previously_problem_skus": previous_skus,
                "updated_at": datetime.utcnow()  # <-- ADD THIS LINE
            },
             "$unset": {
                 "hold_reason": "",
                 "problem_item_skus": "",
                 "notes": ""
             }}
        )
        return True
	

    async def hold_order(self, order_id: int, reason: str, notes: str | None, skus: list | None):
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "On Hold",
                "hold_reason": reason,
                "notes": notes,
                "problem_item_skus": skus or [],
                "updated_at": datetime.utcnow()  # <-- ADD THIS LINE
            }}
        )

        
    async def complete_order_fulfillment(self, order_id: int, packer_name: str, fulfillment_id: int):
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {
                "fulfillment_status_internal": "Completed",
                "packed_by": packer_name,
                "fulfillment_id": fulfillment_id,
                "fulfilled_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()  # <-- ADD THIS LINE
            }}
        )


    async def get_packing_dashboard_metrics(self) -> dict:
        # Simplified version for brevity, can be expanded to full pipeline
        pipeline = [{"$group": {"_id": "$fulfillment_status_internal", "count": {"$sum": 1}}}]
        results = await self.db.orders.aggregate(pipeline).to_list(None)
        status_counts = {item['_id']: item['count'] for item in results}
        return {"status_counts": status_counts}

    async def update_order_packing_status(self, order_id: int, new_status: str, details: dict):
        """Updates the status and details of an order from the packing dashboard."""
        update_doc = {
            "fulfillment_status_internal": new_status,
            "updated_at": datetime.utcnow(), # <-- ADD THIS
            **details
        }
        
        # This part ensures the 'in_progress_at' timestamp is set when needed
        if new_status == "In Progress":
            update_doc["in_progress_at"] = datetime.utcnow()
        
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": update_doc}
        )
    

# Log Messages

    async def log_message(self, message_data: dict):
        """Logs an inbound or outbound message to its own collection."""
        await self.db.message_logs.insert_one(message_data)

    # --- Webhook Processing ---
    async def process_new_order_webhook(self, payload: dict):
        order_id = payload.get("id")
        line_items_with_images = []
        for item in payload.get("line_items", []):
            image_url = await shopify_service.get_product_image_url(item.get("product_id")) if item.get("product_id") else None
            line_items_with_images.append({"title": item.get("title"), "quantity": item.get("quantity"), "sku": item.get("sku"), "image_url": image_url})

        needs_stock_check = False
        for item in payload.get("line_items", []):
            if item.get("variant_id"):
                available_qty = await shopify_service.get_inventory_for_variant(item["variant_id"])
                if available_qty is None or available_qty < item.get("quantity", 1):
                    needs_stock_check = True; break
        
        initial_status = "Needs Stock Check" if needs_stock_check else "Pending"
        
        phones = {p for p in [
            (payload.get("customer") or {}).get("phone"),
            (payload.get("shipping_address") or {}).get("phone"),
            (payload.get("billing_address") or {}).get("phone")
        ] if p}
        
        clean_phones = [security_service.EnhancedSecurityService.sanitize_phone_number(p) for p in phones if p]

        order_doc = {
            "id": order_id, "order_number": payload.get("order_number"), "created_at": payload.get("created_at"),
            "raw": payload, "line_items_with_images": line_items_with_images, "phone_numbers": clean_phones,
            "fulfillment_status_internal": initial_status, "last_synced": datetime.utcnow()
        }
        await self.db.orders.update_one({"id": order_id}, {"$set": order_doc}, upsert=True)
        
        from app.services.order_service import send_packing_alert_background
        await send_packing_alert_background(payload)
        
        customer_phone = (
            (payload.get("customer") or {}).get("phone")
            or (payload.get("shipping_address") or {}).get("phone")
            or (payload.get("billing_address") or {}).get("phone")
        )
        if customer_phone:
            try:
                customer_name = (payload.get("customer") or {}).get("first_name", "there")
                order_number = payload.get("order_number")
                order_url = payload.get("order_status_url")

                # --- THIS IS THE FIX ---
                # Extract the unique order token from the URL provided by Shopify
                order_token = None
                if order_url and "/orders/" in order_url and "/authenticate" in order_url:
                    order_token = order_url.split("/orders/")[1].split("/authenticate")[0]
                
                if not order_token:
                    logger.warning(f"Could not parse order token from order_status_url for order {order_number}")
                    return

                body_params = [customer_name, order_number]
                button_param = order_token # The button parameter is now just the token
                # --- END OF FIX ---
                
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
            except Exception as e:
                logger.error("Failed to send customer order confirmation", exc_info=True)
        
    async def process_updated_order_webhook(self, payload: dict):
        order_id = payload.get("id")
        if not order_id: return
        patch = {"raw": payload, "last_synced": datetime.utcnow()}
        await self.db.orders.update_one({"id": order_id}, {"$set": patch})
        
    async def process_fulfillment_webhook(self, payload: dict):
        fulfillment = payload.get("fulfillment") or payload
        order_id = fulfillment.get("order_id")
        if not order_id: return
        
        tracking_numbers = fulfillment.get("tracking_numbers", [])
        await self.db.orders.update_one(
            {"id": order_id},
            {"$addToSet": {"tracking_numbers": {"$each": tracking_numbers}},
             "$set": {"fulfillment_status": fulfillment.get("status", "fulfilled"), "last_fulfillment": fulfillment}},
            upsert=True
        )

        try:
            order_doc = await self.db.orders.find_one({"id": order_id})
            if not order_doc:
                logger.warning(f"Could not find order {order_id} to send shipping update.")
                return

            customer_phone = (order_doc.get("raw", {}).get("customer") or {}).get("phone")
            if not customer_phone:
                return

            customer_name = (order_doc.get("raw", {}).get("customer") or {}).get("first_name", "there")
            order_number = order_doc.get("order_number")
            
            tracking_url = (fulfillment.get("tracking_urls") or [""])[0]
            if not tracking_url:
                logger.warning(f"No tracking URL found for order {order_id}, cannot send update.")
                return
            button_param = tracking_url

            # --- THIS IS THE FIX ---
            # Extract the carrier name and add it to the body_params list
            carrier_name = fulfillment.get("tracking_company", "our shipping partner")
            body_params = [customer_name, order_number, carrier_name]
            # --- END OF FIX ---
            
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
            logger.info(f"Sent WhatsApp shipping update for order {order_id} to {customer_phone}")

        except Exception as e:
            logger.error(f"Failed to send shipping update for order {order_id}", exc_info=True)

#Packer Perfomance enhancements

    async def get_packer_performance_metrics(self, days: int = 7) -> dict:
        """
        Calculates advanced performance metrics for the packing dashboard using an aggregation pipeline.
        This version is robust and handles both old and new order data.
        """
        start_date = datetime.utcnow() - timedelta(days=days)

        pipeline = [
            {
                # --- THIS IS THE FIX ---
                # Match orders updated, fulfilled, OR created in the time window
                "$match": {
                    "$or": [
                        {"updated_at": {"$gte": start_date}},
                        {"fulfilled_at": {"$gte": start_date}},
                        {"created_at": {"$gte": start_date}}
                    ]
                }
                # --- END OF FIX ---
            },
            {
                "$facet": {
                    "kpi_metrics": [
                        {
                            "$group": {
                                "_id": None,
                                "total_orders": {"$sum": 1},
                                "completed_orders": {"$sum": {"$cond": [{"$eq": ["$fulfillment_status_internal", "Completed"]}, 1, 0]}},
                                "on_hold_orders": {"$sum": {"$cond": [{"$eq": ["$fulfillment_status_internal", "On Hold"]}, 1, 0]}},
                                "avg_time_to_pack_ms": {
                                    "$avg": {
                                        "$cond": {
                                            "if": {"$and": ["$in_progress_at", "$fulfilled_at"]},
                                            "then": {"$subtract": ["$fulfilled_at", "$in_progress_at"]},
                                            "else": None
                                        }
                                    }
                                }
                            }
                        }
                    ],
                    "packer_leaderboard": [
                        {"$match": {"fulfillment_status_internal": "Completed", "packed_by": {"$ne": None}}},
                        {"$group": {"_id": "$packed_by", "orders_packed": {"$sum": 1}}},
                        {"$sort": {"orders_packed": -1}}
                    ],
                    "hold_reasons": [
                        {"$match": {"fulfillment_status_internal": "On Hold", "hold_reason": {"$ne": None}}},
                        {"$group": {"_id": "$hold_reason", "count": {"$sum": 1}}},
                        {"$sort": {"count": -1}}
                    ],
                    "problem_skus": [
                        {"$match": {"fulfillment_status_internal": "On Hold", "problem_item_skus": {"$ne": None, "$not": {"$size": 0}}}},
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
            return {}

        # Format the results into a clean dictionary
        data = result[0]
        kpis = data["kpi_metrics"][0] if data.get("kpi_metrics") else {}
        
        # Convert milliseconds to minutes for avg_time_to_pack
        avg_time_ms = kpis.get("avg_time_to_pack_ms")
        avg_time_min = round(avg_time_ms / 60000, 2) if avg_time_ms else 0

        return {
            "kpis": {
                "total_orders": kpis.get("total_orders", 0),
                "completed_orders": kpis.get("completed_orders", 0),
                "on_hold_orders": kpis.get("on_hold_orders", 0),
                "avg_time_to_pack_minutes": avg_time_min,
                "hold_rate": round(kpis.get("on_hold_orders", 0) / kpis.get("total_orders", 1) * 100, 2)
            },
            "packer_leaderboard": data.get("packer_leaderboard", []),
            "hold_analysis": {
                "by_reason": data.get("hold_reasons", []),
                "top_problem_skus": data.get("problem_skus", [])
            }
        }

# Globally accessible instance
db_service = DatabaseService(settings.mongo_atlas_uri)