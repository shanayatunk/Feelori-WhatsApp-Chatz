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
from app.services import security_service, shopify_service

# This service manages all interactions with the MongoDB database, including
# creating indexes, CRUD operations, and complex aggregation queries for stats.

logger = logging.getLogger(__name__)

class DatabaseService:
    def __init__(self, mongo_uri: str):
        self.client = AsyncIOMotorClient(
            mongo_uri,
            maxPoolSize=settings.max_pool_size,
            minPoolSize=settings.min_pool_size,
            retryWrites=True,
            readPreference='secondaryPreferred'
        )
        self.db = self.client.get_default_database()
        self.circuit_breaker = RedisCircuitBreaker(cache_service.redis, "database")

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
            logger.info("Database indexes created successfully.")
        except Exception as e:
            logger.error(f"Failed to create database indexes: {e}")
            
    # --- Customer Operations ---
    async def get_customer(self, phone_number: str):
        return await self.circuit_breaker.call(self.db.customers.find_one, {"phone_number": phone_number})

    async def create_customer(self, customer_data: dict):
        return await self.circuit_breaker.call(self.db.customers.insert_one, customer_data)

    async def update_conversation_history(self, phone_number: str, message: str, response: str, wamid: str | None = None):
        entry = {"timestamp": datetime.utcnow(), "message": message, "response": response, "wamid": wamid, "status": "sent" if wamid else None}
        await self.circuit_breaker.call(
            self.db.customers.update_one,
            {"phone_number": phone_number},
            {"$push": {"conversation_history": entry}, "$set": {"last_interaction": datetime.utcnow()}}
        )

    async def update_customer_name(self, phone_number: str, name: str):
        await self.db.customers.update_one({"phone_number": phone_number}, {"$set": {"name": name}})
        await cache_service.redis.delete(f"customer:v2:{phone_number}")

    # --- Security Operations ---
    async def log_security_event(self, event_type: str, ip_address: str, details: dict):
        event_data = {"event_type": event_type, "ip_address": ip_address, "timestamp": datetime.utcnow(), "details": details}
        await self.circuit_breaker.call(self.db.security_events.insert_one, event_data)

    # --- Admin & Dashboard Operations ---
    async def get_system_stats(self) -> dict:
        # Complex logic for fetching stats, as in original file
        now = datetime.utcnow()
        last_24_hours = now - timedelta(hours=24)
        pipeline = [{"$facet": {
            "customer_stats": [{"$group": {"_id": None, "total_customers": {"$sum": 1}, "active_24h": {"$sum": {"$cond": [{"$gte": ["$last_interaction", last_24_hours]}, 1, 0]}}}}],
            "message_stats": [{"$unwind": "$conversation_history"}, {"$group": {"_id": None, "total_24h": {"$sum": {"$cond": [{"$gte": ["$conversation_history.timestamp", last_24_hours]}, 1, 0]}}}}]
        }}]
        results = await self.db.customers.aggregate(pipeline).to_list(1)
        customer_stats = results[0]['customer_stats'][0] if results and results[0]['customer_stats'] else {}
        message_stats = results[0]['message_stats'][0] if results and results[0]['message_stats'] else {}
        
        total_customers = customer_stats.get("total_customers", 0)
        active_24h = customer_stats.get("active_24h", 0)
        total_24h_msgs = message_stats.get("total_24h", 0)
        
        return {
            "customers": {"total": total_customers, "active_24h": active_24h},
            "messages": {"total_24h": total_24h_msgs},
            "system": {"queue_size": await cache_service.redis.xlen("webhook_messages") if cache_service.redis else 0}
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
        if target_type == "active": query["last_interaction"] = {"$gte": now - timedelta(hours=24)}
        elif target_type == "recent": query["last_interaction"] = {"$gte": now - timedelta(days=7)}
        return await self.db.customers.find(query, {"phone_number": 1}).to_list(length=None)

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
        order_on_hold = await self.db.orders.find_one({"id": order_id, "fulfillment_status_internal": "On Hold"})
        if not order_on_hold: return False
        
        previous_reason = order_on_hold.get("hold_reason", "Unknown reason")
        previous_skus = order_on_hold.get("problem_item_skus", [])
        
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {"fulfillment_status_internal": "Pending", "previously_on_hold_reason": previous_reason, "previously_problem_skus": previous_skus},
             "$unset": {"hold_reason": "", "problem_item_skus": "", "notes": ""}}
        )
        return True

    async def hold_order(self, order_id: int, reason: str, notes: str | None, skus: list | None):
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {"fulfillment_status_internal": "On Hold", "hold_reason": reason, "notes": notes, "problem_item_skus": skus or []}}
        )
        
    async def complete_order_fulfillment(self, order_id: int, packer_name: str, fulfillment_id: int):
        await self.db.orders.update_one(
            {"id": order_id},
            {"$set": {"fulfillment_status_internal": "Completed", "packed_by": packer_name, "fulfillment_id": fulfillment_id, "fulfilled_at": datetime.utcnow()}}
        )

    async def get_packing_dashboard_metrics(self) -> dict:
        # Simplified version for brevity, can be expanded to full pipeline
        pipeline = [{"$group": {"_id": "$fulfillment_status_internal", "count": {"$sum": 1}}}]
        results = await self.db.orders.aggregate(pipeline).to_list(None)
        status_counts = {item['_id']: item['count'] for item in results}
        return {"status_counts": status_counts}

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
        # Assuming send_packing_alert_background is defined in another service (e.g., order_service)
        from app.services.order_service import send_packing_alert_background
        await send_packing_alert_background(payload)
        
    async def process_updated_order_webhook(self, payload: dict):
        order_id = payload.get("id")
        if not order_id: return
        patch = {"raw": payload, "last_synced": datetime.utcnow()}
        # Add fields to patch as needed
        await self.db.orders.update_one({"id": order_id}, {"$set": patch}, upsert=True)
        
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


# Globally accessible instance
db_service = DatabaseService(settings.mongo_atlas_uri)