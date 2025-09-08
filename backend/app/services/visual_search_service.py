# /app/services/visual_search_service.py

import io
import pickle
import logging
import sqlite3
import httpx
import numpy as np
import torch
import os
import json
import asyncio
import base64
import uuid
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# Import services needed for indexing and offloading
from app.services.shopify_service import shopify_service
from app.services.cache_service import cache_service

logger = logging.getLogger(__name__)

class VisualProductMatcher:
    def __init__(self, db_path: str = "product_embeddings.db"):
        self.db_path = db_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = {}  # In-memory cache for fast lookups
        self._last_db_mod_time: float = 0.0
        self._setup_database()
        logger.info(f"Visual matcher will use device: {self.device}")

    def _setup_database(self):
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS product_embeddings (
                    product_id TEXT PRIMARY KEY, product_handle TEXT NOT NULL,
                    product_title TEXT, image_url TEXT, tags TEXT, embedding BLOB NOT NULL
                )
            ''')
            conn.commit()

    async def _initialize_vision_model(self):
        """Loads the large vision model into memory. ONLY called by the ML worker."""
        if self.model and self.processor:
            return
        logger.info("Initializing vision model for visual search...")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        logger.info("Vision model initialized.")

    async def _load_index_from_db(self):
        """Loads the product embeddings from the SQLite file into the in-memory index."""
        logger.info(f"Loading visual search index from {self.db_path}...")
        try:
            if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
                logger.warning(f"Database file not found or is empty at {self.db_path}.")
                self.index = {}
                return

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT product_handle, embedding, product_title, tags FROM product_embeddings")
                
                temp_index = {
                    row[0]: {
                        "embedding": pickle.loads(row[1]),
                        "title": row[2],
                        "tags": json.loads(row[3]) if row[3] else []
                    } for row in cursor.fetchall()
                }
                
                self.index = temp_index
                self._last_db_mod_time = os.path.getmtime(self.db_path)
                logger.info(f"Successfully loaded {len(self.index)} product embeddings into memory.")
        except Exception as e:
            logger.error(f"Failed to load visual index from DB: {e}", exc_info=True)

    async def _check_and_reload_index(self):
        """Checks if the database file on disk is newer than the in-memory index."""
        try:
            if not os.path.exists(self.db_path): return
            current_mod_time = os.path.getmtime(self.db_path)
            if current_mod_time > self._last_db_mod_time:
                logger.info("Visual search database has been updated. Reloading index...")
                await self._load_index_from_db()
        except Exception as e:
            logger.error(f"Error checking or reloading visual index: {e}", exc_info=True)

    async def index_all_products(self):
        """Fetches all products and builds the embedding database. Called by the scheduled task."""
        logger.info("Starting full product re-indexing...")
        if not self.model: await self._initialize_vision_model()

        products = await shopify_service.get_all_products()
        if not products:
            logger.warning("No products returned from Shopify. Cannot build index.")
            return

        logger.info(f"Fetched {len(products)} products from Shopify. Processing images...")
        count = 0
        
        async with httpx.AsyncClient() as client, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM product_embeddings") # Start fresh
            conn.commit()

            for product in products:
                if not product.image_url: continue
                try:
                    response = await client.get(product.image_url, timeout=20.0)
                    response.raise_for_status()
                    embedding = await self.generate_image_embedding(response.content)
                    if embedding is None: continue

                    cursor.execute(
                        "INSERT OR REPLACE INTO product_embeddings VALUES (?, ?, ?, ?, ?, ?)",
                        (str(product.id), product.handle, product.title, product.image_url, json.dumps(product.tags), pickle.dumps(embedding))
                    )
                    count += 1
                    if count % 50 == 0: conn.commit()
                except Exception as e:
                    logger.error(f"Error processing product {product.handle}: {e}")
            
            conn.commit()
        logger.info(f"Successfully indexed {count} products.")
        await self._load_index_from_db()

    async def generate_image_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Generates an embedding for a single image. ONLY called by the ML worker."""
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
            return features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}"); return None

    async def find_matching_products(self, query_image_bytes: bytes, top_k: int = 15) -> List[Dict]:
        """Finds matches from the in-memory index. ONLY called by the ML worker."""
        await self._check_and_reload_index()
        if not self.index:
            logger.warning("Visual search index is empty. Cannot find matches.")
            await self._load_index_from_db()
            if not self.index: return []

        query_embedding = await self.generate_image_embedding(query_image_bytes)
        if query_embedding is None: return []
        
        handles = list(self.index.keys())
        embeddings = np.array([item['embedding'] for item in self.index.values()])

        sim = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
        top_indices = np.argsort(sim)[-top_k:][::-1]
        
        return [{
            'handle': handles[i],
            'title': self.index[handles[i]]['title'],
            'tags': self.index[handles[i]]['tags'],
            'similarity_score': sim[i]
        } for i in top_indices]

    async def find_matching_products_offloaded(self, query_image_bytes: bytes) -> List[Dict]:
        """Offloads the search task to the ML worker. Called by the WEB workers."""
        redis = cache_service.redis
        job_id = str(uuid.uuid4())
        image_b64 = base64.b64encode(query_image_bytes).decode('utf-8')
        
        await redis.lpush("visual_search:jobs", json.dumps({"job_id": job_id, "image_b64": image_b64}))
        logger.info(f"Offloaded visual search job {job_id} to ML worker.")
        
        result_key = f"visual_search:results:{job_id}"
        for _ in range(40):  # Poll for 20 seconds
            result_json = await redis.get(result_key)
            if result_json:
                await redis.delete(result_key)
                return json.loads(result_json)
            await asyncio.sleep(0.5)

        logger.error(f"Timeout waiting for result for job {job_id}.")
        return []

visual_matcher = VisualProductMatcher()