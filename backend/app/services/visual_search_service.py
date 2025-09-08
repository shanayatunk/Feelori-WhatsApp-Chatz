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
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

# We need to import the shopify_service to fetch products
from app.services.shopify_service import shopify_service

# This service handles the complex task of visual product matching. It can index
# product images into a database of embeddings and find visually similar items.

logger = logging.getLogger(__name__)

class VisualProductMatcher:
    def __init__(self, db_path: str = "product_embeddings.db"):
        self.db_path = db_path
        self.processor = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.index = {}  # In-memory cache for embeddings
        self._last_db_mod_time: float = 0.0  # Tracks when the DB file was last loaded
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
        if self.model and self.processor:
            return
        logger.info("Initializing vision model for visual search...")
        self.processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        logger.info("Vision model initialized.")

    async def _load_index_from_db(self):
        """Loads all product embeddings from the SQLite DB into the in-memory index."""
        logger.info(f"Loading visual search index from {self.db_path}...")
        try:
            # Check if the database file exists and is not empty
            if not os.path.exists(self.db_path) or os.path.getsize(self.db_path) == 0:
                logger.warning(f"Database file not found or is empty at {self.db_path}. Index will be empty.")
                self.index = {}
                self._last_db_mod_time = 0.0
                return

            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT product_handle, embedding, product_title, tags FROM product_embeddings")
                
                temp_index = {}
                for row in cursor.fetchall():
                    handle, embedding_blob, title, tags_json = row
                    temp_index[handle] = {
                        "embedding": pickle.loads(embedding_blob),
                        "title": title,
                        "tags": json.loads(tags_json) if tags_json else []
                    }
                
                self.index = temp_index
                self._last_db_mod_time = os.path.getmtime(self.db_path)
                logger.info(f"Successfully loaded {len(self.index)} product embeddings into memory.")
        except Exception as e:
            logger.error(f"Failed to load visual index from DB: {e}", exc_info=True)

    async def _check_and_reload_index(self):
        """Checks if the database file has been updated on disk and reloads it if so."""
        try:
            if not os.path.exists(self.db_path):
                return

            current_mod_time = os.path.getmtime(self.db_path)
            if current_mod_time > self._last_db_mod_time:
                logger.info("Visual search database has been updated. Reloading index...")
                await self._load_index_from_db()
        except Exception as e:
            logger.error(f"Error checking or reloading the visual index: {e}", exc_info=True)
    
    # --- THIS IS THE CORRECTED AND COMPLETE FUNCTION ---
    async def index_all_products(self):
        """Fetches all products from Shopify and (re)builds the embedding database."""
        logger.info("Starting full product re-indexing...")
        if not self.model:
            await self._initialize_vision_model()

        products = await shopify_service.get_all_products()
        if not products:
            logger.warning("No products returned from Shopify. Cannot build index.")
            return

        logger.info(f"Fetched {len(products)} products from Shopify. Starting image processing.")
        count = 0
        
        async with httpx.AsyncClient() as client, sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            # Clear the table for a fresh start
            cursor.execute("DELETE FROM product_embeddings")
            conn.commit()

            for product in products:
                if not product.image_url:
                    continue
                try:
                    response = await client.get(product.image_url, timeout=20.0)
                    response.raise_for_status()
                    image_bytes = response.content
                    
                    embedding = await self.generate_image_embedding(image_bytes)
                    if embedding is None:
                        continue

                    cursor.execute(
                        "INSERT OR REPLACE INTO product_embeddings (product_id, product_handle, product_title, image_url, tags, embedding) VALUES (?, ?, ?, ?, ?, ?)",
                        (
                            str(product.id), product.handle, product.title, product.image_url,
                            json.dumps(product.tags), pickle.dumps(embedding)
                        )
                    )
                    count += 1
                    if count % 50 == 0:
                        logger.info(f"Processed {count}/{len(products)} products...")
                        conn.commit()

                except httpx.RequestError as e:
                    logger.error(f"Failed to download image for product {product.handle}: {e}")
                except Exception as e:
                    logger.error(f"An error occurred processing product {product.handle}: {e}")
            
            conn.commit()

        logger.info(f"Successfully indexed {count} products.")
        # After the database file is updated, immediately reload it into memory.
        await self._load_index_from_db()
        logger.info("Full product re-indexing complete.")


    async def generate_image_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        if not self.model:
            await self._initialize_vision_model()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            return image_features.cpu().numpy()
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {e}"); return None

    async def find_matching_products(self, query_image_bytes: bytes, top_k: int = 15) -> List[Dict]:
        """Finds the most visually similar products using the fast in-memory index."""
        if not self.model:
            await self._initialize_vision_model()

        await self._check_and_reload_index()

        if not self.index:
            logger.warning("Visual search index is empty. Cannot find matches.")
            await self._load_index_from_db()
            if not self.index:
                return []

        query_embedding = await self.generate_image_embedding(query_image_bytes)
        if query_embedding is None:
            return []
        
        handles = list(self.index.keys())
        embeddings = np.array([item['embedding'] for item in self.index.values()])

        similarities = cosine_similarity(query_embedding.reshape(1, -1), embeddings)[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for i in top_k_indices:
            handle = handles[i]
            product_info = self.index[handle]
            tags_list = product_info.get('tags', [])
                
            results.append({
                'handle': handle,
                'title': product_info.get('title'),
                'tags': tags_list,
                'similarity_score': similarities[i]
            })
        return results

# Globally accessible instance
visual_matcher = VisualProductMatcher()