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
from PIL import Image
from transformers import AutoProcessor, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Optional

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

    async def index_all_products(self):
        """Fetches all products from Shopify and (re)builds the embedding database."""
        # This function should contain the full logic from your original build_index.py
        # For brevity, assuming it fetches products and then saves them.
        # The key is to call _load_index_from_db() at the end.
        logger.info("Starting full product re-indexing...")
        # (Your existing logic to fetch products and save embeddings to the DB goes here)
        
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
            # Ensure tags are a list, defaulting to empty if None
            tags_list = product_info.get('tags')
            if isinstance(tags_list, str):
                try:
                    tags_list = json.loads(tags_list)
                except json.JSONDecodeError:
                    tags_list = []
            elif tags_list is None:
                tags_list = []
                
            results.append({
                'handle': handle,
                'title': product_info.get('title'),
                'tags': tags_list,
                'similarity_score': similarities[i]
            })
        return results

# Globally accessible instance
visual_matcher = VisualProductMatcher()