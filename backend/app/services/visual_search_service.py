# /app/services/visual_search_service.py

import io
import pickle
import logging
import sqlite3
import httpx
import numpy as np
import torch
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
        if self.model and self.processor: return
        try:
            model_name = "openai/clip-vit-base-patch32"
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
        except Exception as e:
            logger.error(f"Failed to initialize vision model: {e}"); raise

    async def generate_image_embedding(self, image_bytes: bytes) -> Optional[np.ndarray]:
        try:
            await self._initialize_vision_model()
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                embedding = features.cpu().numpy().flatten()
                return embedding / np.linalg.norm(embedding)
        except Exception as e:
            logger.error(f"Error generating image embedding: {e}"); return None

    async def find_matching_products(self, query_image_bytes: bytes, top_k: int = 15) -> List[Dict]:
        query_embedding = await self.generate_image_embedding(query_image_bytes)
        if query_embedding is None: return []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT product_id, product_handle, product_title, tags, embedding FROM product_embeddings")
            all_products = cursor.fetchall()
            if not all_products: return []
            
            product_data = {
                'ids': [row[0] for row in all_products],
                'handles': [row[1] for row in all_products],
                'titles': [row[2] for row in all_products],
                'tags': [row[3] for row in all_products],
                'embeddings': np.array([pickle.loads(row[4]) for row in all_products])
            }

        similarities = cosine_similarity(query_embedding.reshape(1, -1), product_data['embeddings'])[0]
        top_k_indices = np.argsort(similarities)[-top_k:][::-1]
        
        return [
            {'product_id': product_data['ids'][i], 'handle': product_data['handles'][i], 'title': product_data['titles'][i], 'tags': (product_data['tags'][i] or '').split(','), 'similarity_score': float(similarities[i])}
            for i in top_k_indices
        ]

# Globally accessible instance
visual_matcher = VisualProductMatcher()