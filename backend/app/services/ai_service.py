# /app/services/ai_service.py

import json
import logging
import asyncio
import numpy as np
from google import genai
from google.genai.types import GenerateContentConfig, Tool, HttpOptions
from openai import AsyncOpenAI
from typing import Optional, Dict
from app.models.domain import Product
from rapidfuzz import process, fuzz
from app.config.settings import settings
from app.config.persona import AI_SYSTEM_PROMPT, VISUAL_SEARCH_PROMPT, QA_PROMPT_TEMPLATE
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import ai_requests_counter
from app.services import shopify_service


# This service encapsulates all interactions with external AI models like
# Google Gemini and OpenAI GPT, including text generation and visual search analysis.

logger = logging.getLogger(__name__)

class AIService:
    def __init__(self):
        if settings.gemini_api_key:
            # Configure the new google-genai client with v1 API (stable endpoints)
            http_options = HttpOptions(api_version='v1')
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)
            
            # Get available models and select the best one
            self.model_name = self._get_available_model()
            logger.info(f"Using Gemini model: {self.model_name}")
        else:
            self.gemini_client = None
            self.model_name = None
        
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None
            
        self.circuit_breaker = CircuitBreaker()
        self.openai_breaker = CircuitBreaker()

    def _get_available_model(self) -> str:
        """Check available models and return the best one for our use case."""
        try:
            # List available models
            models = self.gemini_client.models.list()
            available_models = [model.name for model in models]
            logger.info(f"Available Gemini models: {available_models}")
            
            # Priority order of models we want to use
            preferred_models = [
                'models/gemini-1.5-pro-latest',
                'models/gemini-1.5-pro',
                'models/gemini-1.5-flash-latest',
                'models/gemini-1.5-flash',
                'models/gemini-2.5-flash',
                'models/gemini-1.5-pro-001',
                'models/gemini-1.5-flash-001'
            ]
            
            # Find the first available model from our preferred list
            for model in preferred_models:
                if model in available_models:
                    logger.info(f"Selected model: {model}")
                    return model
            
            # If none of our preferred models are available, use the first available one
            if available_models:
                fallback_model = available_models[0]
                logger.warning(f"Using fallback model: {fallback_model}")
                return fallback_model
                
            raise Exception("No models available")
            
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            # Fallback to a common model name
            return 'models/gemini-1.5-flash'

    async def generate_response(self, message: str, context: dict | None = None) -> str:
        """Generate AI response for general text-based inquiries with failover."""
        serializable_context = json.loads(json.dumps(context, default=str)) if context else {}

        if self.gemini_client:
            try:
                response = await self._generate_gemini_response(message, serializable_context)
                if response:
                    ai_requests_counter.labels(model="gemini", status="success").inc()
                    return response
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                ai_requests_counter.labels(model="gemini", status="error").inc()

        if self.openai_client:
            try:
                response = await self.openai_breaker.call(self._generate_openai_response, message, serializable_context)
                if response:
                    ai_requests_counter.labels(model="openai", status="success").inc()
                    return response
            except Exception as e:
                logger.error(f"OpenAI fallback failed: {e}")
                ai_requests_counter.labels(model="openai", status="error").inc()
        
        return "I'm sorry, I'm having trouble connecting. Could you rephrase your question?"

    async def _generate_openai_json_response(self, prompt: str) -> Optional[Dict]:
        """Generates a JSON response from OpenAI using its JSON mode."""
        if not self.openai_client:
            return None
            
        system_message = "You are a helpful assistant designed to output JSON."
        
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"}, # This enables OpenAI's JSON mode
                messages=[
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": prompt}
                ]
            )
            ai_requests_counter.labels(model="openai-json", status="success").inc()
            return json.loads(response.choices[0].message.content)
        except Exception as e:
            logger.error(f"OpenAI JSON response generation failed: {e}")
            ai_requests_counter.labels(model="openai-json", status="error").inc()
            return None # Return None on failure

    async def _generate_openai_response(self, message: str, context: dict) -> str:
        messages = [{"role": "system", "content": AI_SYSTEM_PROMPT}]
        if context.get("conversation_history"):
            for exchange in context["conversation_history"]:
                messages.append({"role": "user", "content": exchange.get("message", "")})
                messages.append({"role": "assistant", "content": exchange.get("response", "")})
        messages.append({"role": "user", "content": message})

        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.7
        )
        return response.choices[0].message.content.strip()

    async def _generate_gemini_response(self, message: str, context: dict) -> str:
        """Generate response using the new google-genai client."""
        if not self.gemini_client:
            raise Exception("Gemini client not available")
            
        full_prompt = f"{AI_SYSTEM_PROMPT}\n\nContext: {json.dumps(context)}\n\nMessage: {message}"
        
        # Use the new API format
        response = await asyncio.to_thread(
            self.gemini_client.models.generate_content,
            model=self.model_name,
            contents=full_prompt
        )
        
        return response.text.strip()

    async def get_ai_json_response(self, prompt: str, **kwargs) -> dict:
        """
        Generates a JSON response, trying Gemini first and falling back to OpenAI.
        If both fail, it raises an exception to trigger the rule-based system.
        """
        # 1. Try Gemini First
        if self.gemini_client:
            try:
                # Use JSON mode with proper configuration
                response = await asyncio.to_thread(
                    self.gemini_client.models.generate_content,
                    model=self.model_name,
                    contents=f"{prompt}\n\nPlease respond with valid JSON only.",
                    config=GenerateContentConfig(
                        temperature=0.1,
                        response_mime_type="application/json"
                    )
                )
                
                ai_requests_counter.labels(model="gemini-json", status="success").inc()
                return json.loads(response.text)
            except Exception as e:
                logger.error(f"Gemini JSON response generation failed: {e}. Trying OpenAI fallback.")
                ai_requests_counter.labels(model="gemini-json", status="error").inc()

        # 2. Fallback to OpenAI if Gemini failed
        if self.openai_client:
            try:
                openai_response = await self._generate_openai_json_response(prompt)
                if openai_response:
                    return openai_response
            except Exception as e:
                 logger.error(f"OpenAI JSON fallback also failed: {e}")

        # 3. If both AI services fail, raise an exception to trigger the rule-based fallback.
        # If both services fail, raise an exception to trigger the rule-based fallback.
        raise Exception("Both Gemini and OpenAI failed to generate a JSON response.")

    async def get_product_qa(self, query: str, product: Optional[Product] = None) -> str:
        """
        Answers a question. If a product is provided, answers about the product.
        If no product is provided, tries to answer generally.
        """
        prompt = query
        context = None

        if product:
            # If a product *is* provided, create a specific Q&A prompt
            prompt = self.create_qa_prompt(product, query)
        else:
            # No product provided, so just pass the query to the general model
            context = {"conversation_history": []} # Give it empty context
        
        # Call the *existing* generate_response function
        return await self.generate_response(prompt, context)  
  
    def create_qa_prompt(self, product, user_question: str) -> str:
        """Creates a formatted prompt for answering a question about a specific product."""
        return QA_PROMPT_TEMPLATE.format(
            product_title=product.title,
            product_description=product.description or 'No description available',
            product_tags=', '.join(product.tags) if product.tags else 'No tags available',
            product_price=getattr(product, 'price', 'Contact for pricing'),
            user_question=user_question
        )

    # --- Hybrid Visual Search ---
    async def get_keywords_from_image_for_reranking(self, image_bytes: bytes, mime_type: str) -> list[str]:
        """Generates text keywords from an image for re-ranking visual search results."""
        if not self.gemini_client: 
            return []
            
        try:
            # Create image part for the new API
            image_data = {
                'mime_type': mime_type,
                'data': image_bytes
            }
            
            # Use multimodal content with the new client
            contents = [
                VISUAL_SEARCH_PROMPT,
                image_data
            ]
            
            response = await asyncio.to_thread(
                self.gemini_client.models.generate_content,
                model=self.model_name,
                contents=contents
            )
            
            text = (response.text or "").strip().lower()
            return [k.strip() for k in text.split(',') if k.strip()]
            
        except Exception as e:
            logger.error(f"Error getting keywords for reranking: {e}")
            return []

    def _calculate_keyword_relevance(self, keywords: list[str], candidate: dict) -> float:
        """Calculates a relevance score based on keyword matches in title and tags."""
        if not keywords: return 0.0
        relevance_score, matched_keywords = 0.0, set()
        title_lower = candidate['title'].lower()
        tags_lower = [tag.lower() for tag in candidate.get('tags', [])]
        primary_category = keywords[0]

        if primary_category in title_lower:
            relevance_score += 1.5
            matched_keywords.add(primary_category)
        elif primary_category in tags_lower:
            relevance_score += 1.0
            matched_keywords.add(primary_category)

        for keyword in keywords[1:]:
            if keyword in title_lower: relevance_score += 0.5; matched_keywords.add(keyword)
            elif keyword in tags_lower: relevance_score += 0.3; matched_keywords.add(keyword)
        
        if len(matched_keywords) > 1: relevance_score += 0.5 * (len(matched_keywords) - 1)
        return relevance_score

    async def find_exact_product_by_image(self, image_bytes: bytes, mime_type: str) -> dict:
        """Orchestrates a hybrid visual search with re-ranking."""
        # ADD THIS CHECK AT THE TOP
        if not settings.VISUAL_SEARCH_ENABLED:
            return {'success': False, 'message': 'Visual search is temporarily unavailable.'}
        from app.services.visual_search_service import visual_matcher
        try:
            visual_candidates_task = visual_matcher.find_matching_products_offloaded(image_bytes)
            keyword_task = self.get_keywords_from_image_for_reranking(image_bytes, mime_type)
            visual_candidates, keywords = await asyncio.gather(visual_candidates_task, keyword_task)

            if not visual_candidates:
                return {'success': False, 'message': 'No products found in visual index.'}

            ranked_products = []
            for candidate in visual_candidates:
                relevance_score = self._calculate_keyword_relevance(keywords, candidate)
                final_score = (candidate['similarity_score'] * 0.4) + (relevance_score * 0.6)
                candidate.update({'final_score': final_score, 'relevance_score': relevance_score})
                ranked_products.append(candidate)

            ranked_products.sort(key=lambda x: x['final_score'], reverse=True)
            
            best_match = ranked_products[0]
            match_type = 'similar'
            if best_match['similarity_score'] >= 0.92 and best_match['relevance_score'] >= 1.0: match_type = 'exact'
            elif best_match['final_score'] >= 0.8: match_type = 'very_similar'

            final_products = []
            for match in ranked_products:
                if match['final_score'] >= 0.6:
                    product = await shopify_service.get_product_by_handle(match['handle'])
                    if product: final_products.append(product)
            
            if final_products:
                return {'success': True, 'match_type': match_type, 'products': final_products[:5]}
            return {'success': False, 'message': 'No sufficiently matching products found.'}
        except Exception as e:
            logger.error(f"Error in hybrid visual search: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

# Globally accessible instance
ai_service = AIService()