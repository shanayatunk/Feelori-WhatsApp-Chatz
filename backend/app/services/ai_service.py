# /app/services/ai_service.py

import json
import logging
import asyncio
import re
from typing import Optional, Dict, Callable, Any
from google import genai
from google.genai.types import GenerateContentConfig, HttpOptions
from openai import AsyncOpenAI
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from app.models.domain import Product
from app.config.settings import settings, get_business_config
from app.services.db_service import db_service
from app.utils.circuit_breaker import CircuitBreaker
from app.utils.metrics import ai_requests_counter
from app.services.shopify_service import shopify_service
from app.services.string_service import string_service


# This service encapsulates all interactions with external AI models like
# Google Gemini and OpenAI GPT, including text generation and visual search analysis.

logger = logging.getLogger(__name__)

# Small retry decorator for synchronous genai calls (used inside asyncio.to_thread)
def _sync_retry_decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
    return retry(
        reraise=True,
        wait=wait_exponential(multiplier=1, min=1, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(Exception),
    )(fn)


class AIService:
    def __init__(self):
        # Gemini client: configure v1 stable API when key present.
        if settings.gemini_api_key:
            # Set reasonable timeouts - connect: 10s, read: 60s
            http_options = HttpOptions(
                api_version="v1",
                timeout=60000  # Increase timeout to 60 seconds for model operations
            )
            # pass api_key explicitly; vertexai usage can be toggled with additional flags if needed
            self.gemini_client = genai.Client(api_key=settings.gemini_api_key, http_options=http_options)
        else:
            self.gemini_client = None

        # NOTE: lazy model detection — do NOT call models.list() here (avoids network at import)
        self.model_name: Optional[str] = None

        # OpenAI async client (set to None if no key)
        if settings.openai_api_key:
            self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
        else:
            self.openai_client = None

        self.circuit_breaker = CircuitBreaker()
        self.openai_breaker = CircuitBreaker()
        self.default_json_config = GenerateContentConfig(
            temperature=0.1,
            max_output_tokens=1024
        )

    # ---- Model selection (lazy) ----
    async def _ensure_model(self):
        """Lazily discover and set the best available Gemini model."""
        if self.model_name or not self.gemini_client:
            return

        def _get_model_sync():
            try:
                models_iter = self.gemini_client.models.list()
                available = [m.name for m in models_iter]
                logger.info(f"Available Gemini models: {available}")
            except Exception as ex:
                logger.exception("Error listing Gemini models: %s", ex)
                available = []

            preferred_models = [
                "models/gemini-2.5-flash",
                "models/gemini-1.5-pro-latest",
                "models/gemini-1.5-pro",
                "models/gemini-1.5-flash-latest",
                "models/gemini-1.5-flash",
                "models/gemini-1.5-pro-001",
                "models/gemini-1.5-flash-001"
            ]
            
            # Find the first available model from our preferred list
            for model in preferred_models:
                if model in available:
                    logger.info(f"Selected model: {model}")
                    return model
            
            # If none of our preferred models are available, use the first available one
            if available:
                fallback_model = available[0]
                logger.warning(f"Using fallback model: {fallback_model}")
                return fallback_model
                
            raise Exception("No models available")

        try:
            self.model_name = await asyncio.to_thread(_get_model_sync)
            logger.info("Gemini model resolved to: %s", self.model_name)
        except Exception as e:
            logger.error(f"Failed to get available models: {e}")
            # Fallback to a common model name
            self.model_name = 'models/gemini-1.5-flash'

    def _strip_json_fences(self, text: str) -> str:
        """Remove JSON markdown fences if present."""
        # Remove ```json ... ``` or ``` ... ``` blocks
        json_fence_pattern = r'```(?:json)?\s*\n?(.*?)\n?```'
        match = re.search(json_fence_pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    # ---- Robust extractor for different response shapes ----
    def _extract_text_from_genai_response(self, response) -> str:
        """Extract text from various Gemini response formats."""
        if not response:
            return ""
        # common simple property
        text = getattr(response, "text", None)
        if text:
            return text.strip()

        # try candidates -> content -> parts -> text
        candidates = getattr(response, "candidates", None)
        try:
            if candidates and len(candidates) > 0:
                first = candidates[0]
                # candidate.content may be a pydantic/typed object or dict
                content = getattr(first, "content", None) or (first.get("content") if isinstance(first, dict) else None)
                if content:
                    parts = getattr(content, "parts", None) or (content.get("parts") if isinstance(content, dict) else None)
                    if parts and len(parts) > 0:
                        p0 = parts[0]
                        # p0 may be an object with .text or a dict
                        return (getattr(p0, "text", None) or p0.get("text") if isinstance(p0, dict) else str(p0)).strip()
                # fallback: candidate.text
                cand_text = getattr(first, "text", None) or (first.get("text") if isinstance(first, dict) else None)
                if cand_text:
                    return cand_text.strip()
        except Exception:
            logger.debug("Could not parse candidates from genai response", exc_info=True)

        # as last resort, stringify
        try:
            return str(response).strip()
        except Exception:
            return ""

    # ---- Helper to call generate_content synchronously with retries (wrapped and called in thread) ----
    def _sync_generate_with_retry(self, model: str, contents, config: Optional[GenerateContentConfig] = None):
        """Synchronous wrapper for generate_content with retry logic."""
        @_sync_retry_decorator
        def _inner(m, c, cfg):
            return self.gemini_client.models.generate_content(model=m, contents=c, config=cfg)
        return _inner(model, contents, config)

    async def generate_response(self, message: str, context: dict | None = None, business_id: str = "feelori", flow_context: dict | None = None) -> str:
        """Generate AI response for general text-based inquiries with failover."""
        serializable_context = json.loads(json.dumps(context, default=str)) if context else {}

        if self.gemini_client:
            try:
                response = await self._generate_gemini_response(message, serializable_context, business_id, flow_context)
                if response:
                    ai_requests_counter.labels(model="gemini", status="success").inc()
                    return response
            except Exception as e:
                logger.error(f"Gemini API call failed: {e}")
                ai_requests_counter.labels(model="gemini", status="error").inc()

        if self.openai_client:
            try:
                response = await self.openai_breaker.call(self._generate_openai_response, message, serializable_context, business_id, flow_context)
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

    async def _generate_openai_response(self, message: str, context: dict, business_id: str = "feelori", flow_context: dict | None = None) -> str:
        """Generate OpenAI response with conversation context."""
        # Get system prompt from BusinessConfig
        try:
            config = await db_service.db.business_configs.find_one({"business_id": business_id})
            if config and "persona" in config:
                system_prompt = config["persona"].get("prompt", "")
            else:
                # Fallback to string_service cache
                prompt_key = "GOLDEN_SYSTEM_PROMPT" if business_id == "goldencollections" else "FEELORI_SYSTEM_PROMPT"
                system_prompt = string_service.get_string(prompt_key, default="")
        except Exception as e:
            logger.warning(f"Failed to fetch persona from BusinessConfig for {business_id}: {e}")
            prompt_key = "GOLDEN_SYSTEM_PROMPT" if business_id == "goldencollections" else "FEELORI_SYSTEM_PROMPT"
            system_prompt = string_service.get_string(prompt_key, default="")
        
        # Fetch business configuration and create facts dictionary
        config = get_business_config(business_id)
        business_facts = {
            "business_name": config.business_name,
            "support_email": config.support_email,
            "support_phone": config.support_phone,
            "website_url": config.website_url,
            "address": config.business_address,
            "shipping_policy": config.shipping_policy_url
        }
        
        # Build workflow state section if flow_context is provided
        workflow_section = ""
        if flow_context:
            workflow_section = f"""

[SYSTEM-CONTROLLED WORKFLOW STATE (READ-ONLY)]
{json.dumps(flow_context, indent=2, default=str)}

**IMPORTANT INSTRUCTIONS:**
- You may reference this workflow state to generate better, context-aware responses
- You must NOT attempt to change or update this state
- You must NOT assume transitions have occurred unless explicitly stated
- This state is managed by the system and is provided for informational purposes only
"""
        
        # Build enhanced system prompt with business facts
        enhanced_system_prompt = f"""{system_prompt}

[OFFICIAL BUSINESS FACTS - USE THESE FOR ANSWERS]
{json.dumps(business_facts, indent=2)}{workflow_section}"""
        
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        if context.get("conversation_history"):
            for exchange in context["conversation_history"]:
                messages.append({"role": "user", "content": exchange.get("message", "")})
                messages.append({"role": "assistant", "content": exchange.get("response", "")})
        
        # Include context and message in user content
        user_content = f"""[CONVERSATION CONTEXT]
{json.dumps(context)}

[USER MESSAGE]
{message}"""
        messages.append({"role": "user", "content": user_content})

        response = await self.openai_client.chat.completions.create(
            model="gpt-3.5-turbo", messages=messages, max_tokens=150, temperature=0.7
        )
        return response.choices[0].message.content.strip()

    async def _generate_gemini_response(self, message: str, context: dict, business_id: str = "feelori", flow_context: dict | None = None) -> str:
        """Generate response using the new google-genai client."""
        if not self.gemini_client:
            raise Exception("Gemini client not available")

        await self._ensure_model()
        if not self.model_name:
            raise Exception("No Gemini model available")

        # Get system prompt from BusinessConfig
        try:
            config = await db_service.db.business_configs.find_one({"business_id": business_id})
            if config and "persona" in config:
                system_prompt = config["persona"].get("prompt", "")
            else:
                # Fallback to string_service cache
                prompt_key = "GOLDEN_SYSTEM_PROMPT" if business_id == "goldencollections" else "FEELORI_SYSTEM_PROMPT"
                system_prompt = string_service.get_string(prompt_key, default="")
        except Exception as e:
            logger.warning(f"Failed to fetch persona from BusinessConfig for {business_id}: {e}")
            prompt_key = "GOLDEN_SYSTEM_PROMPT" if business_id == "goldencollections" else "FEELORI_SYSTEM_PROMPT"
            system_prompt = string_service.get_string(prompt_key, default="")
        
        # Fetch business configuration and create facts dictionary
        config = get_business_config(business_id)
        business_facts = {
            "business_name": config.business_name,
            "support_email": config.support_email,
            "support_phone": config.support_phone,
            "website_url": config.website_url,
            "address": config.business_address,
            "shipping_policy": config.shipping_policy_url
        }
        
        # Build workflow state section if flow_context is provided
        workflow_section = ""
        if flow_context:
            workflow_section = f"""

[SYSTEM-CONTROLLED WORKFLOW STATE (READ-ONLY)]
{json.dumps(flow_context, indent=2, default=str)}

**IMPORTANT INSTRUCTIONS:**
- You may reference this workflow state to generate better, context-aware responses
- You must NOT attempt to change or update this state
- You must NOT assume transitions have occurred unless explicitly stated
- This state is managed by the system and is provided for informational purposes only
"""
        
        # Build enhanced prompt with business facts
        full_prompt = f"""{system_prompt}

[OFFICIAL BUSINESS FACTS - USE THESE FOR ANSWERS]
{json.dumps(business_facts, indent=2)}{workflow_section}

[CONVERSATION CONTEXT]
{json.dumps(context)}

[USER MESSAGE]
{message}
"""

        # Call the sync client inside a thread, with simple retry wrapper
        try:
            response = await asyncio.to_thread(self._sync_generate_with_retry, self.model_name, full_prompt, None)
            text = self._extract_text_from_genai_response(response)
            return text
        except Exception as ex:
            logger.exception("Gemini generate_content failed: %s", ex)
            raise

    async def get_ai_json_response(self, prompt: str, **kwargs) -> dict:
        """
        Generates a JSON response, trying Gemini first and falling back to OpenAI.
        If both fail, it raises an exception to trigger the rule-based system.
        """
        # 1. Try Gemini First
        if self.gemini_client:
            try:
                await self._ensure_model()
                if self.model_name:
                    # Enhanced JSON-specific prompt
                    json_prompt = f"""{prompt}

CRITICAL INSTRUCTIONS:
1. Respond ONLY with valid JSON - no explanations, no markdown, no code blocks
2. Ensure all JSON keys are properly quoted
3. Do not prefix with "json" or wrap in backticks
4. Verify the JSON is parseable before responding

Example format: {{"key": "value", "items": []}}"""

                    response = await asyncio.to_thread(
                        self._sync_generate_with_retry,
                        self.model_name,
                        json_prompt,
                        self.default_json_config  # Use reusable config
                    )

                    text = self._extract_text_from_genai_response(response)
                    if text:
                        # Log raw response for debugging JSON parsing issues
                        logger.debug(f"Raw Gemini response before cleaning: {text}")

                        # Strip JSON fences and clean the response
                        clean_json = self._strip_json_fences(text)

                        # Additional cleaning for common AI JSON formatting issues
                        clean_json = clean_json.strip()
                        if clean_json.startswith('json'):
                            clean_json = clean_json[4:].strip()

                        # Validate JSON and soft-fail to OpenAI if malformed
                        try:
                            parsed_json = json.loads(clean_json)
                            ai_requests_counter.labels(model="gemini-json", status="success").inc()
                            return parsed_json
                        except json.JSONDecodeError as e:
                            logger.warning(f"Gemini returned invalid JSON: {e}. Trying OpenAI fallback.")
                            ai_requests_counter.labels(model="gemini-json", status="error").inc()
                            # Don't return here - let it fall through to OpenAI fallback

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
        raise Exception("Both Gemini and OpenAI failed to generate a valid JSON response.")


    async def get_product_qa(self, query: str, product: Optional[Product] = None, business_id: str = "feelori") -> str:
        """
        Answers a question. If a product is provided, answers about the product.
        If no product is provided, tries to answer generally.
        """
        prompt = query
        context = None

        if product:
            # If a product *is* provided, create a specific Q&A prompt
            prompt = self.create_qa_prompt(product, query, business_id=business_id)
        else:
            # No product provided, so just pass the query to the general model
            context = {"conversation_history": []} # Give it empty context
        
        # Call the *existing* generate_response function
        return await self.generate_response(prompt, context)  
  
    def create_qa_prompt(self, product, user_question: str, business_id: str = "feelori") -> str:
        """Creates a formatted prompt for answering a question about a specific product."""
        # Note: QA_PROMPT_TEMPLATE is not yet in BusinessConfig, so we use a default template
        # This can be moved to BusinessConfig.persona in the future
        qa_template = """You are FeelOri's jewelry expert assistant. Answer the customer's question using ONLY the product information provided below. Be helpful, accurate, and concise.

PRODUCT INFORMATION:
Title: {product_title}
Description: {product_description}
Tags: {product_tags}
Price: {product_price}

CUSTOMER QUESTION: "{user_question}"

INSTRUCTIONS:
- Answer based ONLY on the provided product information
- If the information isn't available, say "I don't have that specific information, but I can help you contact our team"
- Be friendly and professional
- Keep the answer concise (2-3 sentences max)

ANSWER:"""
        return qa_template.format(
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
            await self._ensure_model()
            if not self.model_name:
                return []
                
            # Create image part for the new API
            image_data = {
                'mime_type': mime_type,
                'data': image_bytes
            }
            
            # Get visual search prompt (not yet in BusinessConfig, using default)
            # This can be moved to BusinessConfig.persona in the future
            visual_search_prompt = """Analyze this jewelry photo. Return ONLY a comma-separated list of 3-4 of the most relevant lowercase keywords for a product search.
1. Start with the single most accurate primary category from this list: `[necklace, earrings, bangle, ring, set, choker, haram, jhumka, stud]`.
2. Add 2-3 other dominant keywords (e.g., primary stone, color, style).
Example: `set, ruby, gold plated, traditional`"""
            
            # Use multimodal content with the new client
            contents = [
                visual_search_prompt,
                image_data
            ]
            
            response = await asyncio.to_thread(
                self._sync_generate_with_retry,
                self.model_name,
                contents,
                None
            )
            
            text = (self._extract_text_from_genai_response(response) or "").strip().lower()
            return [k.strip() for k in text.split(',') if k.strip()]
            
        except Exception as e:
            logger.error(f"Error getting keywords for reranking: {e}")
            return []

    def _calculate_keyword_relevance(self, keywords: list[str], candidate: dict) -> float:
        """Calculates a relevance score based on keyword matches in title and tags."""
        if not keywords: 
            return 0.0
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
            if keyword in title_lower: 
                relevance_score += 0.5 
                matched_keywords.add(keyword)
            elif keyword in tags_lower: 
                relevance_score += 0.3 
                matched_keywords.add(keyword)
        
        if len(matched_keywords) > 1: 
            relevance_score += 0.5 * (len(matched_keywords) - 1)
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
            if best_match['similarity_score'] >= 0.92 and best_match['relevance_score'] >= 1.0: 
                match_type = 'exact'
            elif best_match['final_score'] >= 0.8: 
                match_type = 'very_similar'

            final_products = []
            for match in ranked_products:
                if match['final_score'] >= 0.6:
                    product = await shopify_service.get_product_by_handle(match['handle'])
                    if product: 
                        final_products.append(product)
            
            if final_products:
                return {'success': True, 'match_type': match_type, 'products': final_products[:5]}
            return {'success': False, 'message': 'No sufficiently matching products found.'}
        except Exception as e:
            logger.error(f"Error in hybrid visual search: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

# Globally accessible instance - safe now because __init__ is non-blocking
# For stricter setups, consider moving this to FastAPI startup event handler
ai_service = AIService()