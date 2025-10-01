# backend/tests/unit/test_services.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.whatsapp_service import WhatsAppService
from app.services.shopify_service import ShopifyService
from app.config.settings import settings

@pytest.mark.asyncio
async def test_whatsapp_send_message_success(mocker):
    """Test successful message sending."""
    # FIX: Patch the log_message method on the actual db_service object at its source.
    mock_log = mocker.patch('app.services.db_service.db_service.log_message', new_callable=AsyncMock)

    mock_response = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: {"messages": [{"id": "wamid_123"}]}))
    mocker.patch('app.services.whatsapp_service.WhatsAppService.resilient_api_call', mock_response)
    
    service = WhatsAppService(settings.whatsapp_access_token, settings.whatsapp_phone_id, settings.whatsapp_business_account_id)
    wamid = await service.send_message("+15551234567", "Hello World")
    
    assert wamid == "wamid_123"
    mock_log.assert_awaited_once() # Verify the log was called
    mock_response.assert_awaited_once()

@pytest.mark.asyncio
async def test_shopify_get_products_success(mocker):
    """Test successfully fetching products."""
    mock_gql_response = { "data": { "products": { "edges": [{ "node": {
        "id": "gid://shopify/Product/1", "handle": "test-product", "title": "Test Product",
        "description": "A great product", "tags": ["test"], "featuredImage": {"url": "http://example.com/image.png"},
        "variants": {"edges": [{"node": {
            "id": "gid://shopify/ProductVariant/1", "sku": "SKU123",
            "priceV2": {"amount": "19.99", "currencyCode": "INR"}, "quantityAvailable": 10
        }}]}
    }}]}}}
    
    with patch('app.services.shopify_service.httpx.AsyncClient') as MockClient:
        mock_post = AsyncMock(return_value=MagicMock(status_code=200, json=lambda: mock_gql_response))
        mock_post.return_value.raise_for_status = MagicMock()
        MockClient.return_value.post = mock_post

        service = ShopifyService(settings.shopify_store_url, settings.shopify_access_token, settings.shopify_storefront_access_token)
        products, _ = await service.get_products(query="Test", limit=1)
    
    assert len(products) == 1
    assert products[0].title == "Test Product"

# --- AIService Tests ---
@pytest.mark.asyncio
async def test_ai_service_generates_response(mocker):
    """Test that the AI service generates a response."""
    mock_sync_call = mocker.patch('app.services.ai_service.AIService._sync_generate_with_retry')
    mock_response = MagicMock()
    mocker.patch('app.services.ai_service.AIService._extract_text_from_genai_response', return_value="This is an AI response.")
    mock_sync_call.return_value = mock_response

    from app.services.ai_service import ai_service
    response = await ai_service.generate_response("Tell me a joke", {})
    
    assert response == "This is an AI response."
    mock_sync_call.assert_called_once()

@pytest.mark.asyncio
async def test_ai_service_fallback_response():
    """Test the fallback response when no AI clients are configured."""
    # This test patches the internal clients and is already correct.
    with patch('app.services.ai_service.ai_service.gemini_client', None), \
         patch('app.services.ai_service.ai_service.openai_client', None):
        
        from app.services.ai_service import ai_service
        response = await ai_service.generate_response("hello", {})
    
    assert "I'm sorry, I'm having trouble connecting" in response