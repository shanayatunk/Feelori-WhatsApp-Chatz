# backend/tests/unit/test_services.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from app.server import WhatsAppService, ShopifyService, AIService, Product

# Mock httpx client for all service tests
@pytest.fixture
def mock_httpx_client():
    return AsyncMock()

# --- WhatsAppService Tests ---

@pytest.fixture
def whatsapp_service(mock_httpx_client):
    return WhatsAppService(access_token="test_token", phone_id="12345", http_client=mock_httpx_client)

@pytest.mark.asyncio
async def test_whatsapp_send_message_success(whatsapp_service, mock_httpx_client):
    """Test successful message sending."""
    mock_httpx_client.post.return_value = AsyncMock(status_code=200, json=lambda: {"messages": [{"id": "wamid_123"}]})
    
    success = await whatsapp_service.send_message("+15551234567", "Hello World")
    
    assert success is True
    mock_httpx_client.post.assert_awaited_once()
    # Check if the URL and payload are correct
    args, kwargs = mock_httpx_client.post.call_args
    assert args[0] == "https://graph.facebook.com/v18.0/12345/messages"
    assert kwargs["json"]["to"] == "+15551234567"
    assert kwargs["json"]["text"]["body"] == "Hello World"

@pytest.mark.asyncio
async def test_whatsapp_send_message_failure(whatsapp_service, mock_httpx_client):
    """Test failed message sending due to API error."""
    mock_httpx_client.post.return_value = AsyncMock(status_code=400, text="Invalid request")
    
    success = await whatsapp_service.send_message("+15551234567", "Test")
    
    assert success is False

# --- ShopifyService Tests ---

@pytest.fixture
def shopify_service(mock_httpx_client):
    return ShopifyService(store_url="test.myshopify.com", access_token="test_token", http_client=mock_httpx_client)

@pytest.mark.asyncio
async def test_shopify_get_products_success(shopify_service, mock_httpx_client):
    """Test successfully fetching products."""
    mock_response_data = {
        "products": [{
            "id": 1, "title": "Test Product", "body_html": "A great product",
            "variants": [{"price": "19.99", "inventory_quantity": 10}],
            "images": [{"src": "http://example.com/image.png"}]
        }]
    }
    mock_httpx_client.get.return_value = AsyncMock(status_code=200, json=lambda: mock_response_data)
    
    products = await shopify_service.get_products(query="Test", limit=1)
    
    assert len(products) == 1
    assert isinstance(products[0], Product)
    assert products[0].title == "Test Product"
    assert products[0].price == 19.99

@pytest.mark.asyncio
async def test_shopify_get_products_empty(shopify_service, mock_httpx_client):
    """Test fetching when no products are found."""
    mock_httpx_client.get.return_value = AsyncMock(status_code=200, json=lambda: {"products": []})
    
    products = await shopify_service.get_products()
    
    assert len(products) == 0

# --- AIService Tests ---

@pytest.fixture
def ai_service():
    # Mock the OpenAI client directly
    mock_openai_client = AsyncMock()
    mock_openai_client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="This is an AI response."))]
    )
    
    service = AIService(openai_api_key="fake_key")
    service.openai_client = mock_openai_client
    return service

@pytest.mark.asyncio
async def test_ai_service_generates_response(ai_service):
    """Test that the AI service generates a response using its primary client."""
    response = await ai_service.generate_response("Tell me a joke")
    
    assert response == "This is an AI response."
    ai_service.openai_client.chat.completions.create.assert_awaited_once()

def test_ai_service_fallback_response():
    """Test the fallback response when no AI clients are configured."""
    # Initialize service with no keys
    fallback_service = AIService()
    response = fallback_service._generate_fallback_response("hello")
    
    assert "Hello! Welcome to Feelori!" in response