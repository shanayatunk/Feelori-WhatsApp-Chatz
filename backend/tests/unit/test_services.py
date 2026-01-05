# backend/tests/unit/test_services.py
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from app.services.whatsapp_service import WhatsAppService
from app.services.shopify_service import ShopifyService
from app.config.settings import settings
from app.utils.queue import RedisMessageQueue

# --- WhatsAppService tests ---
@pytest.mark.asyncio
async def test_whatsapp_send_message_success(mocker):
    """Test successful WhatsApp message sending and logging."""
    mock_log = mocker.patch('app.services.db_service.db_service.log_message', new_callable=AsyncMock)

    mock_response = AsyncMock(return_value=MagicMock(
        status_code=200, json=lambda: {"messages": [{"id": "wamid_123"}]}
    ))
    mocker.patch('app.services.whatsapp_service.WhatsAppService.resilient_api_call', mock_response)

    service = WhatsAppService(
        settings.whatsapp_access_token,
        settings.whatsapp_phone_id,
        settings.whatsapp_business_account_id
    )
    wamid = await service.send_message("+15551234567", "Hello World")

    assert wamid == "wamid_123"
    mock_log.assert_awaited_once()
    mock_response.assert_awaited_once()

# --- ShopifyService tests ---
@pytest.mark.asyncio
async def test_shopify_get_products_success(mocker):
    """Test successfully fetching products from Shopify using GraphQL."""
    # 1. Setup the Service (No args needed now)
    service = ShopifyService()
    
    # 2. Mock the Credentials lookup
    mocker.patch.object(
        service, 
        '_get_credentials', 
        return_value=("test.myshopify.com", "test-token", "test-storefront")
    )

    # 3. Create the GraphQL Response Structure (Edges/Nodes)
    graphql_data = {
        "data": {
            "products": {
                "edges": [
                    {
                        "node": {
                            "id": "gid://shopify/Product/123",
                            "title": "Test Crown",
                            "handle": "test-crown",
                            "descriptionHtml": "Beautiful crown",
                            "tags": ["crowns"],
                            "productType": "Accessories",
                            "variants": {
                                "edges": [{
                                    "node": {
                                        "id": "gid://shopify/ProductVariant/456",
                                        "price": "100.00",
                                        "inventoryQuantity": 5
                                    }
                                }]
                            },
                            "images": {
                                "edges": [{
                                    "node": {
                                        "originalSrc": "http://example.com/crown.jpg"
                                    }
                                }]
                            }
                        }
                    }
                ]
            }
        }
    }

    # 4. Mock the HTTP Response (Standard Mock, NOT AsyncMock)
    mock_response = mocker.Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = graphql_data
    mock_response.raise_for_status.return_value = None

    # 5. Mock resilient_api_call to return our mock_response
    # Since _execute_gql_query uses resilient_api_call, we need to mock that
    mocker.patch.object(
        service,
        'resilient_api_call',
        new_callable=mocker.AsyncMock,
        return_value=mock_response
    )

    # 6. Run & Assert
    products, count = await service.get_products("crowns", business_id="goldencollections")
    
    assert len(products) == 1
    assert products[0].title == "Test Crown"
    assert products[0].price == 100.0

# --- AIService tests ---
@pytest.mark.asyncio
async def test_ai_service_generates_response(mocker):
    """Test that the AI service generates a response using Gemini."""
    from app.services.ai_service import ai_service
    
    # Force gemini_client to be present so the check passes
    mocker.patch.object(ai_service, 'gemini_client', mocker.Mock())
    
    mock_gen = mocker.patch('app.services.ai_service.AIService._generate_gemini_response', new_callable=AsyncMock)
    mock_gen.return_value = ("This is an AI response.", None)

    response = await ai_service.generate_response("Tell me a joke", {})

    assert response == "This is an AI response."
    mock_gen.assert_awaited_once()

@pytest.mark.asyncio
async def test_ai_service_fallback_response():
    """Test fallback response when no AI clients are configured."""
    with patch('app.services.ai_service.ai_service.gemini_client', None), \
         patch('app.services.ai_service.ai_service.openai_client', None):
        from app.services.ai_service import ai_service
        response = await ai_service.generate_response("hello", {})

    assert "I'm sorry, I'm having trouble connecting" in response

# --- Queue Worker Logic Test ---
@pytest.mark.asyncio
async def test_process_message_from_queue_handles_send_failure(mocker):
    """
    Ensure _process_message_from_queue continues processing
    even if sending the WhatsApp message fails.
    """
    mocker.patch('app.services.order_service.get_or_create_customer', new_callable=AsyncMock, return_value={"name": "Alice"})
    mocker.patch('app.services.order_service.process_message', new_callable=AsyncMock, return_value="Hello User!")
    mock_whatsapp = mocker.patch('app.services.whatsapp_service.whatsapp_service.send_message', new_callable=AsyncMock, side_effect=Exception("API failure"))
    mock_update_history = mocker.patch('app.services.db_service.db_service.update_conversation_history', new_callable=AsyncMock)
    
    mock_counter_labels = mocker.patch('app.utils.metrics.message_counter.labels')
    mock_counter_labels.return_value.inc = MagicMock()

    data = {
        "from_number": "+15551234567",
        "message_text": "Hi",
        "message_type": "text",
        "profile_name": "Alice"
    }

    queue_instance = RedisMessageQueue(redis_client=AsyncMock())
    await queue_instance._process_message_from_queue(data)

    mock_whatsapp.assert_awaited_once()
    mock_update_history.assert_awaited_once()
    mock_counter_labels.assert_called_with(status="send_failed", message_type="text")
    mock_counter_labels.return_value.inc.assert_called_once()