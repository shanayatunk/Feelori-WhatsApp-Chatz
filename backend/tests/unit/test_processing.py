# backend/tests/unit/test_processing.py

import pytest
from app.server import (
    analyze_intent,
    handle_greeting,
    handle_product_search,
    handle_order_inquiry,
    handle_support_request,
    handle_general_inquiry,
    Product
)

# --- Test analyze_intent ---
# This function does not depend on services, so it doesn't need the fixture.
@pytest.mark.parametrize("message, message_type, expected_intent", [
    ("hello there", "text", "greeting"),
    ("i need to find a dress", "text", "product_search"),
])
@pytest.mark.asyncio
async def test_analyze_intent(message, message_type, expected_intent):
    intent = await analyze_intent(message, message_type, {})
    assert intent == expected_intent

# --- Test Handlers ---

@pytest.mark.asyncio
async def test_handle_greeting():
    response = await handle_greeting("123", {"conversation_history": []})
    assert "Hello! Welcome to Feelori!" in response

@pytest.mark.asyncio
async def test_handle_product_search_found(mock_services): # FIX: Request the mock_services fixture
    """Test product search when products are found."""
    mock_product = Product(id="1", title="Stylish Dress", price=49.99, description="A dress.")
    # FIX: Use the fixture to mock the service call
    mock_services.shopify_service.get_products.return_value = [mock_product]
    
    response = await handle_product_search("show me a dress", {})
    assert "Found this perfect match!" in response
    assert "Stylish Dress" in response

@pytest.mark.asyncio
async def test_handle_product_search_not_found(mock_services): # FIX: Request the mock_services fixture
    """Test product search when no products are found."""
    mock_services.shopify_service.get_products.return_value = []
    
    response = await handle_product_search("show me something unique", {})
    assert "I couldn't find products matching" in response

@pytest.mark.asyncio
async def test_handle_order_inquiry_not_found(mock_services): # FIX: Request the mock_services fixture
    """Test order inquiry when no orders are found."""
    mock_services.shopify_service.search_orders_by_phone.return_value = []
    
    response = await handle_order_inquiry("+15551234567", {})
    assert "I couldn't find any recent orders" in response

@pytest.mark.asyncio
async def test_handle_support_request():
    response = await handle_support_request("I have a problem", {})
    assert "For detailed assistance, you can:" in response

@pytest.mark.asyncio
async def test_handle_general_inquiry(mock_services): # FIX: Request the mock_services fixture
    """Test that the general inquiry handler calls the AI service."""
    mock_ai_response = "This is a general AI-powered response."
    mock_services.ai_service.generate_response.return_value = mock_ai_response
    
    response = await handle_general_inquiry("How are you?", {"conversation_history": []})
    assert response == mock_ai_response