# backend/tests/unit/test_processing.py
import pytest
from unittest.mock import AsyncMock

# FIX: Add all the handler functions you are testing
from app.services.order_service import (
    handle_order_inquiry,
    handle_support_request,
    handle_general_inquiry
)

@pytest.fixture
def mock_customer():
    return {"phone_number": "+15551234567", "name": "Test User", "conversation_history": []}

@pytest.mark.skip(reason="Intent analysis is now AI-driven and requires different tests.")
@pytest.mark.parametrize("message, message_type, expected_intent", [
    ("i need to find a dress", "text", "product_search"),
    ("where is my stuff?", "text", "order_inquiry"),
    ("help me", "text", "support_request"),
])
@pytest.mark.asyncio
async def test_analyze_intent(message, message_type, mock_customer, expected_intent):
    # This test is skipped, but the import would have been needed here
    from app.services.order_service import analyze_intent
    intent = await analyze_intent(message, message_type, mock_customer, None)
    assert intent == expected_intent

@pytest.mark.asyncio
async def test_handle_order_inquiry_found(mocker, mock_customer):
    """Test order inquiry when a recent order is found."""
    mock_order = {
        "order_number": "#1001",
        "raw": {
            "name": "#1001", "created_at": "2025-10-01T10:00:00-04:00",
            "fulfillment_status": "unfulfilled", "financial_status": "paid",
            "current_total_price": "199.99", "currency": "INR",
            "line_items": [{"name": "Test Item", "quantity": 1}]
        }
    }
    mocker.patch('app.services.order_service.db_service.get_recent_orders_by_phone', new_callable=AsyncMock, return_value=[mock_order])
    
    response = await handle_order_inquiry(mock_customer["phone_number"], mock_customer)
    assert "Order #1001" in response
    assert "Unfulfilled" in response

@pytest.mark.asyncio
async def test_handle_support_request(mocker, mock_customer):
    """Test the support request handler."""
    mocker.patch('app.services.order_service.string_service.get_string', return_value="Our team will assist you.")
    response = await handle_support_request("I have a problem", mock_customer)
    assert "Our team will assist you." in response

@pytest.mark.asyncio
async def test_handle_general_inquiry(mocker, mock_customer):
    """Test that the general inquiry handler calls the AI service."""
    mock_ai_response = "This is a general AI-powered response."
    mocker.patch('app.services.order_service.ai_service.generate_response', new_callable=AsyncMock, return_value=mock_ai_response)
    response = await handle_general_inquiry("How are you?", mock_customer)
    assert response == mock_ai_response