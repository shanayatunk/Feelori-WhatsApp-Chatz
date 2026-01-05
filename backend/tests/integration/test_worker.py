# backend/tests/integration/test_worker.py
import pytest
from unittest.mock import AsyncMock
from app.utils.queue import RedisMessageQueue

@pytest.mark.asyncio
async def test_worker_processing_logic_success(mocker):
    """Tests the worker's message processing logic for a successful case."""
    expected_response = "Here is your response."

    # Patch the actual modules/functions used inside _process_message_from_queue
    mock_process_message = mocker.patch(
        "app.services.order_service.process_message",
        new_callable=AsyncMock,
        return_value=expected_response
    )
    mock_update_history = mocker.patch(
        "app.services.order_service.db_service.update_conversation_history",
        new_callable=AsyncMock
    )
    mock_whatsapp = mocker.patch(
        "app.services.whatsapp_service.whatsapp_service",
        new_callable=AsyncMock
    )

    queue = RedisMessageQueue(redis_client=AsyncMock())
    sample_data = {
        "from_number": "123",
        "message_text": "hi",
        "message_type": "text",
        "wamid": "w1",
        "profile_name": "Test",
        "quoted_wamid": None,
        "business_id": "feelori"  # Explicitly adding this makes the test clearer
    }

    # Call the worker's processing method
    await queue._process_message_from_queue(sample_data)

    # Assertions: verify all critical functions were called
    mock_process_message.assert_awaited_once()
    
    # --- FIX: Added business_id="feelori" ---
    mock_whatsapp.send_message.assert_awaited_once_with("123", expected_response, business_id="feelori")
    
    mock_update_history.assert_awaited_once()


@pytest.mark.asyncio
async def test_worker_processing_logic_failure(mocker):
    """Tests that if processing fails, no reply is sent."""
    # Patch process_message to raise an exception
    mocker.patch(
        "app.services.order_service.process_message",
        new_callable=AsyncMock,
        side_effect=Exception("Processing failed")
    )
    # Patch other dependencies
    mocker.patch(
        "app.services.order_service.get_or_create_customer",
        new_callable=AsyncMock,
        return_value={"name": "Test User"}
    )
    mock_whatsapp = mocker.patch(
        "app.services.whatsapp_service.whatsapp_service",
        new_callable=AsyncMock
    )

    queue = RedisMessageQueue(redis_client=AsyncMock())
    sample_data = {
        "from_number": "123",
        "message_text": "hi",
        "message_type": "text",
        "wamid": "w1",
        "profile_name": "Test",
        "quoted_wamid": None
    }

    # Call the worker; should handle exception internally and not raise
    await queue._process_message_from_queue(sample_data)

    # Verify send_message was never called due to processing failure
    mock_whatsapp.send_message.assert_not_awaited()