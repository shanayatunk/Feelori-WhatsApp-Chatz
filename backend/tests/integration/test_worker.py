# backend/tests/integration/test_worker.py

import pytest
from unittest.mock import AsyncMock
from app.server import RedisMessageQueue

@pytest.mark.asyncio
async def test_worker_process_message_success(mocker, mock_services):
    """
    Tests the worker's message processing logic.
    - Mocks the main `process_message` function to simulate a successful response.
    - Verifies that the WhatsApp service is called to send the response back.
    """
    expected_response = "Here are the products you asked for."
    
    mocker.patch("app.server.process_message", new_callable=AsyncMock, return_value=expected_response)
    
    sample_message_data = {
        "from_number": "+15559876543",
        "message_text": "show me dresses",
        "message_type": "text"
    }

    # Instantiate the queue locally for the test
    queue = RedisMessageQueue(redis_client=AsyncMock())
    await queue._process_message(sample_message_data)

    from app.server import process_message
    process_message.assert_awaited_once_with(
        sample_message_data["from_number"],
        sample_message_data["message_text"],
        sample_message_data["message_type"]
    )
    
    mock_services.whatsapp_service.send_message.assert_awaited_once_with(
        sample_message_data["from_number"],
        expected_response
    )

@pytest.mark.asyncio
async def test_worker_process_message_failure(mocker, mock_services):
    """
    Tests that if the main processing logic fails, the worker handles it
    gracefully and does NOT attempt to send a message.
    """
    mocker.patch("app.server.process_message", new_callable=AsyncMock, side_effect=Exception("Processing failed"))
    
    sample_message_data = {
        "from_number": "+15551112222",
        "message_text": "a message that causes an error",
        "message_type": "text"
    }

    queue = RedisMessageQueue(redis_client=AsyncMock())
    await queue._process_message(sample_message_data)
    
    mock_services.whatsapp_service.send_message.assert_not_awaited()