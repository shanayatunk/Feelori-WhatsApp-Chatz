# backend/tests/integration/test_api.py
import hmac
import hashlib
import json
from unittest.mock import AsyncMock

from app.config.settings import settings

API_PREFIX = f"/api/{settings.api_version}"

def test_webhook_verification_success(test_client):
    params = {
        "hub.mode": "subscribe", "hub.challenge": "12345",
        "hub.verify_token": settings.whatsapp_verify_token
    }
    response = test_client.get(f"{API_PREFIX}/webhooks/whatsapp", params=params)
    assert response.status_code == 200
    assert response.text == "12345"

def test_webhook_verification_failure(test_client):
    params = {
        "hub.mode": "subscribe", "hub.challenge": "12345",
        "hub.verify_token": "wrong_token"
    }
    response = test_client.get(f"{API_PREFIX}/webhooks/whatsapp", params=params)
    assert response.status_code == 403

def test_handle_webhook_success(test_client, mocker):
    """Test a valid POST to the webhook, ensuring it's added to the message queue."""
    # FIX: The webhook router calls a function in order_service. We patch that function.
    mock_process_webhook = mocker.patch("app.routes.webhooks.order_service.process_webhook_message", new_callable=AsyncMock)
    
    payload = { "entry": [{ "changes": [{ "field": "messages", "value": { "messages": [{ "from": "15551234567", "id": "wamid.ID", "text": {"body": "Hello"}, "type": "text" }] } }] }] }
    payload_bytes = json.dumps(payload).encode('utf-8')
    signature = "sha256=" + hmac.new(settings.whatsapp_app_secret.encode('utf-8'), payload_bytes, hashlib.sha256).hexdigest()
    headers = { "X-Hub-Signature-256": signature, "Content-Type": "application/json" }
    
    response = test_client.post(f"{API_PREFIX}/webhooks/whatsapp", content=payload_bytes, headers=headers)
    
    assert response.status_code == 200
    # Assert that our mocked background task was called
    mock_process_webhook.assert_awaited_once()

def test_handle_webhook_invalid_signature(test_client, mocker):
    mock_process_webhook = mocker.patch("app.routes.webhooks.order_service.process_webhook_message", new_callable=AsyncMock)
    payload = {"entry": []}
    payload_bytes = json.dumps(payload).encode('utf-8')
    headers = { "X-Hub-Signature-256": "sha256=invalid", "Content-Type": "application/json" }
    
    response = test_client.post(f"{API_PREFIX}/webhooks/whatsapp", content=payload_bytes, headers=headers)
    assert response.status_code == 403
    mock_process_webhook.assert_not_awaited()

def test_admin_stats_unauthorized(test_client):
    response = test_client.get(f"{API_PREFIX}/admin/stats")
    assert response.status_code == 401