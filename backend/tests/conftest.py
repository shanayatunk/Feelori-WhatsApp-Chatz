# backend/tests/conftest.py

import pytest
from fastapi.testclient import TestClient # noqa: E402
from dotenv import load_dotenv
from unittest.mock import AsyncMock

# Load test environment variables before importing the app
load_dotenv(dotenv_path=".env.test")

from app.server import app, ServiceContainer # noqa: E402

@pytest.fixture
def mock_services(mocker):
    """
    Provides a ServiceContainer with all services mocked for unit tests.
    """
    services_mock = ServiceContainer()
    services_mock.db_service = AsyncMock()
    services_mock.cache_service = AsyncMock()
    services_mock.whatsapp_service = AsyncMock()
    services_mock.shopify_service = AsyncMock()
    services_mock.ai_service = AsyncMock()
    services_mock.message_queue = AsyncMock()
    
    # This replaces the global `services` object with our mock container for the test's duration
    mocker.patch("app.server.services", services_mock)
    return services_mock


@pytest.fixture(scope="function") # FIX: Changed scope from "module" to "function"
def test_client(mocker):
    """
    Provides a TestClient for API integration tests.
    It prevents the real background workers from starting.
    """
    mocker.patch("app.server.RedisMessageQueue.start_workers", new_callable=AsyncMock)
    mocker.patch("app.server.RedisMessageQueue.stop_workers", new_callable=AsyncMock)
    
    # The app's lifespan is managed by the TestClient context manager
    with TestClient(app) as client:
        yield client