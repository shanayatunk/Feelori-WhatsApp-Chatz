

import pytest
from fastapi.testclient import TestClient
from dotenv import load_dotenv
from unittest.mock import AsyncMock

# CRITICAL FIX: Load environment variables FIRST, before any app imports.
# This ensures that when the app is imported, Pydantic can find the required variables.
load_dotenv(dotenv_path="backend/.env.test")

# Now it's safe to import the application and its components
from app.main import app # noqa: E402


@pytest.fixture
def mock_services(mocker):
    """
    Provides mocks for key services to isolate tests.
    This is useful for unit tests where you don't want to make real external calls.
    """
    # We mock the actual service instances directly where they are used.
    mocker.patch('app.services.order_service.db_service', new_callable=AsyncMock)
    mocker.patch('app.routes.admin.db_service', new_callable=AsyncMock)
    # Add other service mocks as needed for different tests
    
@pytest.fixture(scope="function")
def test_client(mocker):
    """
    Provides a TestClient for API integration tests.
    It prevents the real background workers from starting.
    """
    # Prevent the message queue from starting its background processing loop during tests
    mocker.patch("app.utils.queue.RedisMessageQueue.start_workers", new_callable=AsyncMock)
    mocker.patch("app.utils.queue.RedisMessageQueue.stop_workers", new_callable=AsyncMock)
    
    # The app's lifespan (startup/shutdown events) is managed by the TestClient
    with TestClient(app) as client:
        yield client

