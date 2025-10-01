# backend/tests/unit/test_security.py

import pytest
# Corrected the import path to point to the actual service file
from app.services.security_service import EnhancedSecurityService as SecurityService

class TestEnhancedSecurityService:

    def test_sanitize_phone_number_valid(self):
        """Tests that valid phone numbers are formatted correctly."""
        assert SecurityService.sanitize_phone_number("123-456-7890") == "+1234567890"
        assert SecurityService.sanitize_phone_number("+1 (555) 867-5309") == "+15558675309"
        assert SecurityService.sanitize_phone_number("447911123456") == "+447911123456"

    def test_sanitize_phone_number_invalid_returns_empty(self):
        """
        Tests that invalid phone numbers return an empty string,
        matching the current implementation.
        """
        assert SecurityService.sanitize_phone_number("123") == "" # Too short
        assert SecurityService.sanitize_phone_number("12345678901234567890") == "" # Too long
        assert SecurityService.sanitize_phone_number(None) == "" # None value
        assert SecurityService.sanitize_phone_number("not a number") == "" # Invalid characters

    def test_validate_message_content_valid(self):
        """Tests that valid message content is stripped correctly."""
        assert SecurityService.validate_message_content("  Hello world!   ") == "Hello world!"

    def test_validate_message_content_too_long(self):
        """Tests that messages exceeding the length limit raise a ValueError."""
        long_message = "a" * 5000
        with pytest.raises(ValueError, match="Message too long"):
            SecurityService.validate_message_content(long_message)

    def test_validate_message_content_suspicious(self):
        """
        Tests that messages with suspicious content currently pass but could be enhanced.
        NOTE: This test currently confirms existing behavior. A future enhancement
        could be to make this test fail by implementing content scanning.
        """
        suspicious_message = "Hello <script>alert('xss')</script>"
        # The current implementation only strips whitespace, so the suspicious content remains.
        assert SecurityService.validate_message_content(suspicious_message) == suspicious_message
