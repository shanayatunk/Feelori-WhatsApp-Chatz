import pytest
from app.server import EnhancedSecurityService, EnhancedSecurityService as SecurityService # Assuming server.py is in app/

class TestEnhancedSecurityService:

    def test_sanitize_phone_number_valid(self):
        """Tests that valid phone numbers are formatted correctly."""
        assert SecurityService.sanitize_phone_number("123-456-7890") == "+1234567890"
        assert SecurityService.sanitize_phone_number("+1 (555) 867-5309") == "+15558675309"
        assert SecurityService.sanitize_phone_number("447911123456") == "+447911123456"

    def test_sanitize_phone_number_invalid(self):
        """Tests that invalid phone numbers raise a ValueError."""
        with pytest.raises(ValueError, match="Invalid phone number format"):
            SecurityService.sanitize_phone_number("123") # Too short
        
        with pytest.raises(ValueError, match="Invalid phone number length"):
            SecurityService.sanitize_phone_number("1234567890123456789012345") # Too long

    def test_validate_message_content_valid(self):
        """Tests valid message content."""
        assert SecurityService.validate_message_content("  Hello world!  ") == "Hello world!"

    def test_validate_message_content_suspicious(self):
        """Tests that messages with suspicious content raise a ValueError."""
        suspicious_message = "Hello <script>alert('xss')</script>"
        with pytest.raises(ValueError, match="Suspicious message content detected"):
            SecurityService.validate_message_content(suspicious_message)