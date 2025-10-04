# backend/tests/unit/test_security.py

import pytest
import html
from app.services.security_service import EnhancedSecurityService as SecurityService

class TestEnhancedSecurityService:

    # --- Phone number sanitization tests ---
    def test_sanitize_phone_number_valid(self):
        """Valid phone numbers should be formatted correctly."""
        assert SecurityService.sanitize_phone_number("123-456-7890") == "+1234567890"
        assert SecurityService.sanitize_phone_number("+1 (555) 867-5309") == "+15558675309"
        assert SecurityService.sanitize_phone_number("447911123456") == "+447911123456"

    def test_sanitize_phone_number_invalid_returns_empty(self):
        """Invalid phone numbers return an empty string."""
        assert SecurityService.sanitize_phone_number("123") == ""  # Too short
        assert SecurityService.sanitize_phone_number("12345678901234567890") == ""  # Too long
        assert SecurityService.sanitize_phone_number(None) == ""  # None value
        assert SecurityService.sanitize_phone_number("not a number") == ""  # Invalid chars

    # --- Message content validation tests ---
    def test_validate_message_content_valid(self):
        """Valid messages should be stripped and escaped correctly."""
        msg = "  Hello world!   "
        expected = html.escape(msg.strip())
        assert SecurityService.validate_message_content(msg) == expected

    def test_validate_message_content_too_long(self):
        """Messages exceeding the limit should raise ValueError."""
        long_message = "a" * 5000
        with pytest.raises(ValueError, match="Message too long"):
            SecurityService.validate_message_content(long_message)

    def test_validate_message_content_suspicious_html(self):
        """Messages containing HTML should be escaped to prevent XSS."""
        suspicious_message = "<script>alert('xss')</script>"
        escaped_message = html.escape(suspicious_message.strip())
        result = SecurityService.validate_message_content(suspicious_message)
        assert result == escaped_message
        # Ensure that dangerous characters are converted
        assert "<" not in result
        assert ">" not in result
        assert "&" in result  # HTML escape uses &lt; and &gt;

    def test_validate_message_content_normal_text(self):
        """Normal text with special characters should be escaped correctly."""
        message = "Hello & Welcome <User>!"
        expected = html.escape(message.strip())
        assert SecurityService.validate_message_content(message) == expected
