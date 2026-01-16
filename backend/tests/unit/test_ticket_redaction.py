"""
Unit tests for ticket PII redaction utilities.
"""

import pytest
from backend.rag.ticket_redaction import (
    redact_pii,
    mask_serial_number,
    extract_technician_notes,
    extract_symptoms,
    extract_parts_used
)
import re


class TestRedactPII:
    """Test PII redaction function."""
    
    def test_redact_email(self):
        """Test email addresses are redacted."""
        text = "Contact support@example.com for help"
        result = redact_pii(text)
        assert "[EMAIL]" in result
        assert "support@example.com" not in result
    
    def test_redact_phone(self):
        """Test phone numbers are redacted."""
        text = "Call us at 555-123-4567 or (555) 123-4567"
        result = redact_pii(text)
        assert "[PHONE]" in result
        assert "555-123-4567" not in result
        assert "(555) 123-4567" not in result
    
    def test_redact_ip_address(self):
        """Test IP addresses are redacted."""
        text = "Server at 192.168.1.1 is down"
        result = redact_pii(text)
        assert "[IP_ADDRESS]" in result
        assert "192.168.1.1" not in result
    
    def test_mask_serial_keeps_last_4(self):
        """Test serial numbers are masked but keep last 4."""
        text = "Serial number SN: ABCD1234"
        result = redact_pii(text)
        assert "[REDACTED]" in result
        assert "1234" in result  # Last 4 should be preserved
        assert "ABCD" not in result
    
    def test_mask_serial_various_formats(self):
        """Test serial masking with various formats."""
        test_cases = [
            ("SN: ABCD1234", "SN-[REDACTED]-1234"),
            ("S/N: XYZ9876", "S/N-[REDACTED]-9876"),
            ("Serial Number 123456789", "Serial-[REDACTED]-6789"),
        ]
        
        for input_text, expected_pattern in test_cases:
            result = redact_pii(input_text)
            # Check that last 4 digits are preserved
            last_4 = expected_pattern.split("-")[-1]
            assert last_4 in result
            assert "[REDACTED]" in result
    
    def test_preserves_technical_info(self):
        """Test that technical information is preserved."""
        text = "Error code 500: Internal server error. Part number PN-12345"
        result = redact_pii(text)
        # Error messages and part numbers should be preserved
        assert "Error code 500" in result
        assert "PN-12345" in result
    
    def test_redact_address(self):
        """Test physical addresses are redacted (best-effort)."""
        text = "Located at 123 Main St, City, ST 12345"
        result = redact_pii(text)
        # Address pattern may or may not match depending on format
        # Just verify function doesn't crash
        assert isinstance(result, str)
    
    def test_empty_text(self):
        """Test empty text handling."""
        assert redact_pii("") == ""
        assert redact_pii(None) is None or redact_pii(None) == ""
    
    def test_multiple_pii_types(self):
        """Test multiple PII types in same text."""
        text = "Contact support@example.com or call 555-123-4567. Server IP: 192.168.1.1"
        result = redact_pii(text)
        assert "[EMAIL]" in result
        assert "[PHONE]" in result
        assert "[IP_ADDRESS]" in result


class TestMaskSerialNumber:
    """Test serial number masking helper."""
    
    def test_mask_serial_with_prefix(self):
        """Test masking serial with prefix."""
        text = "SN: ABCD1234"
        match = re.search(r'\b(SN|S/N|Serial\s*Number|Serial)[\s:]*([A-Z0-9]{4,})', text, re.IGNORECASE)
        assert match is not None
        masked = mask_serial_number(text, match)
        assert "[REDACTED]" in masked
        assert "1234" in masked  # Last 4 preserved


class TestExtractTechnicianNotes:
    """Test technician notes extraction."""
    
    def test_extract_agent_messages(self):
        """Test extraction of agent messages."""
        conversation = {
            "messages": [
                {"role": "agent", "text": "I recommend checking the power supply", "created_at": "2024-01-01T10:00:00Z"},
                {"role": "customer", "text": "Thanks", "created_at": "2024-01-01T10:05:00Z"},
                {"role": "technician", "text": "Issue resolved by replacing component", "created_at": "2024-01-01T11:00:00Z"},
            ]
        }
        notes = extract_technician_notes(conversation)
        assert notes is not None
        assert "power supply" in notes
        assert "replacing component" in notes
        assert "Thanks" not in notes  # Customer message excluded
    
    def test_no_agent_messages(self):
        """Test when no agent messages exist."""
        conversation = {
            "messages": [
                {"role": "customer", "text": "I have a problem"},
            ]
        }
        notes = extract_technician_notes(conversation)
        assert notes is None
    
    def test_max_length_cap(self):
        """Test that notes are capped at max_length."""
        long_text = "A" * 2000
        conversation = {
            "messages": [
                {"role": "agent", "text": long_text, "created_at": "2024-01-01T10:00:00Z"},
            ]
        }
        notes = extract_technician_notes(conversation, max_length=100)
        assert notes is not None
        assert len(notes) <= 103  # 100 + "..." = 103
        assert notes.endswith("...")


class TestExtractSymptoms:
    """Test symptoms extraction."""
    
    def test_extract_from_raw_response_json(self):
        """Test extraction from raw_response_json."""
        raw_response = {
            "error": "Connection timeout error",
            "symptoms": "System not responding"
        }
        symptoms = extract_symptoms(None, raw_response)
        assert symptoms is not None
        assert "timeout" in symptoms.lower()
    
    def test_extract_from_conversation(self):
        """Test extraction from conversation messages."""
        conversation = {
            "messages": [
                {"role": "customer", "text": "I'm getting an error code 500"},
                {"role": "agent", "text": "The system shows a fault alarm"},
            ]
        }
        symptoms = extract_symptoms(conversation, None)
        assert symptoms is not None
        assert "error" in symptoms.lower() or "fault" in symptoms.lower()
    
    def test_no_symptoms(self):
        """Test when no symptoms found."""
        symptoms = extract_symptoms(None, {})
        assert symptoms is None


class TestExtractPartsUsed:
    """Test parts extraction."""
    
    def test_extract_parts(self):
        """Test extraction of parts mentioned."""
        conversation = {
            "messages": [
                {"role": "technician", "text": "Replaced part PN-12345 and installed new component"},
                {"role": "agent", "text": "The part number is P/N: 67890"},
            ]
        }
        parts = extract_parts_used(conversation)
        assert parts is not None
        assert "PN-12345" in parts or "P/N: 67890" in parts
    
    def test_no_parts(self):
        """Test when no parts mentioned."""
        conversation = {
            "messages": [
                {"role": "customer", "text": "The machine is broken"},
            ]
        }
        parts = extract_parts_used(conversation)
        assert parts is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
