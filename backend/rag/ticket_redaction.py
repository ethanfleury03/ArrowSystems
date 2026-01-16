"""
PII redaction utilities for ticket cache artifacts.

Redacts personally identifiable information (PII) from ticket text before indexing.
"""

import re
from typing import Optional


def mask_serial_number(text: str, serial_match: re.Match) -> str:
    """
    Mask a serial number, keeping last 4 characters/digits.
    
    Args:
        text: Full text containing the serial
        serial_match: Regex match object for the serial number
        
    Returns:
        Masked serial string (e.g., "SN-[REDACTED]-1234")
    """
    serial = serial_match.group(0)
    
    # Extract last 4 alphanumeric characters
    alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', serial)
    if len(alphanumeric) >= 4:
        last_4 = alphanumeric[-4:]
    else:
        last_4 = alphanumeric
    
    # Determine prefix (SN, S/N, Serial, etc.)
    prefix_match = serial_match.group(1) if serial_match.lastindex >= 1 else "SN"
    prefix = prefix_match.strip() if prefix_match else "SN"
    
    return f"{prefix}-[REDACTED]-{last_4}"


def redact_pii(text: str) -> str:
    """
    Redact PII from ticket text before indexing.
    
    Redacts:
    - Email addresses -> [EMAIL]
    - Phone numbers -> [PHONE]
    - IP addresses -> [IP_ADDRESS]
    - Physical addresses -> [ADDRESS] (best-effort)
    - Serial numbers -> partial mask keeping last 4 (e.g., SN-[REDACTED]-1234)
    
    Preserves technical information:
    - Error messages
    - Part numbers
    - Model names
    - Technical descriptions
    
    Args:
        text: Text to redact
        
    Returns:
        Text with PII redacted
    """
    if not text:
        return text
    
    result = text
    
    # Email addresses
    # Pattern: word@domain.tld
    result = re.sub(
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        '[EMAIL]',
        result
    )
    
    # Phone numbers (US format: XXX-XXX-XXXX, XXX.XXX.XXXX, (XXX) XXX-XXXX, etc.)
    # Also handles international formats with + prefix
    result = re.sub(
        r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        '[PHONE]',
        result
    )
    
    # IP addresses (IPv4)
    result = re.sub(
        r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',
        '[IP_ADDRESS]',
        result
    )
    
    # Serial numbers (partial mask - keep last 4)
    # Patterns: SN: ABCD1234, S/N: ABCD1234, Serial Number 123456789, Serial: ABC123
    # Match common prefixes and capture alphanumeric sequences
    serial_patterns = [
        r'\b(SN|S/N|Serial\s*Number|Serial)[\s:]*([A-Z0-9]{4,})',
        r'\b(SN|S/N|Serial\s*Number|Serial)[\s:]*([A-Z]{2,}\d{4,})',
        r'\b(SN|S/N|Serial\s*Number|Serial)[\s:]*(\d{6,})',
    ]
    
    for pattern in serial_patterns:
        def replace_serial(match):
            return mask_serial_number(result, match)
        
        result = re.sub(pattern, replace_serial, result, flags=re.IGNORECASE)
    
    # Physical addresses (best-effort)
    # Pattern: Street number + street name + city/state/zip
    # This is heuristic and may have false positives
    # Look for patterns like "123 Main St, City, ST 12345" or "123 Main Street"
    address_pattern = r'\b\d+\s+[A-Z][a-z]+(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Lane|Ln|Boulevard|Blvd|Court|Ct|Place|Pl))?\s*,?\s*[A-Z][a-z]+,?\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?\b'
    result = re.sub(address_pattern, '[ADDRESS]', result, flags=re.IGNORECASE)
    
    return result


def extract_technician_notes(conversation_json: Optional[dict], max_length: int = 1500) -> Optional[str]:
    """
    Extract technician notes from conversation JSON.
    
    Looks for messages where role is "agent" or "technician".
    Returns the most recent relevant messages (up to max_length chars).
    
    Args:
        conversation_json: Conversation JSON dict with messages array
        max_length: Maximum length of extracted notes
        
    Returns:
        Extracted technician notes string, or None if none found
    """
    if not conversation_json:
        return None
    
    messages = conversation_json.get("messages", [])
    if not isinstance(messages, list):
        return None
    
    # Filter for agent/technician messages
    agent_messages = [
        msg for msg in messages
        if isinstance(msg, dict) and msg.get("role") in ("agent", "technician")
    ]
    
    if not agent_messages:
        return None
    
    # Sort by created_at (most recent first)
    try:
        agent_messages.sort(
            key=lambda m: m.get("created_at", ""),
            reverse=True
        )
    except Exception:
        pass  # If sorting fails, use original order
    
    # Take last 5 messages and join
    selected_messages = agent_messages[:5]
    notes_parts = []
    
    for msg in selected_messages:
        text = msg.get("text", "")
        if isinstance(text, str) and text.strip():
            notes_parts.append(text.strip())
    
    if not notes_parts:
        return None
    
    notes = "\n".join(notes_parts)
    
    # Cap length
    if len(notes) > max_length:
        notes = notes[:max_length] + "..."
    
    return notes


def extract_symptoms(conversation_json: Optional[dict], raw_response_json: Optional[dict], max_length: int = 1000) -> Optional[str]:
    """
    Extract error messages/symptoms from conversation or raw_response_json.
    
    Prefers explicit fields in raw_response_json if present.
    Otherwise, searches conversation messages for error-related keywords.
    
    Args:
        conversation_json: Conversation JSON dict
        raw_response_json: Raw response JSON dict
        max_length: Maximum length of extracted symptoms
        
    Returns:
        Extracted symptoms string, or None if none found
    """
    # First, check raw_response_json for explicit error/symptom fields
    if raw_response_json:
        for key in ("error", "errors", "symptom", "symptoms", "error_message", "error_messages"):
            value = raw_response_json.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()[:max_length]
            elif isinstance(value, list):
                # Join list of errors
                error_list = [str(e).strip() for e in value if str(e).strip()]
                if error_list:
                    return "\n".join(error_list)[:max_length]
    
    # Fallback: search conversation messages
    if not conversation_json:
        return None
    
    messages = conversation_json.get("messages", [])
    if not isinstance(messages, list):
        return None
    
    # Keywords that indicate errors/symptoms
    error_keywords = ["error", "fault", "alarm", "exception", "code", "failed", "failure", "issue", "problem", "warning"]
    
    symptom_lines = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        text = msg.get("text", "")
        if not isinstance(text, str):
            continue
        
        # Check if message contains error keywords
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in error_keywords):
            # Extract relevant lines
            lines = text.split("\n")
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in error_keywords):
                    line_stripped = line.strip()
                    if line_stripped and len(line_stripped) > 10:  # Filter very short lines
                        symptom_lines.append(line_stripped)
    
    if not symptom_lines:
        return None
    
    # Join and cap length
    symptoms = "\n".join(symptom_lines[:10])  # Limit to 10 lines
    if len(symptoms) > max_length:
        symptoms = symptoms[:max_length] + "..."
    
    return symptoms


def extract_parts_used(conversation_json: Optional[dict], max_length: int = 500) -> Optional[str]:
    """
    Extract parts used from conversation text (best-effort).
    
    Looks for patterns like "part", "replaced", "installed", "PN:", "P/N".
    
    Args:
        conversation_json: Conversation JSON dict
        max_length: Maximum length of extracted parts text
        
    Returns:
        Extracted parts string, or None if none found
    """
    if not conversation_json:
        return None
    
    messages = conversation_json.get("messages", [])
    if not isinstance(messages, list):
        return None
    
    # Keywords that indicate parts
    part_keywords = ["part", "replaced", "installed", "pn:", "p/n", "part number", "component"]
    
    parts_lines = []
    
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        
        text = msg.get("text", "")
        if not isinstance(text, str):
            continue
        
        text_lower = text.lower()
        if any(keyword in text_lower for keyword in part_keywords):
            # Extract relevant sentences/lines
            lines = text.split("\n")
            for line in lines:
                line_lower = line.lower()
                if any(keyword in line_lower for keyword in part_keywords):
                    line_stripped = line.strip()
                    if line_stripped and len(line_stripped) > 10:
                        parts_lines.append(line_stripped)
    
    if not parts_lines:
        return None
    
    # Join and cap length
    parts = "\n".join(parts_lines[:5])  # Limit to 5 lines
    if len(parts) > max_length:
        parts = parts[:max_length] + "..."
    
    return parts
