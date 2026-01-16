"""
Ticket cache artifacts transformer for RAG ingestion.

Converts cache-eligible Zendesk ticket judgments into LlamaIndex-compatible artifacts.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator


class TicketCacheArtifact(BaseModel):
    """Pydantic model for a cached ticket solution artifact."""
    
    id: str = Field(..., description="Unique artifact ID (format: 'ticket:{ticket_id}')")
    text: str = Field(..., description="Full text content for RAG indexing")
    metadata: Dict[str, Any] = Field(..., description="Metadata dict compatible with LlamaIndex TextNode")
    
    @field_validator('id')
    @classmethod
    def validate_id(cls, v: str) -> str:
        """Ensure ID follows expected format."""
        if not v.startswith('ticket:'):
            raise ValueError(f"ID must start with 'ticket:', got: {v}")
        return v
    
    @field_validator('text')
    @classmethod
    def validate_text(cls, v: str) -> str:
        """Ensure text is non-empty."""
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()
    
    @field_validator('metadata')
    @classmethod
    def validate_metadata(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure required metadata keys exist."""
        required_keys = {'document_id', 'file_name', 'content_type', 'source'}
        missing = required_keys - set(v.keys())
        if missing:
            raise ValueError(f"Missing required metadata keys: {missing}")
        return v


def extract_problem(raw: Dict[str, Any]) -> Optional[str]:
    """Extract problem summary from raw_response_json."""
    p = raw.get("problem")
    if isinstance(p, str):
        return p.strip() or None
    if isinstance(p, dict):
        for k in ("summary", "text", "problem", "description"):
            v = p.get(k)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return None


def extract_steps(raw: Dict[str, Any]) -> List[str]:
    """Extract resolution steps from raw_response_json."""
    res = raw.get("resolution")
    if isinstance(res, dict):
        steps = res.get("steps")
        if isinstance(steps, list):
            out = []
            for s in steps:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
            return out
    # Fallback: other possible shapes
    for k in ("resolution_steps", "steps"):
        steps = raw.get(k)
        if isinstance(steps, list):
            out = []
            for s in steps:
                if isinstance(s, str) and s.strip():
                    out.append(s.strip())
            return out
    return []


def extract_confirmation(raw: Dict[str, Any]) -> tuple[Optional[bool], Optional[Dict[str, Any]]]:
    """Extract confirmation status and evidence from raw_response_json."""
    c = raw.get("confirmation")
    if isinstance(c, dict):
        confirmed = c.get("confirmed")
        evidence = c.get("evidence")
        return (bool(confirmed) if confirmed is not None else None, evidence if isinstance(evidence, dict) else None)
    return (None, None)


def build_ticket_cache_artifact(
    ticket_id: str,
    raw_response_json: Dict[str, Any],
    conversation_json: Optional[Dict[str, Any]] = None,
    extra_meta: Optional[Dict[str, Any]] = None
) -> TicketCacheArtifact:
    """
    Build a TicketCacheArtifact from raw ticket judgment data.
    
    Enhanced version with PII redaction and optional conversation extraction.
    
    Args:
        ticket_id: Zendesk ticket ID (string)
        raw_response_json: Raw JSON from ticket_judgements.raw_response_json
        conversation_json: Optional conversation JSON from tickets_detail.conversation_json
        extra_meta: Optional additional metadata (e.g., machine_model_ids)
        
    Returns:
        TicketCacheArtifact ready for RAG ingestion
        
    Raises:
        ValueError: If required fields are missing or invalid
    """
    # Import redaction helpers (lazy import)
    try:
        from backend.rag.ticket_redaction import (
            redact_pii,
            extract_technician_notes,
            extract_symptoms,
            extract_parts_used
        )
    except ImportError:
        # Fallback: minimal redaction if module not available
        def redact_pii(text: str) -> str:
            import re
            if not text:
                return text
            result = text
            result = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', result)
            result = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', result)
            return result
        
        def extract_technician_notes(conversation_json: Optional[dict], max_length: int = 1500) -> Optional[str]:
            return None
        
        def extract_symptoms(conversation_json: Optional[dict], raw_response_json: Optional[dict], max_length: int = 1000) -> Optional[str]:
            return None
        
        def extract_parts_used(conversation_json: Optional[dict], max_length: int = 500) -> Optional[str]:
            return None
    
    if extra_meta is None:
        extra_meta = {}
    
    # Extract fields from raw_response_json
    outcome = raw_response_json.get("outcome", "unclear")
    problem = extract_problem(raw_response_json)
    steps = extract_steps(raw_response_json)
    confirmed, confirmation_evidence = extract_confirmation(raw_response_json)
    confidence = raw_response_json.get("confidence", 0.0)
    cache_eligible = raw_response_json.get("cache_eligible", 0)
    rationale = raw_response_json.get("rationale", "")
    blockers = raw_response_json.get("blockers", [])
    
    # Extract root cause if available
    root_cause = None
    if isinstance(raw_response_json.get("root_cause"), str):
        root_cause = raw_response_json.get("root_cause").strip()
    elif isinstance(raw_response_json.get("root_cause"), dict):
        root_cause = raw_response_json.get("root_cause", {}).get("text", "").strip()
    
    # Extract optional sections from conversation_json
    symptoms = extract_symptoms(conversation_json, raw_response_json)
    technician_notes = extract_technician_notes(conversation_json)
    parts_used = extract_parts_used(conversation_json)
    
    # Build enhanced text template (sections only included if non-empty)
    text_parts = []
    
    if problem:
        text_parts.append(f"Problem: {problem}")
    
    if symptoms:
        text_parts.append(f"\nError/Symptoms: {symptoms}")
    
    if root_cause:
        text_parts.append(f"\nRoot Cause: {root_cause}")
    
    if steps:
        text_parts.append("\nResolution Steps:")
        for i, step in enumerate(steps, 1):
            text_parts.append(f"{i}. {step}")
    
    if parts_used:
        text_parts.append(f"\nParts Used: {parts_used}")
    
    if technician_notes:
        text_parts.append(f"\nTechnician Notes: {technician_notes}")
    
    if outcome and outcome != "unclear":
        text_parts.append(f"\nOutcome: {outcome}")
    
    # Ensure we have some text
    text = "\n".join(text_parts) if text_parts else f"Ticket {ticket_id}: No problem or resolution steps available."
    
    # Apply PII redaction BEFORE returning
    text = redact_pii(text)
    
    # Build metadata dict
    from datetime import datetime, timezone
    
    metadata = {
        "document_id": f"ticket:{ticket_id}",
        "file_name": f"ticket_{ticket_id}.md",
        "content_type": "ticket_cache",
        "source": "zendesk_ticket",
        "ticket_id": ticket_id,
        "outcome": outcome,
        "confidence": float(confidence) if confidence is not None else 0.0,
        "cache_eligible": int(cache_eligible) if cache_eligible is not None else 0,
        "confirmed": bool(confirmed) if confirmed is not None else False,
        "rationale": rationale or "",
        "blockers": blockers if isinstance(blockers, list) else [],
        # Machine model fields (default to empty lists unless provided)
        "machine_model_ids": extra_meta.get("machine_model_ids", []),
        "machine_model_names": extra_meta.get("machine_model_names", []),
        "machine_models": extra_meta.get("machine_model_names", []),
        "machine_model": extra_meta.get("machine_model_names", []),
        # New metadata fields
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "judgement_version": raw_response_json.get("prompt_version") or extra_meta.get("prompt_version") or None,
    }
    
    # Add optional metadata from extra_meta
    for key in ("created_at", "updated_at", "resolution_mode", "onsite_required"):
        if key in extra_meta:
            metadata[key] = extra_meta[key]
    
    # Add confirmation evidence if available
    if confirmation_evidence:
        metadata["confirmation_evidence"] = confirmation_evidence
    
    # Add any other additional metadata from extra_meta (excluding already handled keys)
    excluded_keys = {"machine_model_ids", "machine_model_names", "machine_model", "machine_models", "prompt_version", "created_at", "updated_at", "resolution_mode", "onsite_required"}
    for key in extra_meta:
        if key not in excluded_keys:
            metadata[key] = extra_meta[key]
    
    artifact = TicketCacheArtifact(
        id=f"ticket:{ticket_id}",
        text=text,
        metadata=metadata
    )
    
    return artifact


def validate_ticket_cache_artifact(artifact: TicketCacheArtifact) -> None:
    """
    Validate a TicketCacheArtifact raises ValueError if invalid.
    
    Args:
        artifact: TicketCacheArtifact to validate
        
    Raises:
        ValueError: If artifact is invalid
    """
    # Pydantic validation already handles basic structure
    # Additional business logic checks:
    
    metadata = artifact.metadata
    
    # Check document_id matches id
    if metadata.get("document_id") != artifact.id:
        raise ValueError(f"metadata.document_id ({metadata.get('document_id')}) must match id ({artifact.id})")
    
    # Check confidence is in valid range
    confidence = metadata.get("confidence", 0.0)
    if not isinstance(confidence, (int, float)) or confidence < 0.0 or confidence > 1.0:
        raise ValueError(f"confidence must be float between 0.0 and 1.0, got: {confidence}")
    
    # Check cache_eligible is 0 or 1
    cache_eligible = metadata.get("cache_eligible", 0)
    if cache_eligible not in (0, 1):
        raise ValueError(f"cache_eligible must be 0 or 1, got: {cache_eligible}")
    
    # Check machine_model fields are lists
    for key in ("machine_model_ids", "machine_model_names", "machine_models", "machine_model"):
        value = metadata.get(key)
        if value is not None and not isinstance(value, list):
            raise ValueError(f"{key} must be a list, got: {type(value)}")
