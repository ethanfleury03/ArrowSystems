#!/usr/bin/env python3
"""
Export cache-eligible ticket judgments as RAG-ready artifacts.

Queries tickets.db for cache-eligible tickets and exports them as JSONL
for ingestion into the RAG system.

Usage:
    python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl
    python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --limit 10
    python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --ticket-id 12345
"""

import argparse
import json
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from ticket_store import get_ticket_store
from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool

# Import redaction and extraction helpers
# Try to import from backend if available, otherwise define inline
try:
    from backend.rag.ticket_redaction import (
        redact_pii,
        extract_technician_notes,
        extract_symptoms,
        extract_parts_used
    )
except ImportError:
    # Fallback: define minimal versions if backend not available
    def redact_pii(text: str) -> str:
        import re
        if not text:
            return text
        result = text
        result = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL]', result)
        result = re.sub(r'\b(?:\+?1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b', '[PHONE]', result)
        result = re.sub(r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b', '[IP_ADDRESS]', result)
        # Serial masking (simplified)
        def mask_serial(m):
            serial = m.group(0)
            alphanumeric = re.sub(r'[^a-zA-Z0-9]', '', serial)
            last_4 = alphanumeric[-4:] if len(alphanumeric) >= 4 else alphanumeric
            prefix = m.group(1) if m.lastindex >= 1 else "SN"
            return f"{prefix}-[REDACTED]-{last_4}"
        result = re.sub(r'\b(SN|S/N|Serial\s*Number|Serial)[\s:]*([A-Z0-9]{4,})', mask_serial, result, flags=re.IGNORECASE)
        return result
    
    def extract_technician_notes(conversation_json: Optional[dict], max_length: int = 1500) -> Optional[str]:
        if not conversation_json:
            return None
        messages = conversation_json.get("messages", [])
        if not isinstance(messages, list):
            return None
        agent_messages = [msg for msg in messages if isinstance(msg, dict) and msg.get("role") in ("agent", "technician")]
        if not agent_messages:
            return None
        try:
            agent_messages.sort(key=lambda m: m.get("created_at", ""), reverse=True)
        except Exception:
            pass
        notes_parts = [msg.get("text", "").strip() for msg in agent_messages[:5] if msg.get("text", "").strip()]
        if not notes_parts:
            return None
        notes = "\n".join(notes_parts)
        return notes[:max_length] + "..." if len(notes) > max_length else notes
    
    def extract_symptoms(conversation_json: Optional[dict], raw_response_json: Optional[dict], max_length: int = 1000) -> Optional[str]:
        if raw_response_json:
            for key in ("error", "errors", "symptom", "symptoms"):
                value = raw_response_json.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()[:max_length]
        return None
    
    def extract_parts_used(conversation_json: Optional[dict], max_length: int = 500) -> Optional[str]:
        return None  # Simplified fallback


# Copy extraction functions from verify_raw_response_schema.py to avoid cross-repo imports
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
) -> Dict[str, Any]:
    """
    Build a ticket cache artifact dict (matches TicketCacheArtifact schema).
    
    Enhanced version with PII redaction and optional conversation extraction.
    
    Args:
        ticket_id: Zendesk ticket ID (string)
        raw_response_json: Raw JSON from ticket_judgements.raw_response_json
        conversation_json: Optional conversation JSON from tickets_detail.conversation_json
        extra_meta: Optional additional metadata
        
    Returns:
        Dict matching TicketCacheArtifact schema
    """
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
    
    return {
        "id": f"ticket:{ticket_id}",
        "text": text,
        "metadata": metadata
    }


# SQL query for effective cache-eligible tickets
# Includes LEFT JOIN to tickets_detail for conversation_json (optional but default ON)
EFFECTIVE_CACHE_ELIGIBLE_SQL = """
SELECT 
    j.ticket_id,
    j.raw_response_json,
    j.cache_eligible,
    j.confidence,
    j.model,
    j.prompt_version,
    j.judged_at,
    m.manual_status,
    m.reviewer,
    d.conversation_json
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
LEFT JOIN tickets_detail d ON j.ticket_id = d.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = 1))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = 1
ORDER BY j.ticket_id
"""


def export_cache_artifacts(
    db_path: str,
    output_path: str,
    ticket_id: Optional[str] = None,
    limit: Optional[int] = None,
    force: bool = False
) -> Dict[str, Any]:
    """
    Export cache-eligible tickets as JSONL artifacts.
    
    Args:
        db_path: Path to tickets.db SQLite database (ignored if using Postgres)
        output_path: Path to output JSONL file
        ticket_id: Optional specific ticket ID to export
        limit: Optional limit on number of tickets to export
        force: If True, overwrite existing output file
        
    Returns:
        Dict with counts and status
    """
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output file exists: {output_path}. Use --force to overwrite.")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Determine backend and execute query
    backend = os.getenv("TICKETS_STORAGE_BACKEND", "sqlite").lower()
    
    if backend == "postgres":
        # Use Postgres via SQLAlchemy
        database_url = os.getenv("DATABASE_URL")
        if not database_url:
            raise ValueError("DATABASE_URL environment variable is required for Postgres backend")
        
        engine = create_engine(database_url, poolclass=NullPool, future=True)
        
        # Build query (Postgres uses :param syntax)
        if ticket_id:
            query = EFFECTIVE_CACHE_ELIGIBLE_SQL.replace(
                "ORDER BY j.ticket_id",
                "AND j.ticket_id = :ticket_id ORDER BY j.ticket_id"
            )
            if limit:
                query += f" LIMIT {limit}"
            with engine.connect() as conn:
                result = conn.execute(text(query), {"ticket_id": ticket_id})
                rows = [dict(row._mapping) for row in result.fetchall()]
        else:
            query = EFFECTIVE_CACHE_ELIGIBLE_SQL
            if limit:
                query += f" LIMIT {limit}"
            with engine.connect() as conn:
                result = conn.execute(text(query))
                rows = [dict(row._mapping) for row in result.fetchall()]
    else:
        # Use SQLite
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Build query
        if ticket_id:
            query = EFFECTIVE_CACHE_ELIGIBLE_SQL.replace(
                "ORDER BY j.ticket_id",
                "AND j.ticket_id = ? ORDER BY j.ticket_id"
            )
            cursor.execute(query, (ticket_id,))
        else:
            query = EFFECTIVE_CACHE_ELIGIBLE_SQL
            if limit:
                query += f" LIMIT {limit}"
            cursor.execute(query)
        
        rows = cursor.fetchall()
        conn.close()
        
        # Convert sqlite3.Row to dict for consistency
        rows = [dict(row) for row in rows]
    
    print(f"Found {len(rows)} cache-eligible tickets")
    
    # Process tickets and write JSONL
    artifacts = []
    failed = 0
    errors = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            ticket_id_str = row['ticket_id']
            
            try:
                # Parse raw_response_json (may be dict from Postgres JSONB or string from SQLite)
                raw_json = row['raw_response_json']
                if isinstance(raw_json, str):
                    if not raw_json:
                        raise ValueError("raw_response_json is empty")
                    raw_json = json.loads(raw_json)
                elif not isinstance(raw_json, dict):
                    raise ValueError(f"raw_response_json is not a dict, got: {type(raw_json)}")
                
                # Parse conversation_json (optional, may be None)
                conversation_json = None
                if 'conversation_json' in row and row['conversation_json']:
                    conv_json = row['conversation_json']
                    if isinstance(conv_json, str):
                        try:
                            conversation_json = json.loads(conv_json)
                        except (json.JSONDecodeError, TypeError):
                            conversation_json = None
                    elif isinstance(conv_json, dict):
                        conversation_json = conv_json
                
                # Build extra_meta from row data
                extra_meta = {
                    "prompt_version": row.get("prompt_version"),
                }
                
                # Build artifact with conversation_json
                artifact = build_ticket_cache_artifact(
                    ticket_id=ticket_id_str,
                    raw_response_json=raw_json,
                    conversation_json=conversation_json,
                    extra_meta=extra_meta
                )
                
                # Write JSONL line
                f.write(json.dumps(artifact, ensure_ascii=False) + '\n')
                artifacts.append(artifact)
                
            except Exception as e:
                failed += 1
                error_msg = f"Failed to process ticket {ticket_id_str}: {e}"
                errors.append(error_msg)
                print(f"ERROR: {error_msg}", file=sys.stderr)
    
    return {
        "total": len(rows),
        "exported": len(artifacts),
        "failed": failed,
        "errors": errors,
        "output_path": str(output_path)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export cache-eligible tickets as RAG-ready artifacts",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export all cache-eligible tickets
  python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl
  
  # Export specific ticket
  python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --ticket-id 12345
  
  # Export first 10 tickets
  python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --limit 10
  
  # Overwrite existing file
  python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl --force
        """
    )
    
    parser.add_argument(
        "--db",
        default="data/tickets.db",
        help="Path to tickets.db SQLite database (default: data/tickets.db)"
    )
    
    parser.add_argument(
        "--out",
        "--jsonl-out",
        dest="output",
        default="out/cache_artifacts.jsonl",
        help="Path to output JSONL file (default: out/cache_artifacts.jsonl)"
    )
    
    parser.add_argument(
        "--ticket-id",
        help="Export specific ticket ID only"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of tickets to export"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output file"
    )
    
    args = parser.parse_args()
    
    try:
        result = export_cache_artifacts(
            db_path=args.db,
            output_path=args.output,
            ticket_id=args.ticket_id,
            limit=args.limit,
            force=args.force
        )
        
        print("\n" + "=" * 70)
        print("EXPORT SUMMARY")
        print("=" * 70)
        print(f"Total tickets found: {result['total']}")
        print(f"Successfully exported: {result['exported']}")
        print(f"Failed: {result['failed']}")
        print(f"Output file: {result['output_path']}")
        
        if result['errors']:
            print(f"\nErrors ({len(result['errors'])}):")
            for error in result['errors'][:10]:
                print(f"  - {error}")
            if len(result['errors']) > 10:
                print(f"  ... and {len(result['errors']) - 10} more errors")
        
        print("=" * 70)
        
        # Exit with error code if failures occurred
        if result['failed'] > 0:
            sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
