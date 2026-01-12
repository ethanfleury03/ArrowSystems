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
from pathlib import Path
from typing import Any, Dict, List, Optional


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
    extra_meta: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Build a ticket cache artifact dict (matches TicketCacheArtifact schema).
    
    This is a standalone version that doesn't require importing from backend.
    
    Args:
        ticket_id: Zendesk ticket ID (string)
        raw_response_json: Raw JSON from ticket_judgements.raw_response_json
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
    
    # Build deterministic text template
    text_parts = []
    
    if problem:
        text_parts.append(f"Problem: {problem}")
    
    if steps:
        text_parts.append("\nResolution Steps:")
        for i, step in enumerate(steps, 1):
            text_parts.append(f"{i}. {step}")
    
    if outcome and outcome != "unclear":
        text_parts.append(f"\nOutcome: {outcome}")
    
    if rationale:
        text_parts.append(f"\nRationale: {rationale}")
    
    if blockers:
        blockers_str = ", ".join(str(b) for b in blockers if b)
        if blockers_str:
            text_parts.append(f"\nBlockers: {blockers_str}")
    
    # Ensure we have some text
    text = "\n".join(text_parts) if text_parts else f"Ticket {ticket_id}: No problem or resolution steps available."
    
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
    }
    
    # Add confirmation evidence if available
    if confirmation_evidence:
        metadata["confirmation_evidence"] = confirmation_evidence
    
    # Add any additional metadata from extra_meta
    for key in extra_meta:
        if key not in ("machine_model_ids", "machine_model_names", "machine_model", "machine_models"):
            metadata[key] = extra_meta[key]
    
    return {
        "id": f"ticket:{ticket_id}",
        "text": text,
        "metadata": metadata
    }


# SQL query for effective cache-eligible tickets
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
    m.reviewer
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
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
        db_path: Path to tickets.db SQLite database
        output_path: Path to output JSONL file
        ticket_id: Optional specific ticket ID to export
        limit: Optional limit on number of tickets to export
        force: If True, overwrite existing output file
        
    Returns:
        Dict with counts and status
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")
    
    output_path = Path(output_path)
    if output_path.exists() and not force:
        raise FileExistsError(f"Output file exists: {output_path}. Use --force to overwrite.")
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Connect to database
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
    
    print(f"Found {len(rows)} cache-eligible tickets")
    
    # Process tickets and write JSONL
    artifacts = []
    failed = 0
    errors = []
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for row in rows:
            ticket_id_str = row['ticket_id']
            
            try:
                # Parse raw_response_json
                raw_json_str = row['raw_response_json']
                if not raw_json_str:
                    raise ValueError("raw_response_json is empty")
                
                raw_json = json.loads(raw_json_str)
                if not isinstance(raw_json, dict):
                    raise ValueError(f"raw_response_json is not a dict, got: {type(raw_json)}")
                
                # Build artifact
                artifact = build_ticket_cache_artifact(
                    ticket_id=ticket_id_str,
                    raw_response_json=raw_json,
                    extra_meta={}  # No extra metadata for now
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
