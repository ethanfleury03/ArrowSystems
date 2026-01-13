#!/usr/bin/env python3
"""
One-time backfill script: Auto-assign machine models to tickets.

Scans ticket conversation text for machine model names/aliases and assigns
matched models to tickets. Supports dry-run mode for safety.

Usage:
    python scripts/backfill_ticket_machine_models.py --dry-run
    python scripts/backfill_ticket_machine_models.py --write --limit 100
    python scripts/backfill_ticket_machine_models.py --write --ticket-id 12345
"""

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import db
from utils.machine_models_loader import load_machine_models, MachineModel
from utils.machine_model_matcher import find_matches, determine_assignment


def build_conversation_blob(ticket_id: str, conn) -> str:
    """
    Build a single conversation blob string from ticket data.
    
    Includes:
    - Subject (from tickets_index)
    - Description (from conversation_json.request.description)
    - All messages (from conversation_json.messages, chronological)
    
    Args:
        ticket_id: Ticket ID
        conn: SQLite connection
        
    Returns:
        Combined conversation text
    """
    parts = []
    
    # Get subject from tickets_index
    cursor = conn.cursor()
    cursor.execute("SELECT subject FROM tickets_index WHERE ticket_id = ?", (ticket_id,))
    row = cursor.fetchone()
    if row and row["subject"]:
        parts.append(f"Subject: {row['subject']}")
    
    # Get conversation from tickets_detail
    conversation = db.get_ticket_detail_json(conn, ticket_id)
    if conversation:
        # Add request description
        request = conversation.get("request", {})
        if isinstance(request, dict):
            description = request.get("description")
            if description:
                parts.append(f"Description: {description}")
        
        # Add all messages (sorted by created_at)
        messages = conversation.get("messages", [])
        if isinstance(messages, list):
            sorted_messages = sorted(
                messages,
                key=lambda m: m.get("created_at", "")
            )
            for msg in sorted_messages:
                text = msg.get("text", "")
                if text:
                    role = msg.get("role", "unknown")
                    parts.append(f"[{role}]: {text}")
    
    return "\n".join(parts)


def init_backfill_tables(conn) -> None:
    """
    Initialize database tables for machine model assignments.
    
    Creates:
    - ticket_machine_model_matches: One row per match
    - ticket_machine_model_assignment: One row per ticket (summary)
    """
    cursor = conn.cursor()
    
    # Create matches table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_machine_model_matches (
            ticket_id TEXT NOT NULL,
            machine_model_id INTEGER NOT NULL,
            machine_model_name TEXT NOT NULL,
            match_source TEXT NOT NULL,
            score INTEGER NOT NULL,
            evidence_snippet TEXT,
            created_at TEXT NOT NULL,
            PRIMARY KEY (ticket_id, machine_model_id, match_source)
        )
    """)
    
    # Create assignment summary table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ticket_machine_model_assignment (
            ticket_id TEXT PRIMARY KEY,
            machine_model_ids TEXT NOT NULL,
            status TEXT NOT NULL,
            confidence REAL NOT NULL,
            method TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    """)
    
    # Create indexes
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticket_machine_model_matches_ticket_id
        ON ticket_machine_model_matches(ticket_id)
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_ticket_machine_model_assignment_status
        ON ticket_machine_model_assignment(status)
    """)
    
    conn.commit()


def process_ticket(
    ticket_id: str,
    conn,
    models: List[MachineModel],
    min_score: int,
    ambiguous_threshold: float
) -> Dict[str, Any]:
    """
    Process a single ticket and find machine model matches.
    
    Args:
        ticket_id: Ticket ID
        conn: SQLite connection
        models: List of MachineModel objects
        min_score: Minimum match score threshold
        ambiguous_threshold: Ambiguity threshold (0.0-1.0)
        
    Returns:
        Result dict with matches and assignment
    """
    # Build conversation blob
    conversation_blob = build_conversation_blob(ticket_id, conn)
    
    if not conversation_blob.strip():
        return {
            "ticket_id": ticket_id,
            "matches": [],
            "assignment": {
                "machine_model_ids": [],
                "status": "unassigned",
                "confidence": 0.0,
                "method": "regex_match_v1"
            },
            "conversation_length": 0
        }
    
    # Find matches
    matches = find_matches(conversation_blob, models, min_score=min_score)
    
    # Determine assignment
    assignment = determine_assignment(matches, ambiguous_threshold=ambiguous_threshold)
    
    return {
        "ticket_id": ticket_id,
        "matches": matches,
        "assignment": assignment,
        "conversation_length": len(conversation_blob)
    }


def save_results(
    results: List[Dict[str, Any]],
    conn,
    dry_run: bool
) -> None:
    """
    Save results to database and export files.
    
    Args:
        results: List of result dicts from process_ticket
        conn: SQLite connection
        dry_run: If True, skip database writes
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    
    if not dry_run:
        cursor = conn.cursor()
        
        for result in results:
            ticket_id = result["ticket_id"]
            matches = result["matches"]
            assignment = result["assignment"]
            
            # Insert/update matches
            for match in matches:
                cursor.execute("""
                    INSERT INTO ticket_machine_model_matches (
                        ticket_id, machine_model_id, machine_model_name,
                        match_source, score, evidence_snippet, created_at
                    )
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(ticket_id, machine_model_id, match_source) DO UPDATE SET
                        score = excluded.score,
                        evidence_snippet = excluded.evidence_snippet,
                        created_at = excluded.created_at
                """, (
                    ticket_id,
                    match["model_id"],
                    match["model_name"],
                    match["match_source"],
                    match["score"],
                    match["evidence_snippet"],
                    timestamp
                ))
            
            # Insert/update assignment summary
            machine_model_ids_json = json.dumps(assignment["machine_model_ids"])
            cursor.execute("""
                INSERT INTO ticket_machine_model_assignment (
                    ticket_id, machine_model_ids, status, confidence, method, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticket_id) DO UPDATE SET
                    machine_model_ids = excluded.machine_model_ids,
                    status = excluded.status,
                    confidence = excluded.confidence,
                    method = excluded.method,
                    updated_at = excluded.updated_at
            """, (
                ticket_id,
                machine_model_ids_json,
                assignment["status"],
                assignment["confidence"],
                assignment["method"],
                timestamp
            ))
        
        conn.commit()


def export_jsonl(results: List[Dict[str, Any]], output_path: str) -> None:
    """Export results to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')


def export_csv(results: List[Dict[str, Any]], output_path: str) -> None:
    """Export results to CSV file for manual review."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticket_id",
            "status",
            "confidence",
            "machine_model_ids",
            "machine_model_names",
            "match_count",
            "top_match_score",
            "top_match_source",
            "evidence_snippet"
        ])
        
        for result in results:
            ticket_id = result["ticket_id"]
            assignment = result["assignment"]
            matches = result["matches"]
            
            machine_model_ids = assignment["machine_model_ids"]
            machine_model_names = [m["model_name"] for m in matches[:len(machine_model_ids)]]
            
            top_match = matches[0] if matches else None
            
            writer.writerow([
                ticket_id,
                assignment["status"],
                f"{assignment['confidence']:.2f}",
                json.dumps(machine_model_ids),
                ", ".join(machine_model_names),
                len(matches),
                top_match["score"] if top_match else 0,
                top_match["match_source"] if top_match else "",
                (top_match["evidence_snippet"][:200] if top_match and top_match.get("evidence_snippet") else "")
            ])


def print_summary(results: List[Dict[str, Any]]) -> None:
    """Print summary statistics."""
    total = len(results)
    assigned = sum(1 for r in results if r["assignment"]["status"] == "assigned")
    ambiguous = sum(1 for r in results if r["assignment"]["status"] == "ambiguous")
    unassigned = sum(1 for r in results if r["assignment"]["status"] == "unassigned")
    
    print("\n" + "="*60)
    print("BACKFILL SUMMARY")
    print("="*60)
    print(f"Total tickets processed: {total}")
    print(f"  Assigned (single match): {assigned}")
    print(f"  Ambiguous (multiple matches): {ambiguous}")
    print(f"  Unassigned (no matches): {unassigned}")
    print("="*60)
    
    # Top 20 most ambiguous tickets
    ambiguous_tickets = [
        r for r in results
        if r["assignment"]["status"] == "ambiguous"
    ]
    ambiguous_tickets.sort(key=lambda x: len(x["matches"]), reverse=True)
    
    if ambiguous_tickets:
        print("\nTop 20 Most Ambiguous Tickets:")
        for i, result in enumerate(ambiguous_tickets[:20], 1):
            matches = result["matches"]
            model_names = [m["model_name"] for m in matches]
            print(f"  {i}. Ticket {result['ticket_id']}: {len(matches)} matches - {', '.join(model_names)}")
    
    # Top 20 most frequently matched models
    model_counts: Dict[str, int] = {}
    for result in results:
        for match in result["matches"]:
            model_name = match["model_name"]
            model_counts[model_name] = model_counts.get(model_name, 0) + 1
    
    if model_counts:
        sorted_models = sorted(model_counts.items(), key=lambda x: x[1], reverse=True)
        print("\nTop 20 Most Frequently Matched Models:")
        for i, (model_name, count) in enumerate(sorted_models[:20], 1):
            print(f"  {i}. {model_name}: {count} tickets")


def main():
    parser = argparse.ArgumentParser(
        description="Backfill machine model assignments for tickets"
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to tickets.db (default: Scraper/data/tickets.db)"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="out",
        help="Output directory for JSONL/CSV files (default: out/)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Dry-run mode (default: True, use --write to enable writes)"
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Enable database writes (overrides --dry-run)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of tickets to process"
    )
    parser.add_argument(
        "--ticket-id",
        type=str,
        default=None,
        help="Process specific ticket ID only"
    )
    parser.add_argument(
        "--min-score",
        type=int,
        default=50,
        help="Minimum match score threshold (default: 50)"
    )
    parser.add_argument(
        "--ambiguous-threshold",
        type=float,
        default=0.1,
        help="Ambiguity threshold (0.0-1.0, default: 0.1)"
    )
    parser.add_argument(
        "--models-source",
        type=str,
        choices=["cloudsql", "json"],
        default="cloudsql",
        help="Machine models source (default: cloudsql)"
    )
    parser.add_argument(
        "--models-json",
        type=str,
        default=None,
        help="Path to machine_models.json (required if models-source=json)"
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run self-test mode (test matcher on sample strings)"
    )
    
    args = parser.parse_args()
    
    # Determine dry-run mode
    dry_run = args.dry_run and not args.write
    
    if dry_run:
        print("[DRY-RUN MODE] No database writes will be performed")
    else:
        print("[WRITE MODE] Database will be updated")
    
    # Self-test mode
    if args.self_test:
        print("Running self-test...")
        from utils.machine_model_matcher import find_matches, determine_assignment
        from utils.machine_models_loader import MachineModel
        
        models = [
            MachineModel(1, "DuraFlex"),
            MachineModel(2, "EZCut 330"),
            MachineModel(3, "2800"),
        ]
        
        test_cases = [
            "I have a DuraFlex machine that's not working",
            "The Dura Flex printer is broken",
            "My EZCut 330 needs repair",
            "Model 2800 is having issues",
            "I need help with my printer",
        ]
        
        for text in test_cases:
            matches = find_matches(text, models, min_score=50)
            assignment = determine_assignment(matches)
            print(f"Text: '{text}'")
            print(f"  Matches: {[m['model_name'] for m in matches]}")
            print(f"  Assignment: {assignment['status']} (confidence: {assignment['confidence']:.2f})")
            print()
        
        print("Self-test complete!")
        return
    
    # Load machine models
    print("Loading machine models...")
    try:
        models = load_machine_models(
            source=args.models_source,
            database_url=os.getenv("DATABASE_URL"),
            json_path=args.models_json
        )
        print(f"Loaded {len(models)} machine models")
    except Exception as e:
        print(f"[ERROR] Failed to load machine models: {e}")
        sys.exit(1)
    
    # Connect to tickets DB
    if args.db:
        db_path = args.db
    else:
        script_dir = Path(__file__).parent.parent
        db_path = script_dir / "data" / "tickets.db"
    
    if not Path(db_path).exists():
        print(f"[ERROR] Tickets database not found: {db_path}")
        sys.exit(1)
    
    print(f"Connecting to tickets database: {db_path}")
    conn = db.get_connection(str(db_path))
    
    try:
        # Initialize tables
        print("Initializing database tables...")
        init_backfill_tables(conn)
        
        # Get ticket IDs to process
        cursor = conn.cursor()
        if args.ticket_id:
            cursor.execute("SELECT ticket_id FROM tickets_index WHERE ticket_id = ?", (args.ticket_id,))
        else:
            cursor.execute("SELECT ticket_id FROM tickets_index ORDER BY ticket_id")
        
        rows = cursor.fetchall()
        ticket_ids = [row["ticket_id"] for row in rows]
        
        if args.limit:
            ticket_ids = ticket_ids[:args.limit]
        
        print(f"Processing {len(ticket_ids)} tickets...")
        
        # Process tickets
        results = []
        for i, ticket_id in enumerate(ticket_ids, 1):
            if i % 100 == 0:
                print(f"  Processed {i}/{len(ticket_ids)} tickets...")
            
            try:
                result = process_ticket(
                    ticket_id,
                    conn,
                    models,
                    min_score=args.min_score,
                    ambiguous_threshold=args.ambiguous_threshold
                )
                results.append(result)
            except Exception as e:
                print(f"  [WARNING] Error processing ticket {ticket_id}: {e}")
                continue
        
        # Save results
        print("\nSaving results...")
        save_results(results, conn, dry_run=dry_run)
        
        # Export files
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        
        jsonl_path = outdir / "ticket_machine_model_backfill.jsonl"
        csv_path = outdir / "ticket_machine_model_backfill.csv"
        
        export_jsonl(results, jsonl_path)
        export_csv(results, csv_path)
        
        print(f"  Exported JSONL: {jsonl_path}")
        print(f"  Exported CSV: {csv_path}")
        
        # Print summary
        print_summary(results)
        
    finally:
        conn.close()


if __name__ == "__main__":
    main()
