#!/usr/bin/env python3
"""
Verify parity between SQLite and Postgres ticket databases.

Compares row counts and samples random tickets to ensure migration was successful.
Handles timestamp differences with tolerance, excludes test tickets by default (matching migration behavior),
and provides detailed mismatch reporting.

Usage:
    python -m backend.scripts.verify_tickets_parity --sqlite-path Scraper/data/tickets.db --sample 50
    
Options:
    --ignore-timestamps: Ignore timestamp differences completely
    --timestamp-tolerance-seconds: Tolerance for timestamp comparisons (default: 1.0s)
    --include-test-tickets: Include test tickets (TEST*, DEMO*) in counts and sampling
"""

import argparse
import hashlib
import json
import os
import random
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file from backend directory BEFORE importing backend modules
try:
    from dotenv import load_dotenv
    backend_dir = project_root / "backend"
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)  # override=True ensures .env values take precedence
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}", file=sys.stderr)
        print(f"   DATABASE_URL must be set as environment variable", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)
    print("   Install with: pip install python-dotenv", file=sys.stderr)
    print("   Or set DATABASE_URL as environment variable", file=sys.stderr)

# Verify DATABASE_URL is available before proceeding
if not os.getenv("DATABASE_URL"):
    print("❌ ERROR: DATABASE_URL not found after loading .env", file=sys.stderr)
    print("   Ensure backend/.env contains DATABASE_URL=...", file=sys.stderr)
    sys.exit(1)

from sqlalchemy import create_engine, text
from sqlalchemy.pool import NullPool


def _is_test_ticket_id(ticket_id: str) -> bool:
    """Check if ticket_id looks like a test ID (matches migration script logic)."""
    ticket_id_upper = ticket_id.upper()
    return (
        ticket_id_upper.startswith("TEST") or
        ticket_id_upper.startswith("DEMO") or
        (not ticket_id.isdigit() and len(ticket_id) < 10)
    )


def normalize_timestamp(value: Any) -> Optional[float]:
    """
    Normalize timestamp to UTC epoch seconds (float).
    Matches migration script's parse_timestamp logic.
    
    Returns:
        Epoch seconds as float, or None if value is None/invalid
    """
    if value is None:
        return None
    
    # If already a datetime object
    if isinstance(value, datetime):
        dt = value
        # If naive, assume UTC (matching migration script behavior)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt.timestamp()
    
    # If string, parse it
    if isinstance(value, str):
        # Handle ISO format with Z
        ts_str = value.replace('Z', '+00:00')
        try:
            dt = datetime.fromisoformat(ts_str)
            # If naive after parsing, assume UTC
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            else:
                dt = dt.astimezone(timezone.utc)
            return dt.timestamp()
        except (ValueError, AttributeError):
            # Try other formats if ISO fails
            try:
                # Try SQLite datetime format: YYYY-MM-DD HH:MM:SS
                dt = datetime.strptime(value, "%Y-%m-%d %H:%M:%S")
                dt = dt.replace(tzinfo=timezone.utc)
                return dt.timestamp()
            except ValueError:
                return None
    
    return None


def timestamps_equal(a: Any, b: Any, tolerance_seconds: float = 1.0) -> bool:
    """
    Compare two timestamps with tolerance.
    
    Returns:
        True if timestamps are within tolerance, False otherwise
    """
    a_norm = normalize_timestamp(a)
    b_norm = normalize_timestamp(b)
    
    # Both None -> equal
    if a_norm is None and b_norm is None:
        return True
    
    # One None, one not -> not equal
    if a_norm is None or b_norm is None:
        return False
    
    # Compare with tolerance
    return abs(a_norm - b_norm) <= tolerance_seconds


def canonical_json(obj: Any) -> str:
    """Convert object to canonical JSON string for hashing."""
    return json.dumps(obj, sort_keys=True, ensure_ascii=False)


def hash_json(obj: Any) -> str:
    """Compute hash of JSON object."""
    return hashlib.sha256(canonical_json(obj).encode('utf-8')).hexdigest()


def compare_row_counts(
    sqlite_conn: sqlite3.Connection,
    pg_engine,
    table_name: str,
    skip_test: bool = True
) -> Tuple[int, int]:
    """
    Compare row counts between SQLite and Postgres.
    If skip_test=True, excludes test ticket IDs (matching migration behavior).
    """
    cursor = sqlite_conn.cursor()
    
    # SQLite count (excluding test tickets if skip_test)
    if skip_test:
        if table_name == "scrape_runs":
            # scrape_runs doesn't have ticket_id, so no filtering needed
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        else:
            # Filter out test ticket IDs
            cursor.execute(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE ticket_id NOT LIKE 'TEST%' 
                AND ticket_id NOT LIKE 'DEMO%'
                AND (ticket_id GLOB '[0-9]*' OR LENGTH(ticket_id) >= 10)
            """)
    else:
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    
    sqlite_count = cursor.fetchone()[0]
    
    # Postgres count (excluding test tickets if skip_test)
    with pg_engine.connect() as conn:
        if skip_test and table_name != "scrape_runs":
            result = conn.execute(text(f"""
                SELECT COUNT(*) FROM {table_name}
                WHERE ticket_id NOT LIKE 'TEST%'
                AND ticket_id NOT LIKE 'DEMO%'
                AND (ticket_id ~ '^[0-9]+$' OR LENGTH(ticket_id) >= 10)
            """))
        else:
            result = conn.execute(text(f"SELECT COUNT(*) FROM {table_name}"))
        pg_count = result.scalar() or 0
    
    return sqlite_count, pg_count


def sample_ticket_ids(sqlite_conn: sqlite3.Connection, table_name: str, sample_size: int, skip_test: bool = True) -> List[str]:
    """
    Get random sample of ticket IDs from SQLite.
    If skip_test=True, excludes test ticket IDs (matching migration behavior).
    """
    cursor = sqlite_conn.cursor()
    
    # Get all ticket IDs (excluding test tickets if skip_test)
    if table_name == "ticket_machine_model_matches":
        if skip_test:
            cursor.execute("""
                SELECT DISTINCT ticket_id FROM ticket_machine_model_matches
                WHERE ticket_id NOT LIKE 'TEST%' 
                AND ticket_id NOT LIKE 'DEMO%'
                AND (ticket_id GLOB '[0-9]*' OR LENGTH(ticket_id) >= 10)
            """)
        else:
            cursor.execute("SELECT DISTINCT ticket_id FROM ticket_machine_model_matches")
    elif table_name == "scrape_runs":
        cursor.execute("SELECT run_id FROM scrape_runs")
        all_ids = [row[0] for row in cursor.fetchall()]
        if len(all_ids) <= sample_size:
            return all_ids
        return random.sample(all_ids, sample_size)
    else:
        if skip_test:
            cursor.execute(f"""
                SELECT ticket_id FROM {table_name}
                WHERE ticket_id NOT LIKE 'TEST%' 
                AND ticket_id NOT LIKE 'DEMO%'
                AND (ticket_id GLOB '[0-9]*' OR LENGTH(ticket_id) >= 10)
            """)
        else:
            cursor.execute(f"SELECT ticket_id FROM {table_name}")
    
    all_ids = [row[0] for row in cursor.fetchall()]
    
    # Sample randomly
    if len(all_ids) <= sample_size:
        return all_ids
    return random.sample(all_ids, sample_size)


def compare_ticket_row(
    sqlite_conn: sqlite3.Connection,
    pg_engine,
    table_name: str,
    ticket_id: str,
    ignore_timestamps: bool = False,
    timestamp_tolerance_seconds: float = 1.0,
    max_timestamp_mismatches: int = 3
) -> Dict[str, Any]:
    """
    Compare a single ticket row between SQLite and Postgres.
    
    Returns:
        Dict with comparison results
    """
    # Get SQLite row
    cursor = sqlite_conn.cursor()
    
    if table_name == "scrape_runs":
        cursor.execute(f"SELECT * FROM {table_name} WHERE run_id = ?", (ticket_id,))
    else:
        cursor.execute(f"SELECT * FROM {table_name} WHERE ticket_id = ?", (ticket_id,))
    
    sqlite_row = cursor.fetchone()
    if not sqlite_row:
        return {"error": f"Ticket {ticket_id} not found in SQLite"}
    
    # Get Postgres row
    with pg_engine.connect() as conn:
        if table_name == "scrape_runs":
            result = conn.execute(
                text(f"SELECT * FROM {table_name} WHERE run_id = :id"),
                {"id": ticket_id}
            )
        else:
            result = conn.execute(
                text(f"SELECT * FROM {table_name} WHERE ticket_id = :id"),
                {"id": ticket_id}
            )
        pg_row = result.fetchone()
    
    if not pg_row:
        return {"error": f"Ticket {ticket_id} not found in Postgres"}
    
    # Convert to dicts for comparison
    sqlite_dict = dict(sqlite_row)
    pg_dict = dict(pg_row._mapping) if hasattr(pg_row, '_mapping') else dict(pg_row)
    
    # Compare fields
    differences = []
    timestamp_mismatch_count = 0
    
    # Handle JSON fields specially
    json_fields = {
        "tickets_detail": ["conversation_json"],
        "ticket_judgements": ["resolution_steps_json", "evidence_json", "blockers_json", "raw_response_json", "review_reasons_json"],
        "ticket_triage": ["triage_raw_response_json"],
        "ticket_machine_model_assignment": ["machine_model_ids"],
        "scrape_runs": ["summary_json"]
    }
    
    for field in sqlite_dict.keys():
        sqlite_val = sqlite_dict[field]
        pg_val = pg_dict.get(field)
        
        # Handle JSON fields
        if table_name in json_fields and field in json_fields[table_name]:
            # Parse SQLite JSON string
            sqlite_parsed = sqlite_val
            if isinstance(sqlite_val, str):
                try:
                    sqlite_parsed = json.loads(sqlite_val)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Parse Postgres JSON (might already be dict/list or JSONB string)
            pg_parsed = pg_val
            if isinstance(pg_val, str):
                try:
                    pg_parsed = json.loads(pg_val)
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Compare JSON using canonical form (sorted keys)
            sqlite_canonical = canonical_json(sqlite_parsed)
            pg_canonical = canonical_json(pg_parsed)
            
            if sqlite_canonical != pg_canonical:
                differences.append({
                    "field": field,
                    "type": "json_mismatch",
                    "sqlite": sqlite_parsed,
                    "pg": pg_parsed
                })
            continue
        
        # Handle boolean fields (SQLite uses 0/1, Postgres uses true/false)
        if isinstance(sqlite_val, int) and sqlite_val in (0, 1) and isinstance(pg_val, bool):
            if bool(sqlite_val) != pg_val:
                differences.append({
                    "field": field,
                    "type": "value_mismatch",
                    "sqlite": sqlite_val,
                    "pg": pg_val
                })
            continue
        
        # Handle timestamp fields
        if field.endswith("_at"):
            if ignore_timestamps:
                continue  # Skip timestamp comparison
            
            # Use tolerance-based comparison
            if not timestamps_equal(sqlite_val, pg_val, timestamp_tolerance_seconds):
                sqlite_norm = normalize_timestamp(sqlite_val)
                pg_norm = normalize_timestamp(pg_val)
                delta_seconds = abs(sqlite_norm - pg_norm) if (sqlite_norm is not None and pg_norm is not None) else None
                
                # Only count as difference if outside tolerance
                if delta_seconds is not None and delta_seconds > timestamp_tolerance_seconds:
                    timestamp_mismatch_count += 1
                    diff_entry = {
                        "field": field,
                        "type": "timestamp_mismatch",
                        "sqlite_raw": str(sqlite_val),
                        "pg_raw": str(pg_val),
                        "sqlite_normalized": sqlite_norm,
                        "pg_normalized": pg_norm,
                        "delta_seconds": delta_seconds
                    }
                    
                    # Only add detailed mismatch if under limit
                    if len([d for d in differences if d.get("type") == "timestamp_mismatch"]) < max_timestamp_mismatches:
                        differences.append(diff_entry)
                    elif len([d for d in differences if d.get("type") == "timestamp_mismatch"]) == max_timestamp_mismatches:
                        # Add a summary entry
                        differences.append({
                            "field": f"{field} (and more timestamp mismatches)",
                            "type": "timestamp_mismatch_summary",
                            "note": "Additional timestamp mismatches suppressed"
                        })
            continue
        
        # Direct comparison
        if sqlite_val != pg_val:
            differences.append({
                "field": field,
                "type": "value_mismatch",
                "sqlite": sqlite_val,
                "pg": pg_val
            })
    
    return {
        "ticket_id": ticket_id,
        "matches": len(differences) == 0,
        "differences": differences,
        "timestamp_mismatch_count": timestamp_mismatch_count
    }


def verify_table(
    sqlite_conn: sqlite3.Connection,
    pg_engine,
    table_name: str,
    sample_size: int = 10,
    skip_test: bool = True,
    ignore_timestamps: bool = False,
    timestamp_tolerance_seconds: float = 1.0
) -> Dict[str, Any]:
    """Verify a single table."""
    print(f"\nVerifying {table_name}...")
    
    # Compare row counts (excluding test tickets if skip_test)
    sqlite_count, pg_count = compare_row_counts(sqlite_conn, pg_engine, table_name, skip_test=skip_test)
    if skip_test:
        print(f"  Row counts (excluding test): SQLite={sqlite_count}, Postgres={pg_count}")
    else:
        print(f"  Row counts: SQLite={sqlite_count}, Postgres={pg_count}")
    
    if sqlite_count != pg_count:
        return {
            "table": table_name,
            "pass": False,
            "reason": f"Row count mismatch: SQLite={sqlite_count}, Postgres={pg_count}",
            "sqlite_count": sqlite_count,
            "pg_count": pg_count,
            "sample_results": []
        }
    
    # Sample and compare rows
    if sqlite_count == 0:
        print(f"  Table is empty, skipping sample comparison")
        return {
            "table": table_name,
            "pass": True,
            "sqlite_count": 0,
            "pg_count": 0,
            "sample_results": []
        }
    
    sample_ids = sample_ticket_ids(sqlite_conn, table_name, min(sample_size, sqlite_count), skip_test=skip_test)
    print(f"  Sampling {len(sample_ids)} rows...")
    
    sample_results = []
    for ticket_id in sample_ids:
        result = compare_ticket_row(
            sqlite_conn, pg_engine, table_name, ticket_id,
            ignore_timestamps=ignore_timestamps,
            timestamp_tolerance_seconds=timestamp_tolerance_seconds
        )
        sample_results.append(result)
        
        if not result.get("matches", False):
            if "error" in result:
                print(f"    ERROR: {result['error']}")
            else:
                # Count non-timestamp mismatches
                non_timestamp_diffs = [d for d in result.get("differences", []) if d.get("type") != "timestamp_mismatch" and d.get("type") != "timestamp_mismatch_summary"]
                timestamp_diffs = result.get("timestamp_mismatch_count", 0)
                
                if non_timestamp_diffs:
                    print(f"    MISMATCH: {ticket_id} - {len(non_timestamp_diffs)} non-timestamp differences")
                    for diff in non_timestamp_diffs[:3]:  # Show first 3
                        print(f"      - {diff['field']}: {diff['type']}")
                elif timestamp_diffs > 0 and not ignore_timestamps:
                    print(f"    TIMESTAMP MISMATCH: {ticket_id} - {timestamp_diffs} timestamp difference(s) (outside tolerance: {timestamp_tolerance_seconds}s)")
                    # Show first timestamp mismatch details
                    for diff in result.get("differences", []):
                        if diff.get("type") == "timestamp_mismatch":
                            print(f"      - {diff['field']}: delta={diff.get('delta_seconds', 'N/A')}s")
                            print(f"        SQLite: {diff.get('sqlite_raw')} -> {diff.get('sqlite_normalized')}")
                            print(f"        Postgres: {diff.get('pg_raw')} -> {diff.get('pg_normalized')}")
                            break
    
    # Consider it a match if:
    # - No differences, OR
    # - Only timestamp differences that are within tolerance (or ignored)
    all_match = True
    for r in sample_results:
        if r.get("matches", False):
            continue
        diffs = r.get("differences", [])
        # Check if all differences are timestamp-related and within tolerance
        non_timestamp_diffs = [d for d in diffs if d.get("type") not in ("timestamp_mismatch", "timestamp_mismatch_summary")]
        if non_timestamp_diffs:
            all_match = False
            break
        # If only timestamp diffs, check if they're within tolerance
        timestamp_diffs = [d for d in diffs if d.get("type") == "timestamp_mismatch"]
        if timestamp_diffs:
            if ignore_timestamps:
                continue  # Ignore timestamps, so this is OK
            # Check if any are outside tolerance
            if any(d.get("delta_seconds", 999) > timestamp_tolerance_seconds for d in timestamp_diffs):
                all_match = False
                break
    
    return {
        "table": table_name,
        "pass": all_match and sqlite_count == pg_count,
        "sqlite_count": sqlite_count,
        "pg_count": pg_count,
        "sample_results": sample_results,
        "sample_size": len(sample_ids),
        "matches": sum(1 for r in sample_results if r.get("matches", False)),
        "mismatches": sum(1 for r in sample_results if not r.get("matches", False))
    }


def main():
    parser = argparse.ArgumentParser(
        description="Verify parity between SQLite and Postgres ticket databases",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--sqlite-path",
        default="Scraper/data/tickets.db",
        help="Path to SQLite tickets database (default: Scraper/data/tickets.db)"
    )
    
    parser.add_argument(
        "--database-url",
        help="Postgres connection string (default: DATABASE_URL env var)"
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=20,
        help="Number of rows to sample per table (default: 20)"
    )
    
    parser.add_argument(
        "--tables",
        help="Comma-separated list of tables to verify (default: all)"
    )
    
    parser.add_argument(
        "--ignore-timestamps",
        action="store_true",
        help="Ignore timestamp differences in comparison (default: False, uses tolerance)"
    )
    
    parser.add_argument(
        "--timestamp-tolerance-seconds",
        type=float,
        default=1.0,
        help="Tolerance for timestamp comparisons in seconds (default: 1.0)"
    )
    
    parser.add_argument(
        "--include-test-tickets",
        action="store_true",
        help="Include test tickets (TEST*, DEMO*) in counts and sampling (default: False, matches migration behavior)"
    )
    
    args = parser.parse_args()
    
    # Validate SQLite path
    sqlite_path = Path(args.sqlite_path)
    if not sqlite_path.exists():
        print(f"ERROR: SQLite database not found: {sqlite_path}")
        sys.exit(1)
    
    # Get Postgres connection
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("ERROR: DATABASE_URL environment variable or --database-url required")
        sys.exit(1)
    
    # Connect to SQLite
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row
    
    # Connect to Postgres
    pg_engine = create_engine(database_url, poolclass=NullPool, future=True)
    
    # Determine tables to verify
    if args.tables:
        tables_to_verify = [t.strip() for t in args.tables.split(",")]
    else:
        tables_to_verify = [
            "tickets_index",
            "tickets_detail",
            "ticket_summaries",
            "ticket_judgements",
            "ticket_triage",
            "ticket_manual_reviews",
            "ticket_machine_model_matches",
            "ticket_machine_model_assignment",
            "scrape_runs"
        ]
    
    print("=" * 70)
    print("TICKET PARITY VERIFICATION: SQLite vs Postgres")
    print("=" * 70)
    print(f"SQLite DB: {sqlite_path}")
    print(f"Postgres: {database_url.split('@')[1] if '@' in database_url else '***'}")
    print(f"Sample size: {args.sample} rows per table")
    if args.ignore_timestamps:
        print(f"Timestamp comparison: IGNORED")
    else:
        print(f"Timestamp tolerance: {args.timestamp_tolerance_seconds}s")
    if not args.include_test_tickets:
        print(f"Test tickets: EXCLUDED (matching migration behavior)")
    print("=" * 70)
    
    # Verify each table
    results = {}
    for table_name in tables_to_verify:
        try:
            result = verify_table(
                sqlite_conn, pg_engine, table_name,
                sample_size=args.sample,
                skip_test=not args.include_test_tickets,
                ignore_timestamps=args.ignore_timestamps,
                timestamp_tolerance_seconds=args.timestamp_tolerance_seconds
            )
            results[table_name] = result
        except Exception as e:
            print(f"\nERROR verifying {table_name}: {e}")
            import traceback
            traceback.print_exc()
            results[table_name] = {
                "table": table_name,
                "pass": False,
                "reason": f"Error: {str(e)}"
            }
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_pass = True
    for table_name, result in results.items():
        status = "PASS" if result.get("pass", False) else "FAIL"
        if not result.get("pass", False):
            all_pass = False
        
        print(f"{table_name}: {status}")
        if "sqlite_count" in result:
            print(f"  Counts: SQLite={result['sqlite_count']}, Postgres={result['pg_count']}")
        if "matches" in result:
            print(f"  Sample: {result['matches']}/{result['sample_size']} matched")
        if "reason" in result:
            print(f"  Reason: {result['reason']}")
    
    print("=" * 70)
    if all_pass:
        print("OVERALL: PASS ✓")
        print("\nAll tables match between SQLite and Postgres!")
        sys.exit(0)
    else:
        print("OVERALL: FAIL ✗")
        print("\nSome tables have mismatches. Review differences above.")
        sys.exit(1)
    
    sqlite_conn.close()
    pg_engine.dispose()


if __name__ == "__main__":
    main()
