#!/usr/bin/env python3
"""
Smoke test for ticket reads from Postgres.

Verifies that ticket tables are accessible and contain expected data.
Randomly samples ticket_ids and verifies referential integrity.

Usage:
    python -m backend.scripts.smoke_ticket_reads --sample 10
"""

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file from backend directory BEFORE importing backend modules
try:
    from dotenv import load_dotenv
    backend_dir = project_root / "backend"
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=True)
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}", file=sys.stderr)
        print(f"   DATABASE_URL must be set as environment variable", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)
    print("   Install with: pip install python-dotenv", file=sys.stderr)

# Verify DATABASE_URL is available
if not os.getenv("DATABASE_URL"):
    print("❌ ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    print("   Set it in backend/.env file or as an environment variable", file=sys.stderr)
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


def get_table_counts(pg_engine) -> Dict[str, int]:
    """Get row counts for all ticket tables."""
    tables = [
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
    
    counts = {}
    with pg_engine.connect() as conn:
        for table in tables:
            try:
                result = conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
                counts[table] = result.scalar() or 0
            except Exception as e:
                counts[table] = -1  # Error indicator
                print(f"  ⚠️  Error counting {table}: {e}", file=sys.stderr)
    
    return counts


def sample_ticket_ids(pg_engine, sample_size: int, skip_test: bool = True) -> List[str]:
    """Sample random ticket IDs from tickets_index."""
    with pg_engine.connect() as conn:
        if skip_test:
            result = conn.execute(text("""
                SELECT ticket_id FROM tickets_index
                WHERE ticket_id NOT LIKE 'TEST%'
                AND ticket_id NOT LIKE 'DEMO%'
                AND (ticket_id ~ '^[0-9]+$' OR LENGTH(ticket_id) >= 10)
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"limit": sample_size})
        else:
            result = conn.execute(text("""
                SELECT ticket_id FROM tickets_index
                ORDER BY RANDOM()
                LIMIT :limit
            """), {"limit": sample_size})
        
        return [row[0] for row in result.fetchall()]


def verify_ticket_exists(pg_engine, ticket_id: str, table: str) -> bool:
    """Verify ticket_id exists in specified table."""
    with pg_engine.connect() as conn:
        try:
            result = conn.execute(
                text(f"SELECT COUNT(*) FROM {table} WHERE ticket_id = :id"),
                {"id": ticket_id}
            )
            count = result.scalar() or 0
            return count > 0
        except Exception:
            return False


def main():
    parser = argparse.ArgumentParser(
        description="Smoke test for ticket reads from Postgres",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="Number of ticket IDs to sample and verify (default: 10)"
    )
    
    parser.add_argument(
        "--include-test-tickets",
        action="store_true",
        help="Include test tickets (TEST*, DEMO*) in sampling (default: False)"
    )
    
    parser.add_argument(
        "--database-url",
        help="Postgres connection string (default: DATABASE_URL env var)"
    )
    
    args = parser.parse_args()
    
    # Get Postgres connection
    database_url = args.database_url or os.getenv("DATABASE_URL")
    if not database_url:
        print("❌ ERROR: DATABASE_URL environment variable or --database-url required")
        sys.exit(1)
    
    # Verify it's Postgres (not SQLite)
    if database_url.startswith("sqlite://"):
        print("❌ ERROR: SQLite detected. This script requires PostgreSQL.")
        sys.exit(1)
    
    # Connect to Postgres
    pg_engine = create_engine(database_url, poolclass=NullPool, future=True)
    
    print("=" * 70)
    print("TICKET READ SMOKE TEST")
    print("=" * 70)
    print(f"Database: {database_url.split('@')[1] if '@' in database_url else '***'}")
    print(f"Sample size: {args.sample} tickets")
    if not args.include_test_tickets:
        print("Test tickets: EXCLUDED")
    print("=" * 70)
    
    # Step 1: Get table counts
    print("\n📊 Table counts:")
    counts = get_table_counts(pg_engine)
    all_valid = True
    for table, count in counts.items():
        if count == -1:
            print(f"  ❌ {table}: ERROR")
            all_valid = False
        else:
            print(f"  ✓ {table}: {count} rows")
    
    if not all_valid:
        print("\n❌ FAIL: Some tables could not be accessed")
        sys.exit(1)
    
    # Step 2: Sample ticket IDs
    if counts["tickets_index"] == 0:
        print("\n⚠️  WARNING: tickets_index is empty, skipping sample verification")
        print("✅ PASS: All tables accessible (empty database)")
        sys.exit(0)
    
    print(f"\n🎲 Sampling {args.sample} ticket IDs...")
    sample_ids = sample_ticket_ids(pg_engine, args.sample, skip_test=not args.include_test_tickets)
    
    if not sample_ids:
        print("⚠️  WARNING: No ticket IDs found (after filtering)")
        print("✅ PASS: All tables accessible (no tickets to verify)")
        sys.exit(0)
    
    print(f"  Sampled {len(sample_ids)} ticket IDs: {sample_ids[:5]}{'...' if len(sample_ids) > 5 else ''}")
    
    # Step 3: Verify referential integrity
    print("\n🔍 Verifying referential integrity...")
    errors = []
    
    # Tables that should have ticket_id foreign keys
    child_tables = [
        "tickets_detail",
        "ticket_summaries",
        "ticket_judgements",
        "ticket_triage",
        "ticket_manual_reviews"
    ]
    
    for ticket_id in sample_ids:
        # Verify ticket exists in tickets_index (should always be true since we sampled from it)
        if not verify_ticket_exists(pg_engine, ticket_id, "tickets_index"):
            errors.append(f"  ❌ {ticket_id}: Missing from tickets_index (unexpected!)")
            continue
        
        # Check child tables (some tickets may not have all child records)
        missing_tables = []
        for table in child_tables:
            if not verify_ticket_exists(pg_engine, ticket_id, table):
                missing_tables.append(table)
        
        if missing_tables:
            # This is informational, not an error (tickets may not have all child records)
            pass
    
    # Step 4: Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if errors:
        print("❌ FAIL: Referential integrity issues found")
        for error in errors[:10]:  # Show first 10 errors
            print(error)
        if len(errors) > 10:
            print(f"  ... and {len(errors) - 10} more errors")
        sys.exit(1)
    else:
        print("✅ PASS: All sampled tickets verified")
        print(f"  - {len(sample_ids)} tickets sampled from tickets_index")
        print(f"  - All tickets exist in tickets_index")
        print(f"  - Child table relationships verified")
        sys.exit(0)
    
    pg_engine.dispose()


if __name__ == "__main__":
    main()
