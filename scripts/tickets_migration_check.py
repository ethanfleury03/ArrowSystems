#!/usr/bin/env python3
r"""
Repo-root helper script to verify ticket tables exist in Postgres.

This script loads backend/.env, connects to the database, and lists all
ticket-related tables to verify the migration was successful.

Usage:
    python scripts/tickets_migration_check.py

All commands must be run from the repository root.
"""

import os
import sys
from pathlib import Path

# Get project root (where this script lives)
project_root = Path(__file__).parent.parent.resolve()

# Change to project root
os.chdir(project_root)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    backend_dir = project_root / "backend"
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}", file=sys.stderr)
        print(f"   DATABASE_URL must be set as environment variable", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)
    print("   Install with: pip install python-dotenv", file=sys.stderr)
    print("   Or set DATABASE_URL as environment variable", file=sys.stderr)

# Verify DATABASE_URL is set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("❌ ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    print("   Set it in backend/.env file or as an environment variable", file=sys.stderr)
    sys.exit(1)

# Expected ticket tables (from migration 011_ticket_tables_postgres)
EXPECTED_TICKET_TABLES = {
    "tickets_index",
    "tickets_detail",
    "ticket_summaries",
    "ticket_judgements",
    "ticket_triage",
    "ticket_manual_reviews",
    "ticket_machine_model_matches",
    "ticket_machine_model_assignment",
    "scrape_runs",
}

# Connect to database and inspect tables
try:
    from sqlalchemy import create_engine, inspect, text
    
    print("🔍 Connecting to database...", file=sys.stderr)
    engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database(), current_user"))
        row = result.fetchone()
        db_name = row[0]
        db_user = row[1]
        print(f"✅ Connected to: {db_name} (user: {db_user})\n", file=sys.stderr)
    
    inspector = inspect(engine)
    all_tables = inspector.get_table_names()
    
    # Find ticket-related tables
    ticket_tables = sorted([t for t in all_tables if "ticket" in t.lower() or t == "scrape_runs"])
    
    print(f"📊 Found {len(ticket_tables)} ticket-related table(s):\n")
    
    if ticket_tables:
        for table in ticket_tables:
            # Check if it's expected
            status = "✅" if table in EXPECTED_TICKET_TABLES else "⚠️ "
            print(f"  {status} {table}")
        
        print()
        
        # Check for missing tables
        missing = EXPECTED_TICKET_TABLES - set(ticket_tables)
        if missing:
            print(f"❌ Missing {len(missing)} expected table(s):")
            for table in sorted(missing):
                print(f"  - {table}")
            print()
            print("💡 Run migrations: python scripts/db_migrate.py upgrade head")
            sys.exit(1)
        else:
            print("✅ All expected ticket tables are present!")
            sys.exit(0)
    else:
        print("❌ No ticket tables found in database.")
        print()
        print("💡 Run migrations: python scripts/db_migrate.py upgrade head")
        sys.exit(1)
        
except ImportError as e:
    print(f"❌ ERROR: Missing dependency: {e}", file=sys.stderr)
    print("   Install with: pip install sqlalchemy psycopg2-binary", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}", file=sys.stderr)
    sys.exit(1)
