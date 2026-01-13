#!/usr/bin/env python3
"""
Validate tickets pipeline: migrations + parity + smoke tests.

Runs a complete validation workflow:
1. Check current migration status
2. Run pending migrations (dry-run check)
3. Run ticket migration (dry-run)
4. Run parity verification
5. Run smoke test for ticket reads

Usage:
    python scripts/validate_tickets_pipeline.py
    
    # Skip migrations (if already applied)
    python scripts/validate_tickets_pipeline.py --skip-migrations
    
    # Custom sample sizes
    python scripts/validate_tickets_pipeline.py --parity-sample 50 --smoke-sample 20
"""

import argparse
import os
import subprocess
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
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)

# Verify DATABASE_URL is set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("❌ ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    print("   Set it in backend/.env file or as an environment variable", file=sys.stderr)
    sys.exit(1)

# Verify SQLite path exists
sqlite_path = project_root / "Scraper" / "data" / "tickets.db"
if not sqlite_path.exists():
    print(f"⚠️  Warning: SQLite database not found at {sqlite_path}", file=sys.stderr)
    print("   Parity checks will be skipped", file=sys.stderr)
    sqlite_path = None


def run_command(cmd: list, description: str, allow_failure: bool = False) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'=' * 70}", file=sys.stderr)
    print(f"STEP: {description}", file=sys.stderr)
    print(f"{'=' * 70}", file=sys.stderr)
    print(f"Running: {' '.join(cmd)}\n", file=sys.stderr)
    
    result = subprocess.run(cmd, cwd=project_root)
    
    if result.returncode != 0:
        if allow_failure:
            print(f"⚠️  {description} failed (non-critical)", file=sys.stderr)
            return False
        else:
            print(f"❌ {description} failed", file=sys.stderr)
            return False
    
    print(f"✅ {description} passed", file=sys.stderr)
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Validate tickets pipeline: migrations + parity + smoke tests",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--skip-migrations",
        action="store_true",
        help="Skip migration checks (use if migrations already applied)"
    )
    
    parser.add_argument(
        "--parity-sample",
        type=int,
        default=20,
        help="Number of rows to sample for parity check (default: 20)"
    )
    
    parser.add_argument(
        "--smoke-sample",
        type=int,
        default=10,
        help="Number of tickets to sample for smoke test (default: 10)"
    )
    
    parser.add_argument(
        "--sqlite-path",
        help="Path to SQLite tickets database (default: Scraper/data/tickets.db)"
    )
    
    args = parser.parse_args()
    
    sqlite_db_path = args.sqlite_path or (sqlite_path if sqlite_path else None)
    
    print("=" * 70, file=sys.stderr)
    print("TICKETS PIPELINE VALIDATION", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print(f"Project root: {project_root}", file=sys.stderr)
    print(f"SQLite DB: {sqlite_db_path or 'NOT FOUND'}", file=sys.stderr)
    print(f"Parity sample: {args.parity_sample} rows", file=sys.stderr)
    print(f"Smoke sample: {args.smoke_sample} tickets", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    all_passed = True
    
    # Step 1: Check current migration status
    if not args.skip_migrations:
        if not run_command(
            [sys.executable, "scripts/db_migrate.py", "current"],
            "Check current migration status",
            allow_failure=False
        ):
            all_passed = False
            print("\n❌ FAIL: Migration status check failed", file=sys.stderr)
            sys.exit(1)
    
    # Step 2: Run pending migrations (dry-run check via upgrade head)
    # Note: This actually applies migrations. For true dry-run, we'd need Alembic's --sql flag
    # but that's more complex. For CI, we want to ensure migrations are applied.
    if not args.skip_migrations:
        print("\n⚠️  Note: Running 'upgrade head' will apply pending migrations.", file=sys.stderr)
        print("   For dry-run validation, use: alembic upgrade head --sql", file=sys.stderr)
        # Skip actual upgrade in validation script - let CI/CD handle it
        # if not run_command(
        #     [sys.executable, "scripts/db_migrate.py", "upgrade", "head"],
        #     "Apply pending migrations",
        #     allow_failure=False
        # ):
        #     all_passed = False
    
    # Step 3: Run ticket migration (dry-run)
    if sqlite_db_path:
        migrate_cmd = [
            sys.executable, "-m", "backend.scripts.migrate_tickets_sqlite_to_postgres",
            "--dry-run",
            "--sqlite-path", str(sqlite_db_path),
            "--orphan-policy", "skip"
        ]
        
        if not run_command(
            migrate_cmd,
            "Ticket migration (dry-run)",
            allow_failure=False
        ):
            all_passed = False
            print("\n❌ FAIL: Ticket migration dry-run failed", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n⚠️  Skipping ticket migration check (SQLite DB not found)", file=sys.stderr)
    
    # Step 4: Run parity verification
    if sqlite_db_path:
        parity_cmd = [
            sys.executable, "-m", "backend.scripts.verify_tickets_parity",
            "--sqlite-path", str(sqlite_db_path),
            "--sample", str(args.parity_sample)
        ]
        
        if not run_command(
            parity_cmd,
            "Parity verification",
            allow_failure=False
        ):
            all_passed = False
            print("\n❌ FAIL: Parity verification failed", file=sys.stderr)
            sys.exit(1)
    else:
        print("\n⚠️  Skipping parity verification (SQLite DB not found)", file=sys.stderr)
    
    # Step 5: Run smoke test
    smoke_cmd = [
        sys.executable, "-m", "backend.scripts.smoke_ticket_reads",
        "--sample", str(args.smoke_sample)
    ]
    
    if not run_command(
        smoke_cmd,
        "Smoke test (ticket reads)",
        allow_failure=False
    ):
        all_passed = False
        print("\n❌ FAIL: Smoke test failed", file=sys.stderr)
        sys.exit(1)
    
    # Final summary
    print("\n" + "=" * 70, file=sys.stderr)
    print("VALIDATION SUMMARY", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    
    if all_passed:
        print("✅ PASS: All validation steps completed successfully", file=sys.stderr)
        sys.exit(0)
    else:
        print("❌ FAIL: Some validation steps failed", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
