#!/usr/bin/env python3
r"""
Repo-root helper script for running Alembic database migrations.

This script loads backend/.env, validates DATABASE_URL, shows target database info,
and invokes Alembic with the correct config file.

Usage:
    python scripts/db_migrate.py current
    python scripts/db_migrate.py history [--last N]
    python scripts/db_migrate.py upgrade head
    python scripts/db_migrate.py downgrade -1

All commands must be run from the repository root.
"""

import os
import sys
import subprocess
from pathlib import Path
from urllib.parse import urlparse

# Get project root (where this script lives)
project_root = Path(__file__).parent.parent.resolve()

# Change to project root
os.chdir(project_root)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from backend directory
env_loaded = False
try:
    from dotenv import load_dotenv
    backend_dir = project_root / "backend"
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        env_loaded = True
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

# Parse and display target database info (safely, without password)
try:
    parsed = urlparse(database_url)
    # Mask password
    if parsed.password:
        masked_url = f"{parsed.scheme}://{parsed.username}:***@{parsed.hostname}"
        if parsed.port:
            masked_url += f":{parsed.port}"
        masked_url += parsed.path
    else:
        masked_url = database_url.split("@")[1] if "@" in database_url else database_url
    
    print(f"✅ DATABASE_URL: {masked_url}", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Warning: Could not parse DATABASE_URL: {e}", file=sys.stderr)

# Connect to database and show target info
try:
    from sqlalchemy import create_engine, text
    
    engine = create_engine(database_url, pool_pre_ping=True, connect_args={"connect_timeout": 5})
    with engine.connect() as conn:
        result = conn.execute(text("SELECT current_database(), current_user"))
        row = result.fetchone()
        db_name = row[0]
        db_user = row[1]
        print(f"🎯 Target Database: {db_name} (user: {db_user})", file=sys.stderr)
except Exception as e:
    print(f"⚠️  Warning: Could not connect to database: {e}", file=sys.stderr)
    print(f"   Continuing anyway...", file=sys.stderr)

# Parse command line arguments
if len(sys.argv) < 2:
    print("Usage: python scripts/db_migrate.py <command> [args...]", file=sys.stderr)
    print("", file=sys.stderr)
    print("Commands:", file=sys.stderr)
    print("  current              Show current migration revision", file=sys.stderr)
    print("  history [--last N]   Show migration history (optionally last N)", file=sys.stderr)
    print("  upgrade head         Apply all pending migrations", file=sys.stderr)
    print("  downgrade -1         Rollback last migration", file=sys.stderr)
    sys.exit(1)

command = sys.argv[1]
alembic_args = sys.argv[2:]

# Handle special cases
if command == "history" and "--last" in alembic_args:
    # Convert --last N to Alembic's -n N format
    try:
        last_idx = alembic_args.index("--last")
        if last_idx + 1 < len(alembic_args):
            n_value = alembic_args[last_idx + 1]
            alembic_args = ["-n", n_value] + [a for i, a in enumerate(alembic_args) if i not in (last_idx, last_idx + 1)]
    except ValueError:
        pass

# Build Alembic command
alembic_cmd = [
    sys.executable,
    "-m", "alembic",
    "-c", "backend/migrations/alembic.ini"
] + [command] + alembic_args

print(f"\n🔧 Running: {' '.join(alembic_cmd)}\n", file=sys.stderr)

# Run Alembic
result = subprocess.run(alembic_cmd, cwd=project_root)
sys.exit(result.returncode)
