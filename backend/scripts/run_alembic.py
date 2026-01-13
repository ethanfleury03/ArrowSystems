#!/usr/bin/env python3
"""
Helper script to run Alembic migrations with proper .env loading.

This script loads backend/.env before running Alembic commands,
ensuring DATABASE_URL is available.

⚠️  PREFERRED: Use the repo-root script instead:
    python scripts/db_migrate.py upgrade head

This script is kept for backward compatibility but the repo-root script
is recommended as it works consistently from any location.

Usage:
    python backend/scripts/run_alembic.py upgrade head
    python backend/scripts/run_alembic.py current
    python backend/scripts/run_alembic.py history
"""

import os
import sys
from pathlib import Path

# Get project root (2 levels up from backend/scripts/)
project_root = Path(__file__).parent.parent.parent.resolve()
backend_dir = project_root / "backend"

# Change to project root
os.chdir(project_root)

# Add project root to Python path
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"⚠️  Warning: .env file not found at {env_path}", file=sys.stderr)
        print(f"   DATABASE_URL must be set as environment variable", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. DATABASE_URL must be set as env var.", file=sys.stderr)

# Verify DATABASE_URL is set
database_url = os.getenv("DATABASE_URL")
if not database_url:
    print("❌ ERROR: DATABASE_URL environment variable is required", file=sys.stderr)
    print("   Set it in backend/.env file or as an environment variable", file=sys.stderr)
    sys.exit(1)

# Mask password in output
if "@" in database_url:
    masked_url = database_url.split("@")[1]
    print(f"✅ DATABASE_URL: postgresql://***@{masked_url}", file=sys.stderr)
else:
    print(f"✅ DATABASE_URL: (set)", file=sys.stderr)

# Run Alembic with the provided arguments
import subprocess

alembic_cmd = [
    sys.executable,
    "-m", "alembic",
    "-c", "backend/migrations/alembic.ini"
] + sys.argv[1:]

print(f"\n🔧 Running: {' '.join(alembic_cmd)}\n", file=sys.stderr)

result = subprocess.run(alembic_cmd, cwd=project_root)
sys.exit(result.returncode)
