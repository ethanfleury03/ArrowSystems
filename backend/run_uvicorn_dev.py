#!/usr/bin/env python3
"""
Development server runner that loads .env file before starting uvicorn.
This script ensures PYTHONPATH is set and .env is loaded before running uvicorn.
"""
import os
import sys
from pathlib import Path

# Get the backend directory (where this script lives) and project root
backend_dir = Path(__file__).parent
project_root = backend_dir.parent.resolve()  # Use absolute path

# CRITICAL: Set PYTHONPATH to project root so child process can find backend module
pythonpath = str(project_root)
os.environ["PYTHONPATH"] = pythonpath

# Add to sys.path for current process
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# Change to project root (required for relative imports in backend.api to work)
os.chdir(project_root)

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
        # Debug: Check DATABASE_URL format (don't print full value for security)
        db_url = os.getenv("DATABASE_URL", "")
        if db_url:
            if db_url.startswith("DATABASE_URL="):
                print(f"❌ ERROR: DATABASE_URL value includes key name. Fix your .env file!", file=sys.stderr)
                print(f"   Your .env file probably has: DATABASE_URL=DATABASE_URL=postgresql+psycopg2://...", file=sys.stderr)
                print(f"   It should be: DATABASE_URL=postgresql+psycopg2://...", file=sys.stderr)
                print(f"   Current value: {db_url[:60]}...", file=sys.stderr)
                # Try to fix it automatically
                fixed_url = db_url.replace("DATABASE_URL=", "", 1)
                if fixed_url != db_url:
                    os.environ["DATABASE_URL"] = fixed_url
                    print(f"✅ Auto-fixed DATABASE_URL (removed duplicate key name)", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)

# Now run uvicorn as a module (this ensures proper package resolution)
if __name__ == "__main__":
    # Ensure PYTHONPATH is set before uvicorn runs (for child process)
    os.environ["PYTHONPATH"] = pythonpath
    
    # Import uvicorn and run as module
    import uvicorn
    import sys
    
    # Run uvicorn.main() which is what "python -m uvicorn" does
    sys.argv = [
        "uvicorn",
        "backend.api:app",
        "--reload",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    uvicorn.main()
