#!/usr/bin/env python3
"""
Simple development server runner that loads .env file before starting uvicorn.
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
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)

# Run uvicorn
if __name__ == "__main__":
    import uvicorn
    # Ensure PYTHONPATH is set before uvicorn runs (for child process)
    os.environ["PYTHONPATH"] = pythonpath
    print(f"✅ PYTHONPATH={pythonpath}", file=sys.stderr)
    uvicorn.run("backend.api:app", reload=True, port=8000, host="127.0.0.1")
