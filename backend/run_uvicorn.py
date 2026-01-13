#!/usr/bin/env python3
"""
Wrapper script that runs inside uvicorn's reload subprocess.
This ensures PYTHONPATH is set BEFORE any imports happen.
"""
import os
import sys
from pathlib import Path

# Set PYTHONPATH BEFORE any imports
# Calculate project root from current working directory
cwd = Path.cwd()
if cwd.name == 'backend':
    project_root = cwd.parent
else:
    project_root = cwd

pythonpath = str(project_root.resolve())
os.environ['PYTHONPATH'] = pythonpath
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# Now import and run uvicorn
if __name__ == "__main__":
    from uvicorn import main
    # Run uvicorn with the app string
    sys.argv = ['uvicorn', 'backend.api:app'] + sys.argv[1:]
    main()
