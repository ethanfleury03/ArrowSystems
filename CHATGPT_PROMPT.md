# Uvicorn Reload ImportError on Windows - Debugging Prompt

## Problem Statement

I'm experiencing a persistent `ImportError: attempted relative import with no known parent package` when using uvicorn's `--reload` feature on Windows. The error occurs specifically in uvicorn's reload subprocess when it tries to import `backend.api:app`.

**Environment:**
- Windows 10 (10.0.26100)
- Python 3.12
- PowerShell
- Uvicorn with `--reload` enabled
- Project structure: `ArrowSystems/backend/api.py` (uses relative imports like `from .utils.database_manager import DatabaseManager`)

**Error Traceback:**
```
Process SpawnProcess-43:
Traceback (most recent call last):
  File "C:\Program Files\Python312\Lib\multiprocessing\process.py", line 314, in _bootstrap
    self.run()
  File "C:\Program Files\Python312\Lib\multiprocessing\process.py", line 108, in run
    self._target(*self._args, **self._kwargs)
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\_subprocess.py", line 80, in subprocess_started
    target(sockets=sockets)
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\server.py", line 67, in run
    return asyncio_run(self.serve(sockets=sockets), loop_factory=self.config.get_loop_factory())
  File "C:\Program Files\Python312\Lib\asyncio\runners.py", line 194, in run
    return runner.run(main)
  File "C:\Program Files\Python312\Lib\asyncio\runners.py", line 118, in run
    return self._loop.run_until_complete(task)
  File "C:\Program Files\Python312\Lib\asyncio\base_events.py", line 687, in run_until_complete
    return future.result()
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\server.py", line 71, in serve
    await self._serve(sockets)
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\server.py", line 78, in _serve
    config.load()
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\config.py", line 439, in load
    self.loaded_app = import_from_string(self.app)
  File "C:\Users\ethan\AppData\Roaming\Python\Python312\site-packages\uvicorn\importer.py", line 19, in import_from_string
    module = importlib.import_module(module_str)
  File "C:\Program Files\Python312\Lib\importlib\__init__.py", line 90, in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
  File "<frozen importlib._bootstrap>", line 1387, in _gcd_import
  File "<frozen importlib._bootstrap>", line 1360, in _find_and_load
  File "<frozen importlib._bootstrap>", line 1331, in _find_and_load_unlocked
  File "<frozen importlib._bootstrap>", line 935, in _load_unlocked
  File "<frozen importlib._bootstrap_external>", line 995, in exec_module
  File "<frozen importlib._bootstrap>", line 488, in _call_with_frames_removed
  File "C:\Users\ethan\ArrowSystems\backend\api.py", line 63, in <module>
    from .utils.database_manager import DatabaseManager
ImportError: attempted relative import with no known parent package
```

**Key Issue:**
The import happens in `uvicorn.config.load()` -> `import_from_string()` which runs BEFORE `subprocess_started` callback. The child process spawned by multiprocessing on Windows doesn't have the project root in `sys.path`, so relative imports fail.

## What I've Tried (All Failed)

### Attempt 1: Setting PYTHONPATH in Environment
- Set `PYTHONPATH` environment variable to project root before running uvicorn
- Result: Parent process works, but child process doesn't inherit it correctly

### Attempt 2: Monkey-patching subprocess.Popen
- Patched `subprocess.Popen` to always include `PYTHONPATH` in subprocess environment
- Result: Didn't work because uvicorn uses `multiprocessing.Process`, not `subprocess.Popen` directly

### Attempt 3: Monkey-patching multiprocessing.Process.__init__
- Patched `multiprocessing.Process.__init__` to pass `PYTHONPATH` in environment
- Result: Didn't work because Windows uses 'spawn' method which doesn't pass environment this way

### Attempt 4: Monkey-patching multiprocessing.spawn.get_preparation_data
- Patched `multiprocessing.spawn.get_preparation_data` to add project root to `sys_path` (which child process uses to initialize `sys.path`)
- Result: `sys_path` was modified but child process still fails import

### Attempt 5: Monkey-patching multiprocessing.spawn.import_main_path
- Patched `import_main_path` to set `sys.path` immediately in child process before any imports
- Result: Function runs, but import still fails - timing issue?

### Attempt 6: Monkey-patching uvicorn.importer.import_from_string
- Patched `uvicorn.importer.import_from_string` to catch `ImportError` and fix `sys.path` before retrying
- Result: Patch runs in parent process, but child process has its own copy of uvicorn module, so patch doesn't apply

### Attempt 7: Monkey-patching uvicorn._subprocess.subprocess_started
- Tried to patch `subprocess_started` callback to set `PYTHONPATH` in child process
- Result: Pickling error - function can't be pickled when defined inside `if __name__ == "__main__"` block. Also, this runs AFTER the import has already failed.

## Current Code State

**File: `backend/run_dev.py`**
```python
#!/usr/bin/env python3
"""
Development server runner that loads .env file before starting uvicorn.
"""
import os
import sys
from pathlib import Path

# Get the backend directory (where this script lives) and project root
backend_dir = Path(__file__).parent
project_root = backend_dir.parent

# Set PYTHONPATH to project root so 'backend' module can be imported
pythonpath = str(project_root.resolve())  # Use absolute path

# CRITICAL: Set PYTHONPATH in environment BEFORE importing uvicorn
os.environ["PYTHONPATH"] = pythonpath

# Also add to sys.path for current process
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# Load .env file from backend directory
try:
    from dotenv import load_dotenv
    env_path = backend_dir / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed.", file=sys.stderr)

# Change to project root (required for relative imports in backend.api to work)
os.chdir(project_root)

if __name__ == "__main__":
    import uvicorn
    
    # Verify backend can be imported in parent process
    try:
        import backend
        print(f"✅ Backend package found at: {backend.__file__}", file=sys.stderr)
    except ImportError as e:
        print(f"❌ ERROR: Cannot import backend package: {e}", file=sys.stderr)
        sys.exit(1)
    
    # CRITICAL FIX: Patch multiprocessing.spawn for Windows
    import multiprocessing.spawn as spawn_module
    
    # Patch get_preparation_data - this dict gets pickled and sent to child
    original_get_preparation_data = spawn_module.get_preparation_data
    
    def patched_get_preparation_data(name):
        prep_data = original_get_preparation_data(name)
        # Modify sys_path - child process uses this to set sys.path
        if 'sys_path' in prep_data:
            sys_path = prep_data['sys_path']
            if not isinstance(sys_path, list):
                sys_path = list(sys_path)
            if pythonpath not in sys_path:
                sys_path.insert(0, pythonpath)
            prep_data['sys_path'] = sys_path
        return prep_data
    
    spawn_module.get_preparation_data = patched_get_preparation_data
    
    # Patch import_main_path - runs VERY EARLY in child process
    if hasattr(spawn_module, 'import_main_path'):
        original_import_main_path = spawn_module.import_main_path
        
        def patched_import_main_path(main_path):
            # Set sys.path IMMEDIATELY in child process (before any imports)
            import sys
            import os
            # Read PYTHONPATH from environment
            pythonpath_env = os.environ.get('PYTHONPATH', '')
            if pythonpath_env:
                for path in pythonpath_env.split(os.pathsep):
                    if path and path not in sys.path:
                        sys.path.insert(0, path)
            # Also calculate from cwd as backup
            cwd = os.getcwd()
            if os.path.basename(cwd) == 'backend':
                project_root = os.path.dirname(cwd)
            else:
                project_root = cwd
            pythonpath_val = os.path.abspath(project_root)
            if pythonpath_val not in sys.path:
                sys.path.insert(0, pythonpath_val)
            os.environ['PYTHONPATH'] = pythonpath_val
            return original_import_main_path(main_path)
        
        spawn_module.import_main_path = patched_import_main_path
    
    # Patch uvicorn.importer.import_from_string (runs in parent, not child)
    try:
        import uvicorn.importer
        original_import_from_string = uvicorn.importer.import_from_string
        
        def patched_import_from_string(app_str):
            try:
                return original_import_from_string(app_str)
            except ImportError as e:
                if 'attempted relative import' in str(e) or 'No module named' in str(e):
                    import sys
                    import os
                    cwd = os.getcwd()
                    if os.path.basename(cwd) == 'backend':
                        project_root = os.path.dirname(cwd)
                    else:
                        project_root = cwd
                    pythonpath_val = os.path.abspath(project_root)
                    if pythonpath_val not in sys.path:
                        sys.path.insert(0, pythonpath_val)
                    os.environ['PYTHONPATH'] = pythonpath_val
                    return original_import_from_string(app_str)
                raise
        
        uvicorn.importer.import_from_string = patched_import_from_string
    except (ImportError, AttributeError):
        pass
    
    print(f"✅ Patched multiprocessing.spawn.get_preparation_data + import_main_path + uvicorn.importer", file=sys.stderr)
    
    # Run uvicorn
    uvicorn.run("backend.api:app", reload=True, port=8000, host="127.0.0.1")
```

**VS Code Task (`.vscode/tasks.json`):**
```json
{
  "label": "Dev: Backend (FastAPI)",
  "type": "shell",
  "command": "$env:PYTHONPATH=\"$PWD\"; $env:DISABLE_RAG='true'; python backend/run_dev.py",
  "isBackground": true,
  "problemMatcher": [],
  "presentation": {
    "group": "dev",
    "panel": "dedicated",
    "clear": true
  }
}
```

## Root Cause Analysis

The fundamental issue is:
1. **Windows multiprocessing uses 'spawn' method**: Creates a fresh Python interpreter for child process
2. **Child process doesn't inherit parent's sys.path**: Even though we set `PYTHONPATH` in environment, the child process initializes `sys.path` from `sys_path` in `get_preparation_data`, which may not include our project root
3. **Import happens before any callbacks**: `uvicorn.config.load()` -> `import_from_string()` runs immediately when child process starts, before `subprocess_started` callback
4. **Relative imports require package context**: `from .utils.database_manager` requires `backend` to be a known package, which requires project root in `sys.path`

## What I Need

A working solution that:
1. Ensures the child process spawned by uvicorn's reload mechanism can import `backend.api:app` on Windows
2. Works with relative imports in `backend/api.py`
3. Doesn't require modifying `backend/api.py` or project structure
4. Works with uvicorn's `--reload` feature

## Questions to Consider

1. Is there a way to ensure `PYTHONPATH` is properly inherited by the child process on Windows spawn?
2. Can we patch uvicorn's reload mechanism to pass environment variables correctly?
3. Is there a way to set `sys.path` in the child process BEFORE uvicorn tries to import the module?
4. Should we use a different approach, like creating a wrapper script that runs inside the child process?
5. Is there a uvicorn configuration option we're missing?

Please provide a working solution with explanation of why it works when the previous attempts failed.
