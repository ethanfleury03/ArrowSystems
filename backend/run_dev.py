#!/usr/bin/env python3
"""
Development server runner that loads .env file before starting uvicorn.
This ensures environment variables are loaded from backend/.env for local development.
"""
import os
import sys
from pathlib import Path

# Get the backend directory (where this script lives) and project root
backend_dir = Path(__file__).parent
project_root = backend_dir.parent

# Set PYTHONPATH to project root so 'backend' module can be imported
# This is required for uvicorn --reload subprocess to find the backend package
pythonpath = str(project_root.resolve())  # Use absolute path

# CRITICAL: Set PYTHONPATH in environment BEFORE importing uvicorn
# This ensures uvicorn's reload subprocess inherits it
os.environ["PYTHONPATH"] = pythonpath

# Also add to sys.path for current process
if pythonpath not in sys.path:
    sys.path.insert(0, pythonpath)

# Load .env file from backend directory
# Note: Environment variables set before this script runs will override .env file values
try:
    from dotenv import load_dotenv
    
    env_path = backend_dir / ".env"
    
    if env_path.exists():
        load_dotenv(env_path, override=False)  # Don't override existing env vars
        print(f"✅ Loaded environment variables from {env_path}", file=sys.stderr)
    else:
        print(f"⚠️  Warning: {env_path} not found. Using system environment variables only.", file=sys.stderr)
except ImportError:
    print("⚠️  Warning: python-dotenv not installed. Using system environment variables only.", file=sys.stderr)

# Log DISABLE_RAG status if set
if os.getenv("DISABLE_RAG"):
    print(f"✅ DISABLE_RAG={os.getenv('DISABLE_RAG')} (from environment)", file=sys.stderr)

# Change to project root (required for relative imports in backend.api to work)
os.chdir(project_root)

# Now run uvicorn
# Using backend.api:app because backend/ is a package with relative imports
# CRITICAL: Uvicorn's reload subprocess needs PYTHONPATH set in the environment
# The reload subprocess uses subprocess.Popen which inherits os.environ
if __name__ == "__main__":
    import uvicorn
    
    # Print PYTHONPATH for debugging
    print(f"✅ PYTHONPATH={os.environ.get('PYTHONPATH')}", file=sys.stderr)
    print(f"✅ Working directory: {os.getcwd()}", file=sys.stderr)
    print(f"✅ Project root: {project_root}", file=sys.stderr)
    
    # CRITICAL: Ensure PYTHONPATH is set before uvicorn imports anything
    # Uvicorn's reload mechanism spawns: python -m uvicorn backend.api:app
    # That subprocess needs PYTHONPATH to find the backend package
    os.environ["PYTHONPATH"] = pythonpath
    
    # Verify backend can be imported
    try:
        import backend
        print(f"✅ Backend package found at: {backend.__file__}", file=sys.stderr)
    except ImportError as e:
        print(f"❌ ERROR: Cannot import backend package: {e}", file=sys.stderr)
        print(f"   PYTHONPATH={os.environ.get('PYTHONPATH')}", file=sys.stderr)
        print(f"   sys.path={sys.path[:3]}...", file=sys.stderr)
        sys.exit(1)
    
    # CRITICAL: Monkey-patch uvicorn's subprocess creation to ensure PYTHONPATH is passed
    # Uvicorn uses multiprocessing and subprocess, we need to patch both
    import subprocess
    import multiprocessing
    
    # Patch subprocess.Popen
    original_popen = subprocess.Popen
    def patched_popen(*args, **kwargs):
        # Ensure PYTHONPATH is in the environment for subprocess
        if 'env' not in kwargs:
            kwargs['env'] = os.environ.copy()
        elif kwargs['env'] is not None:
            kwargs['env'] = kwargs['env'].copy()
        else:
            kwargs['env'] = os.environ.copy()
        # Always set PYTHONPATH
        kwargs['env']['PYTHONPATH'] = pythonpath
        return original_popen(*args, **kwargs)
    subprocess.Popen = patched_popen
    
    # Patch multiprocessing.Process to pass PYTHONPATH
    original_process_init = multiprocessing.Process.__init__
    def patched_process_init(self, *args, **kwargs):
        # Ensure environment is passed
        if 'env' in kwargs and kwargs['env'] is not None:
            kwargs['env'] = kwargs['env'].copy()
            kwargs['env']['PYTHONPATH'] = pythonpath
        return original_process_init(self, *args, **kwargs)
    multiprocessing.Process.__init__ = patched_process_init
    
    # CRITICAL FIX: On Windows spawn, child process needs project root in sys.path BEFORE imports
    # The import happens in uvicorn.config.load() -> import_from_string() BEFORE subprocess_started runs
    # Solution: Ensure PYTHONPATH is in environment AND patch multiprocessing.spawn to set sys.path
    
    import multiprocessing.spawn as spawn_module
    
    # CRITICAL: Patch get_preparation_data - this dict gets pickled and sent to child
    # Child uses sys_path to initialize sys.path BEFORE any imports
    original_get_preparation_data = spawn_module.get_preparation_data
    
    def patched_get_preparation_data(name):
        prep_data = original_get_preparation_data(name)
        # CRITICAL: Modify sys_path - child process uses this to set sys.path
        if 'sys_path' in prep_data:
            sys_path = prep_data['sys_path']
            # Ensure it's a mutable list
            if not isinstance(sys_path, list):
                sys_path = list(sys_path)
            # Add project root at the beginning if not already there
            if pythonpath not in sys_path:
                sys_path.insert(0, pythonpath)
            prep_data['sys_path'] = sys_path
        # CRITICAL: Also ensure PYTHONPATH is in init_main_from_path (environment)
        if 'init_main_from_path' in prep_data:
            # This is the path to the script that will be run in child
            # We can't modify it, but we ensure PYTHONPATH is set
            pass
        return prep_data
    
    spawn_module.get_preparation_data = patched_get_preparation_data
    
    # CRITICAL: Patch the function that actually initializes sys.path in child process
    # This runs VERY EARLY in child, before any module imports
    if hasattr(spawn_module, 'import_main_path'):
        original_import_main_path = spawn_module.import_main_path
        
        def patched_import_main_path(main_path):
            # Set sys.path IMMEDIATELY in child process (before any imports)
            import sys
            import os
            # Read PYTHONPATH from environment (should be set by parent)
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
    
    # CRITICAL: Patch uvicorn's import_from_string to catch ImportError and fix sys.path
    # This is the ACTUAL function that tries to import backend.api - it runs BEFORE subprocess_started
    # NOTE: This patch runs in parent, but uvicorn in child will have its own copy
    # So we need to ensure sys.path is correct BEFORE this runs
    try:
        import uvicorn.importer
        original_import_from_string = uvicorn.importer.import_from_string
        
        def patched_import_from_string(app_str):
            # If import fails, try adding project root to sys.path and retry
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
                    # Retry import after fixing sys.path
                    return original_import_from_string(app_str)
                raise
        
        uvicorn.importer.import_from_string = patched_import_from_string
        print(f"✅ Patched uvicorn.importer.import_from_string", file=sys.stderr)
    except (ImportError, AttributeError) as e:
        print(f"⚠️  Warning: Could not patch uvicorn.importer: {e}", file=sys.stderr)
    
    # NOTE: We don't patch subprocess_started because:
    # 1. It runs AFTER the import has already failed
    # 2. It can't be pickled when defined inside if __name__ == "__main__"
    # Instead, we rely on get_preparation_data and import_main_path patches
    print(f"✅ Patched multiprocessing.spawn.get_preparation_data + import_main_path + uvicorn.importer", file=sys.stderr)
    
    # Run uvicorn - sys_path patch ensures child process can import backend.api
    uvicorn.run("backend.api:app", reload=True, port=8000, host="127.0.0.1")
