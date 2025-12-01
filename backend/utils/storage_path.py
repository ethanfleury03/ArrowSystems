"""
Storage path resolution utilities for RAG index.

Handles finding the correct index directory based on environment and configuration.
"""

import os
from pathlib import Path
from typing import Optional
from backend.config.env import settings


def resolve_storage_path() -> Optional[Path]:
    """
    Resolve the storage path for the RAG index.
    
    Priority order:
    1. RAG_INDEX_DIR environment variable (if set and directory exists)
    2. In prod: /app/latest_model (canonical Cloud Run path) - ALWAYS returned if directory exists, even if files are missing
    3. Dev/local paths: latest_model, ../latest_model, /workspace/*, etc.
    
    Returns:
        Path to index directory, or None if not found
        
    Note: In production, this will return /app/latest_model if the directory exists,
    even if index files are missing. This allows lazy initialization to attempt loading
    and provide better error messages. In dev, it only returns paths with valid index files.
    """
    # 1) Explicit override via environment variable
    env_path = os.getenv("RAG_INDEX_DIR")
    if env_path:
        path = Path(env_path)
        if path.is_dir():
            return path
    
    # In production, /app/latest_model is the REQUIRED path
    # This is where Cloud Run mounts the GCS bucket
    # Return it if the directory exists, even if files are missing (lazy init will handle errors)
    if settings.ENV in ("prod", "production", "cloud"):
        prod_path = Path("/app/latest_model")
        if prod_path.is_dir():
            # In production, return the path even if files are missing
            # This allows lazy initialization to attempt loading and provide better error messages
            return prod_path
        # If directory doesn't exist, continue to check dev paths (for local testing)
    
    # Build candidate paths for dev/local environments
    # In dev, we require valid index files to exist
    candidates: list[Path] = [
        Path("latest_model"),                    # Current directory
        Path("../latest_model"),                 # Parent directory (for scripts/)
        Path("/workspace/latest_model"),         # RunPod workspace
        Path("/workspace/ArrowSystems/latest_model"),  # RunPod with ArrowSystems
        Path("/workspace/storage"),              # Old storage location
        Path("./storage"),                       # Local storage
    ]
    
    # Try each candidate path (in dev, require valid index files)
    for candidate in candidates:
        if candidate.is_dir():
            # Verify it looks like a valid index (has docstore.json)
            docstore_path = candidate / "docstore.json"
            if docstore_path.exists():
                return candidate
    
    # No valid index found
    # Return None to allow graceful degradation
    # The caller (lifespan function) will log appropriate warnings and continue startup
    # RAG endpoints will return 503 when index is missing, but non-RAG routes will work
    return None

