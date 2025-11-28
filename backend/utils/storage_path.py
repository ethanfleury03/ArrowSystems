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
    2. In prod: /app/latest_model (canonical Cloud Run path)
    3. Dev/local paths: latest_model, ../latest_model, /workspace/*, etc.
    
    Returns:
        Path to index directory, or None if not found
    """
    # 1) Explicit override via environment variable
    env_path = os.getenv("RAG_INDEX_DIR")
    if env_path:
        path = Path(env_path)
        if path.is_dir():
            return path
    
    # Build candidate paths based on environment
    candidates: list[Path] = []
    
    # In production, /app/latest_model is the FIRST and REQUIRED path
    # This is where Cloud Run mounts the GCS bucket
    if settings.ENV in ("prod", "production", "cloud"):
        prod_path = Path("/app/latest_model")
        candidates.append(prod_path)
        
        # In production, if /app/latest_model doesn't exist, it's a fatal error
        # (We'll check this after trying all candidates)
    
    # Add dev/local paths (for development, RunPod, etc.)
    candidates.extend([
        Path("latest_model"),                    # Current directory
        Path("../latest_model"),                 # Parent directory (for scripts/)
        Path("/workspace/latest_model"),         # RunPod workspace
        Path("/workspace/ArrowSystems/latest_model"),  # RunPod with ArrowSystems
        Path("/workspace/storage"),              # Old storage location
        Path("./storage"),                       # Local storage
    ])
    
    # Try each candidate path
    for candidate in candidates:
        if candidate.is_dir():
            # Verify it looks like a valid index (has docstore.json)
            docstore_path = candidate / "docstore.json"
            if docstore_path.exists():
                return candidate
    
    # No valid index found
    # Return None in all environments (production and dev/test)
    # This allows the caller to handle missing index gracefully
    # The caller (lifespan function) will log appropriate warnings and continue startup
    # RAG endpoints will return 503 when index is missing, but non-RAG routes will work
    return None

