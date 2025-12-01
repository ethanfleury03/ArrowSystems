"""
Storage path resolution utilities for RAG index.

Handles finding the correct index directory based on environment and configuration.
"""

import os
from pathlib import Path
from typing import Optional
from backend.config.env import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)


def resolve_storage_path() -> Optional[Path]:
    """
    Resolve the storage path for the RAG index.
    
    Priority order:
    1. RAG_INDEX_DIR environment variable (if set)
    2. In prod: /app/latest_model (canonical Cloud Run path) - ALWAYS returned in prod
    3. Dev/local paths: latest_model, ../latest_model, /workspace/*, etc.
    
    Returns:
        Path to index directory, or None if not found
        
    Note: In production, this ALWAYS returns /app/latest_model (even if directory doesn't exist).
    This allows lazy initialization to attempt loading and provide better error messages.
    In dev, it only returns paths with valid index files.
    """
    # 1) Explicit override via environment variable
    env_path = os.getenv("RAG_INDEX_DIR")
    if env_path:
        path = Path(env_path).resolve()  # Make absolute
        logger.info("rag_storage_path_env_override", 
                   env_path=env_path,
                   resolved_path=str(path),
                   exists=path.exists(),
                   is_dir=path.is_dir() if path.exists() else False)
        if path.exists() and path.is_dir():
            return path
        else:
            logger.warning("rag_storage_path_env_override_invalid",
                         env_path=env_path,
                         resolved_path=str(path),
                         exists=path.exists())
    
    # In production, /app/latest_model is the REQUIRED and ONLY path
    # This is where Cloud Run mounts the GCS bucket
    # ALWAYS return it in prod, even if directory doesn't exist (lazy init will handle errors)
    if settings.ENV in ("prod", "production", "cloud"):
        prod_path = Path("/app/latest_model").resolve()  # Ensure absolute
        exists = prod_path.exists()
        is_dir = prod_path.is_dir() if exists else False
        
        logger.info("rag_storage_path_prod_resolution",
                   prod_path=str(prod_path),
                   exists=exists,
                   is_dir=is_dir,
                   env=settings.ENV)
        
        if not exists:
            logger.error("rag_storage_path_prod_missing",
                        prod_path=str(prod_path),
                        message="Production storage path does not exist! "
                               "Check Cloud Run volume mount configuration. "
                               "Expected: Volume source=arrow-rag-support-prod-rag/latest_model/, "
                               "Mount path=/app/latest_model")
        elif not is_dir:
            logger.error("rag_storage_path_prod_not_directory",
                        prod_path=str(prod_path),
                        message="Production storage path exists but is not a directory!")
        else:
            # Check if files are present (for logging, but don't fail)
            docstore_path = prod_path / "docstore.json"
            has_files = docstore_path.exists()
            logger.info("rag_storage_path_prod_status",
                       prod_path=str(prod_path),
                       has_docstore=has_files,
                       message="Production storage path found" + 
                               (" with index files" if has_files else " but index files missing"))
        
        # ALWAYS return the prod path, even if it doesn't exist
        # This allows lazy initialization to attempt loading and provide better error messages
        return prod_path
    
    # Build candidate paths for dev/local environments
    # In dev, we require valid index files to exist
    candidates: list[Path] = [
        Path("latest_model").resolve(),                    # Current directory (make absolute)
        Path("../latest_model").resolve(),                 # Parent directory (for scripts/)
        Path("/workspace/latest_model"),                   # RunPod workspace
        Path("/workspace/ArrowSystems/latest_model"),     # RunPod with ArrowSystems
        Path("/workspace/storage"),                        # Old storage location
        Path("./storage").resolve(),                       # Local storage (make absolute)
    ]
    
    logger.info("rag_storage_path_dev_searching",
               candidates=[str(c) for c in candidates],
               env=settings.ENV)
    
    # Try each candidate path (in dev, require valid index files)
    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            # Verify it looks like a valid index (has docstore.json)
            docstore_path = candidate / "docstore.json"
            if docstore_path.exists():
                logger.info("rag_storage_path_dev_found",
                           path=str(candidate),
                           message="Found valid index in dev environment")
                return candidate
    
    # No valid index found in dev
    logger.warning("rag_storage_path_dev_not_found",
                 candidates=[str(c) for c in candidates],
                 message="No valid index found in dev environment. "
                        "RAG will be disabled until index is available.")
    return None

