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
    2. In prod: settings.RAG_INDEX_LOCAL_DIR (Cloud Run-safe local dir) - ALWAYS returned in prod
    3. Dev/local paths: latest_model, ../latest_model, /workspace/*, etc.
    
    Returns:
        Path to index directory, or None if not found
        
    Note: In production, this ALWAYS returns settings.RAG_INDEX_LOCAL_DIR (even if directory doesn't exist).
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
    
    # In production, use the configured local directory for index artifacts.
    # On Cloud Run, this should be a writable path (prefer /tmp/latest_model).
    # ALWAYS return it in prod, even if directory doesn't exist (download will create it).
    if settings.ENV in ("prod", "production", "cloud"):
        configured = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
        prod_path = Path(configured).resolve()  # Ensure absolute
        exists = prod_path.exists()
        is_dir = prod_path.is_dir() if exists else False

        # Ensure directory is writable; Cloud Run image filesystem may be read-only.
        # If not writable, fall back to /tmp/latest_model.
        try:
            prod_path.mkdir(parents=True, exist_ok=True)
            test_path = prod_path / ".write_test"
            test_path.write_text("ok", encoding="utf-8")
            test_path.unlink(missing_ok=True)
        except Exception as e:
            fallback = Path("/tmp/latest_model").resolve()
            logger.warning(
                "rag_storage_path_prod_not_writable",
                requested=str(prod_path),
                fallback=str(fallback),
                error=str(e),
                message="Production storage path not writable; falling back to /tmp/latest_model",
            )
            prod_path = fallback
            exists = prod_path.exists()
            is_dir = prod_path.is_dir() if exists else False
            prod_path.mkdir(parents=True, exist_ok=True)
        
        logger.info("[storage] Using production storage directory",
                   prod_path=str(prod_path),
                   exists=exists,
                   is_dir=is_dir,
                   env=settings.ENV,
                   message="Production storage path resolved - index will be downloaded from GCS on startup")
        
        if exists and not is_dir:
            logger.error("rag_storage_path_prod_not_directory",
                        prod_path=str(prod_path),
                        message="Production storage path exists but is not a directory!")
        elif exists:
            # Check if files are present (for logging, but don't fail)
            docstore_path = prod_path / "docstore.json"
            has_files = docstore_path.exists()
            logger.info("rag_storage_path_prod_status",
                       prod_path=str(prod_path),
                       has_docstore=has_files,
                       message="Production storage path found" + 
                               (" with index files" if has_files else " (files will be downloaded on startup)"))
        
        # ALWAYS return the prod path, even if it doesn't exist
        # The download logic will create the directory and download files before loading
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

