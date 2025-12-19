"""
RAG Index Load Manager - Cloud Run-safe deterministic index loading.

Manages the lifecycle of RAG index download and loading with proper state tracking,
singleflight semantics, and Cloud Run-safe execution.
"""

import os
import asyncio
import time
import threading
from typing import Optional, Dict, Any
from pathlib import Path

from backend.logging_config import get_logger
from backend.config.env import settings

logger = get_logger(__name__)


class IndexLoadState:
    """
    Singleton state manager for RAG index loading.
    
    Ensures only one load attempt runs at a time (singleflight) and provides
    deterministic state tracking for Cloud Run environments.
    """
    
    _instance: Optional['IndexLoadState'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if hasattr(self, '_initialized') and self._initialized:
            return
        
        self._lock = asyncio.Lock()
        self._ready_event = asyncio.Event()
        self._status: str = "not_started"  # "not_started" | "loading" | "ready" | "failed"
        self._error: Optional[str] = None
        self._started_at: Optional[float] = None
        self._finished_at: Optional[float] = None
        self._initialized = True
    
    @property
    def status(self) -> str:
        """Get current load status."""
        return self._status
    
    @property
    def error(self) -> Optional[str]:
        """Get error message if status is 'failed'."""
        return self._error
    
    @property
    def started_at(self) -> Optional[float]:
        """Get timestamp when loading started."""
        return self._started_at
    
    @property
    def finished_at(self) -> Optional[float]:
        """Get timestamp when loading finished."""
        return self._finished_at
    
    def get_state(self) -> Dict[str, Any]:
        """Get full state dictionary."""
        return {
            "status": self._status,
            "error": self._error,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "elapsed_s": (self._finished_at - self._started_at) if (self._started_at and self._finished_at) else None,
        }
    
    async def wait_for_ready(self, timeout: Optional[float] = None) -> bool:
        """
        Wait for index to be ready.
        
        Args:
            timeout: Maximum time to wait in seconds. None means wait indefinitely.
        
        Returns:
            True if ready, False if timeout or failed.
        """
        if self._status == "ready":
            return True
        
        try:
            if timeout is not None:
                await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            else:
                await self._ready_event.wait()
            
            return self._status == "ready"
        except asyncio.TimeoutError:
            return False
    
    async def ensure_loaded(self, force: bool = False) -> None:
        """
        Ensure index is loaded, with singleflight semantics.
        
        Args:
            force: If True, force reload even if already loaded or failed.
        
        Raises:
            RuntimeError: If loading fails (with error message).
        """
        # Fast path: already ready
        if self._status == "ready" and not force:
            return
        
        # If loading in progress, wait for it
        if self._status == "loading":
            logger.info("rag_index_load_waiting", message="Index load already in progress, waiting...")
            await self._ready_event.wait()
            if self._status == "failed":
                raise RuntimeError(self._error or "Index loading failed")
            if self._status == "ready":
                return
        
        # Acquire lock to ensure only one load attempt
        async with self._lock:
            # Double-check after acquiring lock
            if self._status == "ready" and not force:
                return
            if self._status == "loading":
                await self._ready_event.wait()
                if self._status == "failed":
                    raise RuntimeError(self._error or "Index loading failed")
                if self._status == "ready":
                    return
            
            # Reset state if forcing reload
            if force:
                self._ready_event.clear()
                self._error = None
                self._started_at = None
                self._finished_at = None
            
            # Start loading
            self._status = "loading"
            self._started_at = time.time()
            self._error = None
            self._ready_event.clear()
            
            try:
                logger.info(
                    "rag_index_load_start",
                    status=self._status,
                    started_at=self._started_at,
                    message="Starting RAG index download and load"
                )
                
                # Resolve storage path
                from backend.utils.storage_path import resolve_storage_path
                from backend.utils.test_mode import is_test_mode, get_index_dir
                
                if is_test_mode():
                    storage_path = get_index_dir()
                else:
                    storage_path_obj = resolve_storage_path()
                    if storage_path_obj is None:
                        storage_path = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
                    else:
                        storage_path = str(storage_path_obj.resolve())
                
                bucket_name = getattr(settings, "RAG_INDEX_GCS_BUCKET", "arrow-rag-support-prod-rag")
                index_prefix = getattr(settings, "RAG_INDEX_GCS_PREFIX", "latest_model/")
                
                logger.info(
                    "rag_index_load_config",
                    bucket=bucket_name,
                    prefix=index_prefix,
                    local_dir=storage_path,
                    message=f"Index load config: gs://{bucket_name}/{index_prefix} -> {storage_path}"
                )
                
                # Step 1: Download index from GCS (if in production and files missing)
                if settings.is_prod:
                    # Check for missing files
                    required = ["docstore.json", "index_store.json", "default__vector_store.json"]
                    missing = []
                    for f in required:
                        if not os.path.exists(os.path.join(storage_path, f)):
                            missing.append(f)
                    if missing:
                        logger.info(
                            "rag_index_download_needed",
                            missing_files=missing,
                            message=f"Missing {len(missing)} required index files, downloading from GCS"
                        )
                        from backend.rag.startup_downloader import download_index_from_gcs
                        download_ok = await asyncio.to_thread(download_index_from_gcs)
                        if not download_ok:
                            from backend.rag.startup_downloader import get_last_download_error
                            error_msg = get_last_download_error() or "Index download failed (unknown)"
                            raise RuntimeError(f"Index download failed: {error_msg}")
                        logger.info("rag_index_download_complete", message="Index download completed successfully")
                    else:
                        logger.info("rag_index_files_present", message="All required index files already present locally")
                
                # Step 2: Load index into pipeline
                logger.info("rag_index_load_pipeline_start", message="Loading index into RAG pipeline")
                from backend.rag_pipeline import get_rag_pipeline
                from backend.utils.database_manager import get_db_manager_instance
                
                db_manager = get_db_manager_instance()
                cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface')
                
                # Get or create global pipeline instance
                pipeline = get_rag_pipeline(cache_dir=cache_dir, db_manager=db_manager, storage_dir=storage_path)
                
                # Initialize pipeline (this loads models and index)
                # This is a blocking call that will download models if needed and load the index
                initialized = pipeline.ensure_initialized(storage_path)
                if not initialized:
                    error_msg = pipeline.debug_status().get("last_error") or "Pipeline initialization failed"
                    raise RuntimeError(f"Pipeline initialization failed: {error_msg}")
                
                # Verify pipeline is actually ready
                if not pipeline.is_initialized():
                    raise RuntimeError("Pipeline initialization completed but is_initialized() returned False")
                
                # Success!
                self._status = "ready"
                self._finished_at = time.time()
                self._error = None
                elapsed = self._finished_at - self._started_at
                
                logger.info(
                    "rag_index_load_done",
                    status=self._status,
                    elapsed_s=elapsed,
                    started_at=self._started_at,
                    finished_at=self._finished_at,
                    message=f"RAG index load completed successfully in {elapsed:.2f}s"
                )
                
            except Exception as e:
                self._status = "failed"
                self._finished_at = time.time()
                self._error = f"{type(e).__name__}: {str(e)}"
                elapsed = (self._finished_at - self._started_at) if self._started_at else None
                
                elapsed_str = f"{elapsed:.2f}s" if elapsed else "unknown"
                logger.error(
                    "rag_index_load_failed",
                    status=self._status,
                    error=self._error,
                    elapsed_s=elapsed,
                    exc_info=True,
                    message=f"RAG index load failed after {elapsed_str}"
                )
                raise RuntimeError(self._error) from e
            
            finally:
                # Always set event to unblock waiters
                self._ready_event.set()


# Global singleton instance
_index_load_state: Optional[IndexLoadState] = None


def get_index_load_state() -> IndexLoadState:
    """Get the global IndexLoadState singleton."""
    global _index_load_state
    if _index_load_state is None:
        _index_load_state = IndexLoadState()
    return _index_load_state

