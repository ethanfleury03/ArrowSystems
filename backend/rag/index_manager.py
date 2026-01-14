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
from backend.utils.resource_monitor import log_resource_checkpoint
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
        self._load_task: Optional[asyncio.Task] = None  # Global task for single-flight
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
        # Check for stuck states and auto-transition to error
        self._check_stuck_state()
        return {
            "status": self._status,
            "error": self._error,
            "started_at": self._started_at,
            "finished_at": self._finished_at,
            "elapsed_s": (self._finished_at - self._started_at) if (self._started_at and self._finished_at) else None,
        }
    
    def _check_stuck_state(self) -> None:
        """
        Check if state has been stuck in loading/downloading for too long.
        Auto-transition to 'failed' if stuck beyond timeout threshold.
        """
        if self._status not in ("loading", "not_started"):
            return
        
        if self._started_at is None:
            return
        
        # Get timeout from environment (default: 15 minutes)
        max_load_time = int(os.getenv("RAG_MAX_LOAD_TIME_SEC", "900"))  # 15 minutes default
        
        elapsed = time.time() - self._started_at
        if elapsed > max_load_time:
            error_msg = (
                f"Index loading timed out after {elapsed:.0f} seconds (max: {max_load_time}s). "
                f"This usually indicates a stuck download or index load operation. "
                f"Check GCS permissions, network connectivity, and file sizes."
            )
            logger.error(
                "rag_index_load_timeout_auto_fail",
                elapsed_seconds=elapsed,
                max_time_seconds=max_load_time,
                status=self._status,
                message=error_msg,
            )
            self._status = "failed"
            self._error = error_msg
            self._finished_at = time.time()
            self._ready_event.set()  # Unblock any waiters
    
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
        
        Uses a global task to ensure only one load attempt runs at a time.
        If a load is already in progress, waits for it instead of starting a new one.
        
        Args:
            force: If True, force reload even if already loaded or failed.
        
        Raises:
            RuntimeError: If loading fails (with error message).
        """
        # Fast path: already ready
        if self._status == "ready" and not force:
            return
        
        # Single-flight: If a load task is already running, wait for it
        if self._load_task is not None and not self._load_task.done():
            logger.info(
                "rag_index_load_waiting_existing_task",
                task_done=self._load_task.done(),
                message="Index load task already in progress, waiting for existing task..."
            )
            try:
                # Wait for the existing task with shield to prevent cancellation
                await asyncio.shield(self._load_task)
            except asyncio.CancelledError:
                # If we're cancelled, the task continues running (shield protects it)
                logger.warning(
                    "rag_index_load_wait_cancelled",
                    message="Wait for existing load task was cancelled, but task continues running"
                )
                raise
            # Check final status after task completes
            if self._status == "failed":
                raise RuntimeError(self._error or "Index loading failed")
            if self._status == "ready":
                return
        
        # If loading in progress (but no task - shouldn't happen, but handle defensively)
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
            
            # Check again if task was created while waiting for lock
            if self._load_task is not None and not self._load_task.done():
                logger.info(
                    "rag_index_load_waiting_after_lock",
                    message="Load task was created while waiting for lock, waiting for it..."
                )
                try:
                    await asyncio.shield(self._load_task)
                except asyncio.CancelledError:
                    logger.warning(
                        "rag_index_load_wait_cancelled_after_lock",
                        message="Wait for load task was cancelled after lock, but task continues"
                    )
                    raise
                if self._status == "failed":
                    raise RuntimeError(self._error or "Index loading failed")
                if self._status == "ready":
                    return
            
            if self._status == "loading":
                await self._ready_event.wait()
                if self._status == "failed":
                    raise RuntimeError(self._error or "Index loading failed")
                if self._status == "ready":
                    return
            
            # Determine trigger source for logging (before creating task)
            import traceback
            stack = traceback.extract_stack()
            trigger_source = "unknown"
            for frame in reversed(stack[-10:]):  # Check last 10 frames
                filename = frame.filename
                if "startup" in filename or "api.py" in filename:
                    if "startup_event" in str(frame):
                        trigger_source = "startup"
                        break
                    elif "/query" in str(frame) or "query_knowledge_base" in str(frame):
                        trigger_source = "/query"
                        break
                    elif "/rag/status" in str(frame) or "rag_status" in str(frame):
                        trigger_source = "/rag/status"
                        break
            
            # Reset state if forcing reload
            if force:
                self._ready_event.clear()
                self._error = None
                self._started_at = None
                self._finished_at = None
                # Cancel existing task if any
                if self._load_task is not None and not self._load_task.done():
                    logger.warning(
                        "rag_index_load_cancelling_existing_task",
                        message="Force reload requested, but existing task is running (will wait for it to complete first)"
                    )
                    # Don't cancel - wait for it to finish, then start new one
                    try:
                        await asyncio.shield(self._load_task)
                    except Exception:
                        pass
                    self._load_task = None
            
            # Start loading state
            load_start_time = time.time()
            self._status = "loading"
            self._started_at = load_start_time
            self._error = None
            self._ready_event.clear()
            # Resource checkpoint: loading started
            log_resource_checkpoint("rag_index_load_start")
            
            # Create the load task (single-flight: only one task exists at a time)
            async def _load_worker() -> None:
                """Internal worker that performs the actual load."""
                try:
                    await self._do_load(trigger_source)
                except Exception:
                    # Exceptions are handled in _do_load
                    raise
            
            # Create and store the global task
            self._load_task = asyncio.create_task(_load_worker())
            
            # Wait for the task to complete (with shield to prevent cancellation)
            try:
                await asyncio.shield(self._load_task)
            except asyncio.CancelledError:
                # If we're cancelled, the task continues running (shield protects it)
                logger.warning(
                    "rag_index_load_wait_cancelled_shield",
                    message="Wait for load task was cancelled, but task continues running (shield protected)"
                )
                raise
            
            # Check final status
            if self._status == "failed":
                raise RuntimeError(self._error or "Index loading failed")
            if self._status != "ready":
                raise RuntimeError(f"Index loading completed but status is {self._status}")
    
    async def _do_load(self, trigger_source: str = "unknown") -> None:
        """
        Internal method that performs the actual index load.
        
        This is separated from ensure_loaded to allow the task to run independently
        while ensure_loaded can wait for it with shield.
        
        Args:
            trigger_source: Source that triggered the load (for logging)
        """
        try:
            logger.info(
                "rag_index_load_start",
                status=self._status,
                started_at=self._started_at,
                trigger=trigger_source,
                message=f"Starting RAG index download and load (triggered by: {trigger_source})"
            )
            log_resource_checkpoint("rag_gcs_download_start")
            
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
            download_duration = None
            if settings.is_prod:
                # Check for missing files
                required = ["docstore.json", "index_store.json", "default__vector_store.json"]
                missing = []
                for f in required:
                    if not os.path.exists(os.path.join(storage_path, f)):
                        missing.append(f)
                if missing:
                    download_start = time.time()
                    logger.info(
                        "rag_index_download_needed",
                        missing_files=missing,
                        message=f"Missing {len(missing)} required index files, downloading from GCS"
                    )
                    from backend.rag.startup_downloader import download_index_from_gcs
                    download_ok = await asyncio.to_thread(download_index_from_gcs)
                    download_duration = time.time() - download_start
                    if not download_ok:
                        from backend.rag.startup_downloader import get_last_download_error
                        error_msg = get_last_download_error() or "Index download failed (unknown)"
                        raise RuntimeError(f"Index download failed: {error_msg}")
                    logger.info(
                        "rag_index_download_complete",
                        duration_seconds=download_duration,
                        message=f"Index download completed successfully in {download_duration:.2f}s"
                    )
                    log_resource_checkpoint("rag_gcs_download_complete")
                    
                    # Step 1b: Try to download optional ticket cache index (non-blocking)
                    try:
                        from backend.rag.ticket_index_downloader import download_ticket_index_from_gcs
                        ticket_download_ok = await asyncio.to_thread(download_ticket_index_from_gcs)
                        if ticket_download_ok:
                            logger.info("[TICKET] Ticket index downloaded successfully")
                        else:
                            logger.debug("[TICKET] Ticket index not available or download skipped")
                    except Exception as e:
                        logger.warning(
                            "[TICKET] Ticket index download failed - continuing without it",
                            error=str(e)
                        )
                    
                    # Validate downloaded files have non-trivial size
                    for f in required:
                        file_path = os.path.join(storage_path, f)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path)
                            if size <= 1024:  # 1KB threshold
                                raise RuntimeError(
                                    f"Downloaded file {f} is too small ({size} bytes). "
                                    f"Expected > 1KB. File may be corrupted or empty."
                                )
                            logger.info(
                                "rag_index_file_validated",
                                filename=f,
                                size_bytes=size,
                                message=f"Validated {f}: {size:,} bytes"
                            )
                else:
                    logger.info("rag_index_files_present", message="All required index files already present locally")
                    
                    # Validate existing files have non-trivial size
                    for f in required:
                        file_path = os.path.join(storage_path, f)
                        if os.path.exists(file_path):
                            size = os.path.getsize(file_path)
                            if size <= 1024:
                                raise RuntimeError(
                                    f"Existing file {f} is too small ({size} bytes). "
                                    f"Expected > 1KB. File may be corrupted."
                                    )
            
            # Step 2: Load index into pipeline
            # Set phase to "loading" before starting pipeline load
            from backend.rag.index_state import set_phase
            set_phase("loading", local_dir=storage_path)
            
            load_start_time = time.time()
            logger.info("rag_index_load_pipeline_start", message="Loading index into RAG pipeline")
            log_resource_checkpoint("rag_index_load_pipeline_start")
            
            # Checkpoint: Validate required files exist before load
            required_files = ["docstore.json", "index_store.json", "default__vector_store.json"]
            logger.info(
                "rag_index_load_validate_files_start",
                storage_dir=storage_path,
                required_files=required_files,
                message="Validating required index files before load"
            )
            
            missing_files = []
            file_sizes = {}
            for f in required_files:
                file_path = os.path.join(storage_path, f)
                if os.path.exists(file_path):
                    size = os.path.getsize(file_path)
                    file_sizes[f] = size
                    if size <= 1024:
                        missing_files.append(f"{f} (too small: {size} bytes)")
                else:
                    missing_files.append(f"{f} (missing)")
            
            if missing_files:
                error_msg = f"Missing or invalid required files: {', '.join(missing_files)}"
                logger.error(
                    "rag_index_load_validate_files_failed",
                    storage_dir=storage_path,
                    missing_files=missing_files,
                    message=error_msg
                )
                raise RuntimeError(error_msg)
            
            logger.info(
                "rag_index_load_validate_files_complete",
                storage_dir=storage_path,
                file_sizes=file_sizes,
                message=f"All required files validated: {', '.join([f'{f}={file_sizes[f]:,} bytes' for f in required_files])}"
            )
            
            # Checkpoint: About to read/parse docstore.json
            docstore_path = os.path.join(storage_path, "docstore.json")
            logger.info(
                "rag_index_load_docstore_start",
                docstore_path=docstore_path,
                docstore_size_bytes=file_sizes.get("docstore.json", 0),
                message="Starting docstore.json read/parse"
            )
            
            from backend.rag_pipeline import get_rag_pipeline
            from backend.api import get_db_manager_instance
            
            db_manager = get_db_manager_instance()
            cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface')
            
            # Get or create global pipeline instance
            pipeline = get_rag_pipeline(cache_dir=cache_dir, db_manager=db_manager, storage_dir=storage_path)
            
            # Checkpoint: About to read/parse vector store
            vector_store_path = os.path.join(storage_path, "default__vector_store.json")
            logger.info(
                "rag_index_load_vector_store_start",
                vector_store_path=vector_store_path,
                vector_store_size_bytes=file_sizes.get("default__vector_store.json", 0),
                message="Starting default__vector_store.json read/parse"
            )
            
            # CRITICAL: Move blocking pipeline.ensure_initialized() to thread pool
            # This prevents blocking the event loop and making HTTP endpoints unresponsive
            logger.info(
                "rag_index_load_pipeline_init_start",
                storage_dir=storage_path,
                message="Starting pipeline.ensure_initialized() in thread pool (non-blocking)"
            )
            
            # Get timeout from environment (default: 10 minutes)
            max_load_time = int(os.getenv("RAG_MAX_LOAD_TIME_SEC", "600"))  # 10 minutes default
            
            try:
                # Run the blocking initialization in a thread pool
                initialized = await asyncio.wait_for(
                    asyncio.to_thread(pipeline.ensure_initialized, storage_path),
                    timeout=max_load_time
                )
            except asyncio.TimeoutError:
                elapsed = time.time() - load_start_time
                error_msg = (
                    f"Pipeline initialization timed out after {elapsed:.0f} seconds (max: {max_load_time}s). "
                    f"This usually indicates a stuck index load operation. "
                    f"Check file sizes, disk I/O, and memory usage."
                )
                logger.error(
                    "rag_index_load_pipeline_timeout",
                    elapsed_seconds=elapsed,
                    max_time_seconds=max_load_time,
                    storage_dir=storage_path,
                    message=error_msg
                )
                raise RuntimeError(error_msg)
            
            load_duration = time.time() - load_start_time
            
            logger.info(
                "rag_index_load_pipeline_init_complete",
                duration_seconds=load_duration,
                initialized=initialized,
                message=f"Pipeline.ensure_initialized() completed in {load_duration:.2f}s"
            )
            
            if not initialized:
                error_msg = pipeline.debug_status().get("last_error") or "Pipeline initialization failed"
                logger.error(
                    "rag_index_load_pipeline_init_failed",
                    error=error_msg,
                    message=f"Pipeline initialization returned False: {error_msg}"
                )
                raise RuntimeError(f"Pipeline initialization failed: {error_msg}")
            
            # Checkpoint: Verify pipeline is actually ready
            logger.info(
                "rag_index_load_pipeline_verify_start",
                message="Verifying pipeline is initialized"
            )
            
            if not pipeline.is_initialized():
                error_msg = "Pipeline initialization completed but is_initialized() returned False"
                logger.error(
                    "rag_index_load_pipeline_verify_failed",
                    error=error_msg,
                    message=error_msg
                )
                raise RuntimeError(error_msg)
            
            logger.info(
                "rag_index_load_pipeline_verify_complete",
                message="Pipeline verified as initialized"
            )
            
            # Checkpoint: Finalize pipeline (build vector store object)
            logger.info(
                "rag_index_load_pipeline_finalize_start",
                message="Finalizing pipeline (building vector store object)"
            )
            
            # Pipeline is ready - log completion
            logger.info(
                "rag_index_pipeline_load_complete",
                duration_seconds=load_duration,
                message=f"Pipeline index load completed in {load_duration:.2f}s"
            )
            log_resource_checkpoint("rag_index_loaded")
            
            # Log sample metadata keys for compatibility checking
            try:
                from backend.rag_pipeline import get_rag_pipeline
                loaded_pipeline = get_rag_pipeline()
                if loaded_pipeline and loaded_pipeline.is_initialized():
                    orchestrator = loaded_pipeline.orchestrator
                    if orchestrator and orchestrator.index:
                        # Try to get a sample node to check metadata keys
                        try:
                            docstore = orchestrator.index.storage_context.docstore
                            if docstore:
                                # Get first node ID from docstore
                                all_doc_ids = list(docstore.docs.keys())
                                if all_doc_ids:
                                    sample_id = all_doc_ids[0]
                                    sample_node = docstore.get_document(sample_id)
                                    if sample_node and hasattr(sample_node, 'metadata'):
                                        meta_keys = list(sample_node.metadata.keys()) if sample_node.metadata else []
                                        logger.info(
                                            "rag_index_metadata_sample",
                                            sample_node_id=sample_id,
                                            metadata_keys=meta_keys,
                                            metadata_keys_count=len(meta_keys),
                                            message=f"Sample node metadata keys: {meta_keys}"
                                        )
                        except Exception as meta_check_error:
                            logger.warning(
                                "rag_index_metadata_check_failed",
                                error=str(meta_check_error),
                                message="Could not check sample metadata keys (non-fatal)"
                            )
            except Exception:
                pass  # Non-fatal - just logging
            
            # Success! Update phase to "ready"
            try:
                from backend.rag.index_state import set_phase
                set_phase("ready", local_dir=storage_path)
            except Exception:
                pass  # Non-fatal
            
            self._status = "ready"
            self._finished_at = time.time()
            self._error = None
            total_elapsed = self._finished_at - self._started_at
            
            # Build message with timing breakdown
            timing_parts = []
            if download_duration is not None:
                timing_parts.append(f"download: {download_duration:.2f}s")
            if 'load_duration' in locals():
                timing_parts.append(f"load: {load_duration:.2f}s")
            timing_msg = f" ({', '.join(timing_parts)})" if timing_parts else ""
            
            logger.info(
                "rag_index_load_done",
                status=self._status,
                total_elapsed_s=total_elapsed,
                download_duration_s=download_duration,
                load_duration_s=load_duration if 'load_duration' in locals() else None,
                started_at=self._started_at,
                finished_at=self._finished_at,
                trigger=trigger_source,
                message=f"RAG index load completed successfully in {total_elapsed:.2f}s{timing_msg} (triggered by: {trigger_source})"
            )
            log_resource_checkpoint("rag_bg_task_success")
            
        except asyncio.CancelledError:
            # If cancelled (shouldn't happen with shield, but handle defensively)
            # Keep state as "loading" and let task continue (don't corrupt state)
            logger.warning(
                "rag_index_load_cancelled",
                message="Index load was cancelled (should not happen with shield). Keeping state as loading."
            )
            # Don't reset status - keep it as "loading" so the task can continue
            # Don't set finished_at - task is still running
            self._ready_event.set()  # Unblock waiters
            raise  # Re-raise to propagate cancellation
        except Exception as e:
            self._status = "failed"
            self._finished_at = time.time()
            # Store full exception details including traceback
            import traceback
            error_traceback = traceback.format_exc()
            # Store error as string (extract key message, truncate traceback if too long)
            error_str = str(e)
            if len(error_traceback) > 2000:
                error_traceback = error_traceback[:2000] + "... (truncated)"
            self._error = f"{type(e).__name__}: {error_str}"
            elapsed = (self._finished_at - self._started_at) if self._started_at else None
            
            # Update phase to error
            try:
                from backend.rag.index_state import set_phase
                set_phase("error", error=self._error)
            except Exception:
                pass
            
            elapsed_str = f"{elapsed:.2f}s" if elapsed else "unknown"
            logger.error(
                "rag_index_load_failed",
                status=self._status,
                error_type=type(e).__name__,
                error_message=error_str,
                elapsed_s=elapsed,
                trigger=trigger_source,
                traceback=error_traceback,
                exc_info=True,
                message=f"RAG index load failed after {elapsed_str}: {type(e).__name__}: {error_str} (triggered by: {trigger_source})"
            )
            log_resource_checkpoint("rag_bg_task_failed")
            raise RuntimeError(self._error) from e
        
        finally:
                # CRITICAL: Always log final state in finally block (runs even on exception/cancellation)
                try:
                    from backend.rag.index_state import get_index_state
                    index_state = get_index_state()
                    current_phase = index_state.get("phase", "unknown")
                except Exception:
                    current_phase = "unknown"
                
                logger.info(
                    "rag_index_load_finally",
                    status=self._status,
                    phase=current_phase,
                    error=self._error,
                    started_at=self._started_at,
                    finished_at=self._finished_at,
                    elapsed_s=(self._finished_at - self._started_at) if (self._started_at and self._finished_at) else None,
                    message=f"Index load finally block: status={self._status}, phase={current_phase}, error={self._error or 'none'}"
                )
                
                # Always set event to unblock waiters
                self._ready_event.set()
                
                # Clear the global task reference
                self._load_task = None


# Global singleton instance
_index_load_state: Optional[IndexLoadState] = None


def get_index_load_state() -> IndexLoadState:
    """Get the global IndexLoadState singleton."""
    global _index_load_state
    if _index_load_state is None:
        _index_load_state = IndexLoadState()
    return _index_load_state

