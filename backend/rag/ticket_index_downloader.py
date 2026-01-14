"""
Optional ticket cache index downloader for production environments.

Downloads ticket cache index artifacts from GCS if they exist.
This is OPTIONAL - if the index doesn't exist, the system continues normally.

Thread-safe and idempotent: uses lockfile to prevent concurrent downloads.
"""

import os
import time
from pathlib import Path
from typing import Optional

from backend.config.env import settings
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Fixed GCS prefix for ticket cache index
TICKET_INDEX_GCS_PREFIX = "ticket_cache/latest_model/"

# Required files for ticket index (must match main index pattern)
REQUIRED_TICKET_FILES = [
    "docstore.json",
    "index_store.json",
    "default__vector_store.json",
]

# Last download error for observability
_last_ticket_download_error: Optional[str] = None


def get_last_ticket_download_error() -> Optional[str]:
    """Get last ticket index download error."""
    return _last_ticket_download_error


def _is_cloud_run() -> bool:
    """Check if running on Cloud Run."""
    return bool(os.getenv("K_SERVICE") or os.getenv("K_REVISION"))


def is_valid_ticket_index_dir(dir_path: Path) -> bool:
    """
    Check if a directory contains a valid ticket index.
    
    Validates that all required files exist and are non-empty (>= 1024 bytes).
    Uses same pattern as main index validity checks.
    
    Args:
        dir_path: Path to directory to check
        
    Returns:
        True if directory contains valid ticket index, False otherwise
    """
    if not dir_path or not dir_path.exists() or not dir_path.is_dir():
        return False
    
    try:
        for filename in REQUIRED_TICKET_FILES:
            file_path = dir_path / filename
            if not file_path.exists():
                return False
            # Check file size (must be non-empty, >= 1024 bytes)
            if file_path.stat().st_size < 1024:
                return False
        return True
    except Exception as e:
        logger.debug("[TICKET] Error validating ticket index dir", dir=str(dir_path), error=str(e))
        return False


def _probe_writable_dir(dir_path: Path) -> bool:
    """
    Probe if a directory is writable by attempting to create a test file.
    
    Args:
        dir_path: Path to directory to test
        
    Returns:
        True if directory is writable, False otherwise
    """
    try:
        test_file = dir_path / f".write_test.{os.getpid()}"
        test_file.write_text("test")
        test_file.unlink()
        return True
    except Exception:
        return False


def _get_ticket_index_local_dir() -> Path:
    """
    Determine the local directory path for ticket index.
    
    Prefers /tmp/ticket_cache_model on Cloud Run.
    Falls back to parent of main index dir if writable, otherwise /tmp.
    
    Returns:
        Path to ticket index directory
    """
    # Prefer /tmp/ticket_cache_model on Cloud Run (always writable)
    if _is_cloud_run():
        return Path("/tmp/ticket_cache_model")
    
    # Try parent of main index dir if available
    main_local_dir = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
    main_dir_path = Path(main_local_dir)
    
    if main_dir_path.exists() and main_dir_path.parent.exists():
        candidate_dir = main_dir_path.parent / "ticket_cache_model"
        # Probe writability
        if _probe_writable_dir(main_dir_path.parent):
            return candidate_dir
    
    # Fallback to /tmp
    return Path("/tmp/ticket_cache_model")


def _normalize_prefix(prefix: str) -> str:
    """Normalize GCS prefix to end with /."""
    p = prefix.strip()
    if not p:
        return ""
    p = p.strip("/")
    return f"{p}/" if p else ""


def check_ticket_index_exists(bucket_name: str, prefix: str) -> bool:
    """
    Check if ticket index exists in GCS.
    
    Returns True if prefix contains at least one object other than .keep.
    Returns False if prefix is empty or only contains .keep.
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.debug("[TICKET] google-cloud-storage not installed - cannot check ticket index existence")
        return False
    
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # List objects under prefix
        blobs = list(bucket.list_blobs(prefix=prefix, max_results=10))
        
        # Filter out .keep files
        real_files = [b for b in blobs if not b.name.endswith(".keep")]
        
        exists = len(real_files) > 0
        
        if exists:
            logger.info(
                "[TICKET] Ticket index exists in GCS",
                bucket=bucket_name,
                prefix=prefix,
                file_count=len(real_files),
                sample_files=[b.name for b in real_files[:3]]
            )
        else:
            logger.debug(
                "[TICKET] Ticket index not found in GCS (prefix empty or only .keep)",
                bucket=bucket_name,
                prefix=prefix,
                total_blobs=len(blobs)
            )
        
        return exists
    except Exception as e:
        logger.warning(
            "[TICKET] Failed to check ticket index existence",
            bucket=bucket_name,
            prefix=prefix,
            error=str(e)
        )
        return False


def download_ticket_index_from_gcs() -> bool:
    """
    Download ticket cache index files from GCS into local directory.
    
    This is OPTIONAL - if the index doesn't exist, returns False without error.
    Thread-safe and idempotent: uses lockfile and checks for existing valid index.
    
    Source: gs://<RAG_INDEX_GCS_BUCKET>/ticket_cache/latest_model/
    Local:  /tmp/ticket_cache_model (or fallback based on writability)
    
    Returns:
        True if download succeeded or already exists, False if index doesn't exist or download failed
    """
    global _last_ticket_download_error
    _last_ticket_download_error = None
    
    try:
        from google.cloud import storage
    except ImportError:
        logger.debug("[TICKET] google-cloud-storage not installed - skipping ticket index download")
        return False
    
    bucket_name = settings.RAG_INDEX_GCS_BUCKET
    ticket_prefix = _normalize_prefix(TICKET_INDEX_GCS_PREFIX)
    
    # Determine local directory (with writability check)
    ticket_local_dir = _get_ticket_index_local_dir()
    
    # Idempotency check: if already downloaded and valid, skip
    if is_valid_ticket_index_dir(ticket_local_dir):
        logger.info(
            "[TICKET] Ticket index already downloaded - skipping",
            local_dir=str(ticket_local_dir)
        )
        return True
    
    # Check if ticket index exists in GCS
    if not check_ticket_index_exists(bucket_name, ticket_prefix):
        logger.info(
            "[TICKET] Ticket index not present in GCS - continuing without it",
            bucket=bucket_name,
            prefix=ticket_prefix
        )
        return False
    
    # File lock to ensure only one worker downloads at a time (multi-worker gunicorn)
    lock_path = Path("/tmp/ticket_index_download.lock")
    lock_file = None
    try:
        try:
            # Best-effort file lock using fcntl (Linux/Cloud Run)
            import fcntl  # type: ignore
            lock_file = open(lock_path, "w+")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logger.info("[TICKET] Acquired download file lock", lock=str(lock_path))
        except Exception:
            # Fallback: create lock file exclusively; if cannot acquire, wait/poll
            acquired = False
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                acquired = True
                logger.info("[TICKET] Created lock file (no fcntl)", lock=str(lock_path))
            except Exception:
                logger.warning("[TICKET] Could not acquire lock file; entering wait loop", lock=str(lock_path))
            if not acquired:
                # Wait until either lock disappears or final dir becomes valid
                WAIT_TOTAL = int(os.getenv("TICKET_DOWNLOAD_LOCK_WAIT_SEC", "600"))
                WAIT_STEP = int(os.getenv("TICKET_DOWNLOAD_LOCK_WAIT_STEP_SEC", "2"))
                waited = 0
                while waited < WAIT_TOTAL:
                    # Check if final dir is valid (another worker may have finished)
                    if is_valid_ticket_index_dir(ticket_local_dir):
                        logger.info("[TICKET] Detected valid index during lock wait; skipping download", final_dir=str(ticket_local_dir))
                        return True
                    # If lock file disappeared, break and attempt normal path
                    if not lock_path.exists():
                        break
                    time.sleep(WAIT_STEP)
                    waited += WAIT_STEP
                # Try to create the lock again after waiting
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    os.close(fd)
                    logger.info("[TICKET] Lock file acquired after waiting", lock=str(lock_path))
                except Exception:
                    # As a last resort, if final dir is still invalid and cannot acquire lock, fail fast
                    logger.error("[TICKET] Failed to acquire download lock and final dir invalid; aborting download to avoid races", lock=str(lock_path))
                    _last_ticket_download_error = "Unable to acquire download lock; another worker may be initializing"
                    return False
        
        # Re-check validity after acquiring lock (another worker may have finished)
        if is_valid_ticket_index_dir(ticket_local_dir):
            logger.info("[TICKET] Ticket index became valid while acquiring lock; skipping download", local_dir=str(ticket_local_dir))
            return True
        
        # Use atomic temp directory for download
        tmp_parent = ticket_local_dir.parent
        tmp_dir = tmp_parent / f"{ticket_local_dir.name}.tmp.{os.getpid()}"
        
        # Ensure tmp dir is clean
        try:
            if tmp_dir.exists():
                import shutil
                shutil.rmtree(tmp_dir, ignore_errors=True)
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("[TICKET] Failed to prepare temp dir for atomic download", tmp_dir=str(tmp_dir), error=str(e))
            _last_ticket_download_error = f"Temp dir prepare failed: {type(e).__name__}: {e}"
            return False
        
        logger.info(
            "[TICKET] Starting ticket index download",
            bucket=bucket_name,
            prefix=ticket_prefix,
            local_dir=str(ticket_local_dir)
        )
        print(f"[TICKET] Starting GCS ticket index download from gs://{bucket_name}/{ticket_prefix} to {str(ticket_local_dir)}...", flush=True)
        
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        
        # Download required files
        downloaded = 0
        for filename in REQUIRED_TICKET_FILES:
            blob_name = f"{ticket_prefix}{filename}"
            blob = bucket.blob(blob_name)
            
            if not blob.exists():
                error_msg = f"Required file {filename} not found in ticket index"
                logger.error("[TICKET] " + error_msg, blob_name=blob_name)
                _last_ticket_download_error = error_msg
                # Clean up temp dir
                import shutil
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return False
            
            local_path = tmp_dir / filename
            blob.download_to_filename(str(local_path))
            
            # Validate file size
            if local_path.stat().st_size < 1024:
                error_msg = f"Downloaded file {filename} is too small ({local_path.stat().st_size} bytes)"
                logger.error("[TICKET] " + error_msg)
                _last_ticket_download_error = error_msg
                import shutil
                if tmp_dir.exists():
                    shutil.rmtree(tmp_dir, ignore_errors=True)
                return False
            
            downloaded += 1
            logger.debug(f"[TICKET] Downloaded {filename} ({local_path.stat().st_size:,} bytes)")
        
        # Download optional files (index_manifest.json, etc.)
        optional_files = ["index_manifest.json"]
        for filename in optional_files:
            blob_name = f"{ticket_prefix}{filename}"
            blob = bucket.blob(blob_name)
            if blob.exists():
                local_path = tmp_dir / filename
                blob.download_to_filename(str(local_path))
                logger.debug(f"[TICKET] Downloaded optional {filename}")
        
        # Atomic rename: temp -> final
        if ticket_local_dir.exists():
            import shutil
            shutil.rmtree(ticket_local_dir, ignore_errors=True)
        
        tmp_dir.rename(ticket_local_dir)
        
        logger.info(
            "[TICKET] Ticket index download complete",
            bucket=bucket_name,
            prefix=ticket_prefix,
            local_dir=str(ticket_local_dir),
            files_downloaded=downloaded
        )
        print(f"[TICKET] ✅ Ticket index downloaded successfully to {str(ticket_local_dir)}", flush=True)
        
        return True
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.warning(
            "[TICKET] Ticket index download failed - continuing without it",
            bucket=bucket_name,
            prefix=ticket_prefix,
            error=error_msg,
            exc_info=True
        )
        _last_ticket_download_error = error_msg
        return False
    finally:
        # Release lock file (if fcntl was used)
        if lock_file:
            try:
                import fcntl  # type: ignore
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            try:
                lock_file.close()
            except Exception:
                pass
        # Clean up lock file (for fallback lock mechanism)
        try:
            if lock_path.exists():
                lock_path.unlink()
        except Exception:
            pass


def get_ticket_index_local_dir() -> Optional[Path]:
    """
    Get the local directory path for ticket index if it exists and is valid.
    
    Returns:
        Path to ticket index directory if valid, None otherwise
    """
    ticket_local_dir = _get_ticket_index_local_dir()
    
    if is_valid_ticket_index_dir(ticket_local_dir):
        return ticket_local_dir
    
    return None
