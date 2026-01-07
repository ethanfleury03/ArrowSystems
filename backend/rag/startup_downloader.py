"""
RAG index downloader for production environments.

Downloads RAG index artifacts from GCS into a local directory so the RAG pipeline can load them.

Key behaviors:
- Bucket/prefix/local dir are configurable via env/settings (no hardcoding).
- On Cloud Run, defaults to a writable local directory (prefer /tmp/latest_model).
- If no objects exist under the configured prefix, we fall back to bucket root for known filenames.
"""

import os
import time
import threading
from pathlib import Path
from typing import Optional

from backend.config.env import settings
from backend.logging_config import get_logger
from backend.rag.index_state import (
    set_phase, reset_state, init_file_tracking,
    update_file_start, update_file_success, update_file_error
)
from backend.utils.resource_monitor import log_resource_checkpoint

logger = get_logger(__name__)

REQUIRED_FILES = [
    "docstore.json",
    "index_store.json",
    "default__vector_store.json",
]

OPTIONAL_FILES = [
    "graph_store.json",
    "image__vector_store.json",
]

FALLBACK_ROOT_FILENAMES = REQUIRED_FILES + OPTIONAL_FILES

# Last download error for observability (surfaced via /rag/status and /query errors)
_last_download_error: Optional[str] = None


def get_last_download_error() -> Optional[str]:
    return _last_download_error


def _is_cloud_run() -> bool:
    return bool(os.getenv("K_SERVICE") or os.getenv("K_REVISION"))


def _normalize_prefix(prefix: Optional[str]) -> str:
    p = (prefix or "").strip()
    if not p:
        return ""
    p = p.strip("/")
    return f"{p}/" if p else ""


def _resolve_local_dir(requested_local_dir: str, gcs_prefix: str) -> Path:
    """
    Resolve the local directory path, avoiding double-prefix bugs.
    
    If RAG_INDEX_LOCAL_DIR already ends with the final segment of RAG_INDEX_GCS_PREFIX,
    do NOT append it again.
    
    Example:
    - RAG_INDEX_LOCAL_DIR=/tmp/latest_model, RAG_INDEX_GCS_PREFIX=latest_model/ -> /tmp/latest_model
    - RAG_INDEX_LOCAL_DIR=/tmp, RAG_INDEX_GCS_PREFIX=latest_model/ -> /tmp/latest_model
    """
    requested = Path(requested_local_dir).resolve()
    
    # Extract the final segment from GCS prefix (e.g., "latest_model" from "latest_model/")
    prefix_segment = gcs_prefix.rstrip("/").split("/")[-1] if gcs_prefix else None
    
    # If requested dir already ends with the prefix segment, use it as-is
    if prefix_segment and requested.name == prefix_segment:
        resolved = requested
    elif prefix_segment:
        # Append the prefix segment
        resolved = requested / prefix_segment
    else:
        resolved = requested
    
    return _ensure_writable_dir(str(resolved))


def _ensure_writable_dir(local_dir: str) -> Path:
    """
    Ensure local_dir exists and is writable. If not, fall back to /tmp/latest_model.
    """
    candidate = Path(local_dir).resolve()
    try:
        candidate.mkdir(parents=True, exist_ok=True)
        test_path = candidate / ".write_test"
        test_path.write_text("ok", encoding="utf-8")
        test_path.unlink(missing_ok=True)
        return candidate
    except Exception as e:
        fallback = Path("/tmp/latest_model").resolve()
        logger.warning(
            "[RAG] Local index dir not writable; falling back to /tmp/latest_model",
            requested_dir=str(candidate),
            fallback_dir=str(fallback),
            error=str(e),
            cloud_run=_is_cloud_run(),
        )
        fallback.mkdir(parents=True, exist_ok=True)
        return fallback


def _list_objects(bucket, prefix: str) -> list[str]:
    """
    List object names under prefix (best-effort). Used for debugging prefix mismatch.
    """
    try:
        names: list[str] = []
        for i, blob in enumerate(bucket.list_blobs(prefix=prefix)):
            names.append(blob.name)
            if i >= 2000:
                break
        return names
    except Exception as e:
        logger.warning("[RAG] Failed to list objects under prefix (continuing)", prefix=prefix, error=str(e))
        return []


def download_index_from_gcs() -> bool:
    """
    Download RAG index files from GCS into the configured local directory.

    Source: gs://<RAG_INDEX_GCS_BUCKET>/<RAG_INDEX_GCS_PREFIX>
    Local:  <RAG_INDEX_LOCAL_DIR>
    """
    global _last_download_error
    _last_download_error = None

    try:
        from google.cloud import storage
    except ImportError:
        logger.error("[RAG] google-cloud-storage not installed - cannot download index from GCS", exc_info=True)
        _last_download_error = "ImportError: google-cloud-storage not installed"
        set_phase("error", error=_last_download_error)
        return False

    bucket_name = settings.RAG_INDEX_GCS_BUCKET
    index_prefix = _normalize_prefix(getattr(settings, "RAG_INDEX_GCS_PREFIX", "latest_model/"))
    requested_local_dir = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
    
    # Resolve local dir (avoid double-prefix)
    local_path = _resolve_local_dir(requested_local_dir, index_prefix)
    # Use atomic temp directory for download, then rename to final location
    tmp_parent = local_path.parent
    tmp_dir = tmp_parent / f"{local_path.name}.tmp.{os.getpid()}"
    
    # Log resolved paths for debugging
    logger.info(
        "[RAG] Resolved index paths",
        bucket=bucket_name,
        prefix=index_prefix,
        requested_local_dir=requested_local_dir,
        resolved_local_dir=str(local_path),
        cloud_run=_is_cloud_run(),
    )
    print(f"[RAG] Starting GCS index download from gs://{bucket_name}/{index_prefix} to {str(local_path)}...", flush=True)
    
    # Initialize state tracking
    reset_state()
    set_phase("downloading", bucket=bucket_name, prefix=index_prefix, local_dir=str(local_path))
    
    logger.info(
        "[RAG] Starting GCS index download...",
        bucket=bucket_name,
        prefix=index_prefix,
        local_dir=str(local_path),
        cloud_run=_is_cloud_run(),
    )

    # Initialize GCS client
    try:
        print(f"[RAG] Initializing GCS client for bucket: {bucket_name}...", flush=True)
        client = storage.Client()
        bucket = client.bucket(bucket_name)
        print("[RAG] ✅ GCS client initialized successfully", flush=True)
        logger.info("[RAG] GCS client initialized", bucket=bucket_name)
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"[RAG] ❌ Failed to initialize GCS client: {error_msg}", flush=True)
        logger.error("[RAG] Failed to initialize GCS client", bucket=bucket_name, error=error_msg, exc_info=True)
        _last_download_error = error_msg
        set_phase("error", error=error_msg)
        return False

    # File lock to ensure only one worker downloads at a time (multi-worker gunicorn)
    lock_path = Path("/tmp/rag_index_download.lock")
    lock_file = None
    try:
        try:
            # Best-effort file lock using fcntl (Linux/Cloud Run)
            import fcntl  # type: ignore
            lock_file = open(lock_path, "w+")
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            logger.info("[RAG] Acquired download file lock", lock=str(lock_path))
        except Exception:
            # Fallback: create lock file exclusively; if cannot acquire, wait/poll
            acquired = False
            try:
                fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.close(fd)
                acquired = True
                logger.info("[RAG] Created lock file (no fcntl)", lock=str(lock_path))
            except Exception:
                logger.warning("[RAG] Could not acquire lock file; entering wait loop", lock=str(lock_path))
            if not acquired:
                # Wait until either lock disappears or final dir becomes valid
                WAIT_TOTAL = int(os.getenv("RAG_DOWNLOAD_LOCK_WAIT_SEC", "600"))
                WAIT_STEP = int(os.getenv("RAG_DOWNLOAD_LOCK_WAIT_STEP_SEC", "2"))
                waited = 0
                while waited < WAIT_TOTAL:
                    # Check if final dir is valid
                    try:
                        final_valid = True
                        for f in REQUIRED_FILES:
                            p = local_path / f
                            if not p.exists() or p.stat().st_size <= 1024:
                                final_valid = False
                                break
                        if final_valid:
                            logger.info("[RAG] Detected valid index during lock wait; skipping download", final_dir=str(local_path))
                            set_phase("downloaded", local_dir=str(local_path))
                            return True
                    except Exception:
                        pass
                    # If lock file disappeared, break and attempt normal path
                    if not lock_path.exists():
                        break
                    time.sleep(WAIT_STEP)
                    waited += WAIT_STEP
                # Try to create the lock again after waiting
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    os.close(fd)
                    logger.info("[RAG] Lock file acquired after waiting", lock=str(lock_path))
                except Exception:
                    # As a last resort, if final dir is still invalid and cannot acquire lock, fail fast
                    logger.error("[RAG] Failed to acquire download lock and final dir invalid; aborting download to avoid races", lock=str(lock_path))
                    _last_download_error = "Unable to acquire download lock; another worker may be initializing"
                    set_phase("error", error=_last_download_error)
                    return False
        # Ensure tmp dir is clean
        try:
            if tmp_dir.exists():
                for p in tmp_dir.glob("*"):
                    try:
                        p.unlink()
                    except Exception:
                        pass
            tmp_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error("[RAG] Failed to prepare temp dir for atomic download", tmp_dir=str(tmp_dir), error=str(e))
            _last_download_error = f"Temp dir prepare failed: {type(e).__name__}: {e}"
            set_phase("error", error=_last_download_error)
            return False

        log_resource_checkpoint("rag_gcs_download_start")
        # Track download results
        required_success: list[str] = []
        required_failures: list[str] = []
        optional_results: dict[str, str] = {}
        download_errors: dict[str, str] = {}

        # List objects under prefix (helps debug prefix mismatch)
        objects_under_prefix: list[str] = []
        if index_prefix:
            objects_under_prefix = _list_objects(bucket, index_prefix)
            logger.info(
                "[RAG] Objects under configured prefix",
                bucket=bucket_name,
                prefix=index_prefix,
                count=len(objects_under_prefix),
            )
        else:
            logger.info("[RAG] Prefix is empty (bucket root). Skipping prefix listing to avoid huge scans.")

        prefixes_to_try: list[str] = [index_prefix]
        if index_prefix and len(objects_under_prefix) == 0:
            prefixes_to_try = [index_prefix, ""]
            logger.warning(
                "[RAG] No objects found under configured prefix; attempting fallback root lookup for known filenames",
                bucket=bucket_name,
                prefix=index_prefix,
            )
        elif not index_prefix:
            prefixes_to_try = [""]

        # Initialize file tracking
        all_files = REQUIRED_FILES + OPTIONAL_FILES
        init_file_tracking(all_files)

        def _download_one(prefix: str, filename: str) -> bool:
            gcs_obj = f"{prefix}{filename}" if prefix else filename
            # Always download into temp dir first (atomic)
            local_file_path = tmp_dir / filename
            t0 = time.time()
            
            try:
                blob = bucket.blob(gcs_obj)
                
                # Get blob size if available (for progress tracking)
                try:
                    blob.reload()
                    size_bytes = blob.size or 0
                except Exception:
                    size_bytes = 0
                
                update_file_start(filename, size_bytes)
                
                gcs_path = f"gs://{bucket_name}/{gcs_obj}"
                print(f"[RAG] Downloading {filename} from {gcs_path}...", flush=True)
                logger.info(
                    "[RAG] Downloading file...",
                    filename=filename,
                    gcs_path=gcs_path,
                    size_bytes=size_bytes,
                    attempt=1,
                )
                
                # Wrap download in timeout to prevent indefinite hangs
                # Default timeout: 10 minutes per file (configurable via env var)
                download_timeout = int(os.getenv("RAG_DOWNLOAD_TIMEOUT_SEC", "600"))  # 10 minutes default
                
                # Configure retry with timeout to prevent indefinite retries
                from google.api_core import retry as api_retry
                
                # Custom retry that respects timeout
                custom_retry = api_retry.Retry(
                    predicate=api_retry.if_exception_type(Exception),
                    initial=1.0,  # Initial delay 1 second
                    maximum=60.0,  # Max delay 60 seconds
                    multiplier=2.0,  # Exponential backoff
                    timeout=download_timeout,  # Total timeout for all retries
                )
                
                # Use threading timeout as additional safety net
                download_result = [None]
                download_exception = [None]
                
                def _download_with_timeout():
                    try:
                        # Pass custom retry to download method
                        download_result[0] = blob.download_to_filename(
                            str(local_file_path),
                            retry=custom_retry
                        )
                    except Exception as e:
                        download_exception[0] = e
                
                download_thread = threading.Thread(target=_download_with_timeout, daemon=True)
                download_thread.start()
                download_thread.join(timeout=download_timeout + 10)  # Add 10s buffer for thread overhead
                
                if download_thread.is_alive():
                    # Thread is still running - timeout occurred
                    elapsed = time.time() - t0
                    error_msg = (
                        f"Download timed out after {download_timeout} seconds. "
                        f"This usually indicates network issues, very large files, or GCS connectivity problems. "
                        f"File: {filename}, Size: {size_bytes:,} bytes. "
                        f"Check GCS bucket permissions and network connectivity."
                    )
                    logger.error(
                        "[RAG] Download timeout",
                        filename=filename,
                        gcs_path=gcs_path,
                        timeout_seconds=download_timeout,
                        elapsed_s=elapsed,
                        size_bytes=size_bytes,
                        message=error_msg,
                    )
                    update_file_error(filename, error_msg, elapsed)
                    return False
                
                if download_exception[0]:
                    # Re-raise the exception from the thread
                    raise download_exception[0]
                
                if not local_file_path.exists():
                    elapsed = time.time() - t0
                    error_msg = "Download completed but file not found locally"
                    logger.error(
                        "[RAG] Download completed but file not found locally",
                        filename=filename,
                        local_path=str(local_file_path),
                        gcs_path=gcs_path,
                        elapsed_s=elapsed,
                    )
                    update_file_error(filename, error_msg, elapsed)
                    return False
                
                # Get actual file size
                actual_size = local_file_path.stat().st_size
                elapsed = time.time() - t0
                
                update_file_success(filename, actual_size, elapsed)
                
                logger.info(
                    "[RAG] Downloaded file",
                    filename=filename,
                    gcs_path=gcs_path,
                    size_bytes=actual_size,
                    local_path=str(local_file_path),
                    elapsed_s=elapsed,
                )
                print(f"[RAG] ✅ Downloaded {filename} ({actual_size:,} bytes in {elapsed:.2f}s)", flush=True)
                return True
            
            except Exception as e:
                elapsed = time.time() - t0
                error_type = type(e).__name__
                error_msg = str(e)
                status_code = getattr(e, "status_code", None)
                
                full_error = f"{error_type}: {error_msg}"
                if status_code:
                    full_error = f"{error_type} (status={status_code}): {error_msg}"
                
                logger.error(
                    "[RAG] Download failed",
                    filename=filename,
                    gcs_path=f"gs://{bucket_name}/{gcs_obj}",
                    error=full_error,
                    elapsed_s=elapsed,
                    status_code=status_code,
                    exc_info=True,
                )
                download_errors.setdefault(filename, full_error)
                update_file_error(filename, full_error, elapsed)
                return False

        # Download required files
        logger.info("[RAG] Downloading required index files...", files=REQUIRED_FILES, prefixes_to_try=prefixes_to_try)
        for filename in REQUIRED_FILES:
            downloaded = False
            for pfx in prefixes_to_try:
                if _download_one(pfx, filename):
                    downloaded = True
                    break
            if downloaded:
                required_success.append(filename)
            else:
                required_failures.append(filename)

        # Download optional files (non-blocking)
        logger.info("[RAG] Downloading optional index files...", files=OPTIONAL_FILES, prefixes_to_try=prefixes_to_try)
        for filename in OPTIONAL_FILES:
            downloaded = False
            for pfx in prefixes_to_try:
                if _download_one(pfx, filename):
                    downloaded = True
                    break
            optional_results[filename] = "success" if downloaded else "not_found"

        # Validate results
        if required_failures:
            failure_reasons = {f: download_errors.get(f) for f in required_failures if download_errors.get(f)}
            error_msg = (
                f"Index download failed for required files. "
                f"bucket=gs://{bucket_name}/ prefix={index_prefix!r} missing={required_failures} "
                f"local_dir={str(local_path)} "
                f"sample_errors={failure_reasons}"
            )
            logger.error(
                "[RAG] Index download failed — missing required files",
                bucket=bucket_name,
                prefix=index_prefix,
                prefixes_tried=prefixes_to_try,
                required_failures=required_failures,
                required_success=required_success,
                failure_reasons=failure_reasons,
                objects_under_prefix_count=len(objects_under_prefix),
                objects_under_prefix_sample=objects_under_prefix[:25],
                local_dir=str(local_path),
                message=f"Failed to download {len(required_failures)} required file(s): {', '.join(required_failures)}",
            )
            _last_download_error = error_msg
            set_phase("error", error=error_msg)
            return False

        # Verify all required files are present in temp dir
        missing_locally = [f for f in REQUIRED_FILES if not (tmp_dir / f).exists()]
        if missing_locally:
            try:
                local_listing = sorted([p.name for p in tmp_dir.iterdir() if p.is_file()])
            except Exception:
                local_listing = []
            error_msg = (
                f"Validation failed: required files missing locally after download: {missing_locally}. "
                f"bucket=gs://{bucket_name}/ prefix={index_prefix!r} local_dir={str(tmp_dir)}"
            )
            logger.error(
                "[RAG] Validation failed — files missing after download",
                missing_files=missing_locally,
                local_dir=str(tmp_dir),
                local_files=local_listing,
                bucket=bucket_name,
                prefix=index_prefix,
                prefixes_tried=prefixes_to_try,
                message=f"Files not found locally after download: {', '.join(missing_locally)}",
            )
            _last_download_error = error_msg
            set_phase("error", error=error_msg)
            return False

        # If final dir already exists and is valid, prefer it and discard temp
        try:
            final_valid = True
            for f in REQUIRED_FILES:
                p = local_path / f
                if not p.exists() or p.stat().st_size <= 1024:
                    final_valid = False
                    break
            if final_valid:
                logger.info("[RAG] Final dir already valid; skipping atomic move", final_dir=str(local_path))
                # Cleanup tmp
                try:
                    if tmp_dir.exists():
                        for p in tmp_dir.glob("*"):
                            try:
                                p.unlink()
                            except Exception:
                                pass
                        tmp_dir.rmdir()
                except Exception:
                    pass
                set_phase("downloaded", local_dir=str(local_path))
                return True
        except Exception:
            pass

        # Atomically move temp files into final location, file-by-file (avoid clobbering valid dir)
        try:
            # Ensure parent and final dir exist
            local_path.parent.mkdir(parents=True, exist_ok=True)
            local_path.mkdir(parents=True, exist_ok=True)
            # Move files from tmp to final
            moved = 0
            for p in tmp_dir.glob("*"):
                target = local_path / p.name
                try:
                    p.replace(target)
                    moved += 1
                except Exception as e:
                    logger.error("[RAG] Failed moving file to final dir", src=str(p), dst=str(target), error=str(e))
                    _last_download_error = f"Atomic move failed: {type(e).__name__}: {e}"
                    set_phase("error", error=_last_download_error)
                    return False
            logger.info("[RAG] Atomic move complete", files_moved=moved, final_dir=str(local_path))
        finally:
            # Best-effort cleanup of temp dir
            try:
                if tmp_dir.exists():
                    for p in tmp_dir.glob("*"):
                        try:
                            p.unlink()
                        except Exception:
                            pass
                    tmp_dir.rmdir()
            except Exception:
                pass

        # Download complete - mark as downloaded (files exist and validated)
        set_phase("downloaded", local_dir=str(local_path))
        log_resource_checkpoint("rag_gcs_download_complete")
        
        # Get total bytes from state (already updated by update_file_success)
        from backend.rag.index_state import get_index_state
        state = get_index_state()
        total_bytes_downloaded = state.get("bytes_downloaded", 0)
        
        # CRITICAL: Plain text log for gcloud textPayload searches
        logger.info(f"[RAG] Download complete: files_done={len(required_success)}/{len(REQUIRED_FILES)} bytes={total_bytes_downloaded:,} local_dir={local_path}")
        print(f"[RAG] Download complete: files_done={len(required_success)}/{len(REQUIRED_FILES)} bytes={total_bytes_downloaded:,} local_dir={local_path}", flush=True)
        
        logger.info(
            "[RAG] Index download and validation complete",
            local_dir=str(local_path),
            required_files=REQUIRED_FILES,
            optional_results=optional_results,
            files_done=len(required_success),
            files_total=len(REQUIRED_FILES),
            total_bytes=total_bytes_downloaded,
            message="Ready to load RAG index",
        )
        return True
    finally:
        # Release lock
        try:
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
            # Remove lock file if we created it
            try:
                if lock_path.exists():
                    lock_path.unlink()
            except Exception:
                pass
        except Exception:
            pass

