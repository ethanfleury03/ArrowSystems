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
from pathlib import Path
from typing import Optional

from backend.config.env import settings
from backend.logging_config import get_logger
from backend.rag.index_state import (
    set_phase, reset_state, init_file_tracking,
    update_file_start, update_file_success, update_file_error
)

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
        local_file_path = local_path / filename
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
            
            blob.download_to_filename(str(local_file_path))
            
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

    # Verify all required files are present locally
    missing_locally = [f for f in REQUIRED_FILES if not (local_path / f).exists()]
    if missing_locally:
        try:
            local_listing = sorted([p.name for p in local_path.iterdir() if p.is_file()])
        except Exception:
            local_listing = []
        error_msg = (
            f"Validation failed: required files missing locally after download: {missing_locally}. "
            f"bucket=gs://{bucket_name}/ prefix={index_prefix!r} local_dir={str(local_path)}"
        )
        logger.error(
            "[RAG] Validation failed — files missing after download",
            missing_files=missing_locally,
            local_dir=str(local_path),
            local_files=local_listing,
            bucket=bucket_name,
            prefix=index_prefix,
            prefixes_tried=prefixes_to_try,
            message=f"Files not found locally after download: {', '.join(missing_locally)}",
        )
        _last_download_error = error_msg
        set_phase("error", error=error_msg)
        return False

    # Download complete - mark as ready
    set_phase("ready")
    
    logger.info(
        "[RAG] Index download and validation complete",
        local_dir=str(local_path),
        required_files=REQUIRED_FILES,
        optional_results=optional_results,
        files_done=len(required_success),
        files_total=len(REQUIRED_FILES),
        message="Ready to load RAG index",
    )
    print(f"[RAG] ✅ Index download and validation complete - downloaded {len(required_success)} required files", flush=True)
    return True

