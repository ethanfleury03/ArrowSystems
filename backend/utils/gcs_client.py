"""
Google Cloud Storage client utilities.

Provides helper functions for accessing files from Cloud Storage buckets.

Uses Application Default Credentials (ADC) which works automatically on Cloud Run
via the metadata server, or via GOOGLE_APPLICATION_CREDENTIALS in local dev.
"""

import os
import logging
from typing import Optional, BinaryIO, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import of Google Cloud Storage
_gcs_client = None
_gcs_available = None
_gcs_auth_info = None  # Cached auth info for logging
_gcs_last_init_error = None  # Cached last init error for surfacing in API responses


def _is_cloud_run() -> bool:
    """Detect if running on Cloud Run."""
    # Cloud Run sets K_SERVICE environment variable
    return bool(os.getenv("K_SERVICE"))


def _check_gcs_available() -> bool:
    """Check if google-cloud-storage is available."""
    global _gcs_available
    if _gcs_available is None:
        try:
            from google.cloud import storage
            _gcs_available = True
        except ImportError:
            logger.warning("google-cloud-storage not installed. Install with: pip install google-cloud-storage")
            _gcs_available = False
    return _gcs_available


def _get_auth_info():
    """Get authentication information for logging."""
    global _gcs_auth_info
    if _gcs_auth_info is not None:
        return _gcs_auth_info
    
    try:
        import google.auth
        creds, project = google.auth.default()
        creds_type = type(creds).__name__
        
        # Try to get service account email if available
        service_account_email = None
        if hasattr(creds, 'service_account_email'):
            service_account_email = creds.service_account_email
        elif hasattr(creds, '_service_account_email'):
            service_account_email = getattr(creds, '_service_account_email', None)
        
        _gcs_auth_info = {
            "project": project,
            "creds_type": creds_type,
            "service_account_email": service_account_email,
            "is_cloud_run": _is_cloud_run(),
            "has_goog_app_creds": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        }
        return _gcs_auth_info
    except Exception as e:
        logger.debug(f"Could not get auth info: {e}")
        return {
            "project": None,
            "creds_type": "unknown",
            "service_account_email": None,
            "is_cloud_run": _is_cloud_run(),
            "has_goog_app_creds": bool(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")),
        }


def get_gcs_client():
    """
    Get or create a GCS client using Application Default Credentials (ADC).
    
    On Cloud Run: Uses service account identity via metadata server automatically.
    On local dev: Uses GOOGLE_APPLICATION_CREDENTIALS if set, otherwise ADC.
    
    Returns:
        storage.Client instance or None if not available
    """
    global _gcs_client
    global _gcs_last_init_error
    
    if not _check_gcs_available():
        logger.error("Google Cloud Storage library not installed. Install with: pip install google-cloud-storage")
        return None
    
    if _gcs_client is None:
        try:
            from google.cloud import storage
            
            # Create client - this will use ADC automatically
            # On Cloud Run: uses metadata server (service account)
            # On local: uses GOOGLE_APPLICATION_CREDENTIALS if set, otherwise ADC
            _gcs_client = storage.Client()
            _gcs_last_init_error = None
            
            # Log authentication info on first initialization
            auth_info = _get_auth_info()
            if auth_info["is_cloud_run"]:
                logger.info(
                    {
                        "event": "gcs_client_initialized",
                        "environment": "cloud_run",
                        "project": auth_info["project"],
                        "creds_type": auth_info["creds_type"],
                        "service_account_email": auth_info["service_account_email"],
                        "message": "Using Cloud Run service account identity via metadata server",
                    }
                )
            else:
                logger.info(
                    {
                        "event": "gcs_client_initialized",
                        "environment": "local_dev",
                        "project": auth_info["project"],
                        "creds_type": auth_info["creds_type"],
                        "has_goog_app_creds": auth_info["has_goog_app_creds"],
                        "message": "Using Application Default Credentials (local dev mode)",
                    }
                )
            
        except Exception as e:
            error_msg = str(e)
            error_type = type(e).__name__
            is_cloud_run = _is_cloud_run()
            _gcs_last_init_error = f"{error_type}: {error_msg}"
            
            if "Could not automatically determine credentials" in error_msg or "DefaultCredentialsError" in error_type:
                # Log real underlying error; avoid blanket IAM advice unless we know it's permission-related
                logger.error(
                    {
                        "event": "gcs_auth_failed",
                        "environment": "cloud_run" if is_cloud_run else "local_dev",
                        "error_type": error_type,
                        "error_message": error_msg,
                    },
                    exc_info=True,
                )
            else:
                logger.error(
                    {
                        "event": "gcs_client_init_failed",
                        "error": str(e),
                        "error_type": error_type,
                        "environment": "cloud_run" if is_cloud_run else "local_dev",
                    },
                    exc_info=True
                )
            return None
    
    return _gcs_client


def get_gcs_last_init_error() -> Optional[str]:
    """Return the last GCS client initialization error (if any)."""
    return _gcs_last_init_error


class GCSUploadError(RuntimeError):
    """Raised when a GCS upload fails. Message preserves the underlying exception details."""


def parse_gcs_path(gcs_path: str) -> tuple[Optional[str], Optional[str]]:
    """
    Parse a GCS path into bucket and blob name.
    
    Args:
        gcs_path: Path in format gs://bucket/path or bucket/path
    
    Returns:
        Tuple of (bucket_name, blob_name) or (None, None) if invalid
    """
    if not gcs_path:
        return None, None
    
    # Remove gs:// prefix if present
    path = gcs_path.replace('gs://', '').strip('/')
    
    if not path:
        return None, None
    
    parts = path.split('/', 1)
    bucket_name = parts[0]
    blob_name = parts[1] if len(parts) > 1 else None
    
    return bucket_name, blob_name


def get_docs_bucket_name() -> Optional[str]:
    """Get the docs bucket name from environment variable."""
    return os.getenv("DOCS_BUCKET_NAME")


def download_blob(bucket_name: str, blob_name: str) -> Optional[bytes]:
    """
    Download a blob from Cloud Storage.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Blob name/path within bucket
    
    Returns:
        Blob contents as bytes, or None if error
    """
    client = get_gcs_client()
    if not client:
        logger.error("GCS client not available")
        return None
    
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.download_as_bytes()
    except Exception as e:
        logger.error(f"Failed to download blob {blob_name} from {bucket_name}: {e}")
        return None


def download_document(gcs_path: str) -> Optional[bytes]:
    """
    Download a document from Cloud Storage using its gcs_path.
    
    Args:
        gcs_path: Full GCS path (gs://bucket/path) or relative path
    
    Returns:
        Document contents as bytes, or None if error
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    if not bucket_name or not blob_name:
        logger.error(f"Invalid GCS path: {gcs_path}")
        return None
    
    return download_blob(bucket_name, blob_name)


def download_document_by_filename(filename: str, bucket_name: Optional[str] = None) -> Optional[bytes]:
    """
    Download a document by filename from the docs bucket.
    
    Args:
        filename: Document filename
        bucket_name: Optional bucket name (defaults to DOCS_BUCKET_NAME env var)
    
    Returns:
        Document contents as bytes, or None if error
    """
    if not bucket_name:
        bucket_name = get_docs_bucket_name()
    
    if not bucket_name:
        logger.error("DOCS_BUCKET_NAME environment variable not set")
        return None
    
    return download_blob(bucket_name, filename)


def blob_exists(bucket_name: str, blob_name: str) -> bool:
    """Check if a blob exists in Cloud Storage."""
    client = get_gcs_client()
    if not client:
        return False
    
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        return blob.exists()
    except Exception as e:
        logger.error(f"Failed to check blob existence {blob_name} in {bucket_name}: {e}")
        return False


def generate_signed_url(bucket_name: str, blob_name: str, expiration_minutes: int = 60) -> Optional[str]:
    """
    Generate a signed URL for a blob.
    
    Args:
        bucket_name: GCS bucket name
        blob_name: Blob name/path
        expiration_minutes: URL expiration time in minutes
    
    Returns:
        Signed URL string, or None if error
    """
    client = get_gcs_client()
    if not client:
        return None
    
    try:
        from datetime import timedelta
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(minutes=expiration_minutes),
            method="GET"
        )
        return url
    except Exception as e:
        logger.error(f"Failed to generate signed URL for {blob_name} in {bucket_name}: {e}")
        return None


def upload_bytes(bucket_name: str, object_name: str, content: bytes, content_type: str = "application/pdf") -> Optional[str]:
    """
    Upload bytes content to GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        object_name: Object name/path within bucket
        content: File content as bytes
        content_type: MIME type (default: application/pdf)
    
    Returns:
        GCS URI string (gs://bucket/object) if successful, None if error
    """
    client = get_gcs_client()
    if not client:
        auth_info = _get_auth_info()
        raise GCSUploadError(
            "GCS upload failed: storage client not available. "
            f"environment={'cloud_run' if auth_info.get('is_cloud_run') else 'local_dev'}, "
            f"creds_type={auth_info.get('creds_type')}, "
            f"service_account_email={auth_info.get('service_account_email')}, "
            f"last_init_error={get_gcs_last_init_error() or 'unknown'}"
        )
    
    try:
        # Lazy import to avoid hard dependency when running without google-cloud-storage
        try:
            from google.api_core.exceptions import GoogleAPICallError  # type: ignore
        except Exception:  # pragma: no cover
            GoogleAPICallError = Exception  # type: ignore

        bucket = client.bucket(bucket_name)
        # Check if bucket exists
        if not bucket.exists():
            raise GCSUploadError(f"GCS upload failed: bucket does not exist: {bucket_name}")
        
        blob = bucket.blob(object_name)
        blob.upload_from_string(content, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        # Preserve and surface the real underlying GCS exception (do not blanket-rewrite into IAM advice)
        error_type = type(e).__name__
        error_msg = str(e)
        size_bytes = len(content) if content is not None else 0

        # Best-effort extraction of HTTP status + API error details
        status_code = None
        errors = None
        reason = None
        try:
            status_code = getattr(e, "code", None)
            if callable(status_code):
                status_code = status_code()
        except Exception:
            status_code = None
        try:
            if status_code is None and hasattr(e, "response") and getattr(e, "response") is not None:
                status_code = getattr(getattr(e, "response"), "status", None) or getattr(getattr(e, "response"), "status_code", None)
        except Exception:
            pass
        try:
            errors = getattr(e, "errors", None)
            if isinstance(errors, list) and errors:
                reason = errors[0].get("reason") if isinstance(errors[0], dict) else None
        except Exception:
            errors = None
            reason = None

        logger.exception(
            {
                "event": "gcs_upload_failed",
                "bucket": bucket_name,
                "object_name": object_name,
                "content_type": content_type,
                "size_bytes": size_bytes,
                "exc_type": error_type,
                "exc_message": error_msg,
                "status_code": status_code,
                "errors": errors,
                "reason": reason,
            }
        )

        hint = None
        # Only suggest IAM when the *actual* error indicates permission/auth problems
        if status_code in (401, 403) or "PermissionDenied" in error_type or "Forbidden" in error_type:
            hint = "Permission denied. Confirm runtime service account + bucket IAM, and verify bucket/prefix are correct."

        base = f"GCS upload failed: {error_type}: {error_msg} (bucket={bucket_name}, object={object_name})"
        if hint:
            base = f"{base} Hint: {hint}"

        # If this was a google api call error, rethrow with chaining so logs keep traceback
        try:
            from google.api_core.exceptions import GoogleAPICallError  # type: ignore
            if isinstance(e, GoogleAPICallError):
                raise GCSUploadError(base) from e
        except Exception:
            # google api core not available or isinstance check failed; fall through
            pass

        raise GCSUploadError(base) from e


def upload_file(bucket_name: str, object_name: str, local_path: str, content_type: str = "application/pdf") -> Optional[str]:
    """
    Upload a local file to GCS bucket.
    
    Args:
        bucket_name: GCS bucket name
        object_name: Object name/path within bucket
        local_path: Path to local file
        content_type: MIME type (default: application/pdf)
    
    Returns:
        GCS URI string (gs://bucket/object) if successful, None if error
    """
    if not os.path.exists(local_path):
        logger.error(f"Local file not found: {local_path}")
        return None
    
    client = get_gcs_client()
    if not client:
        auth_info = _get_auth_info()
        raise GCSUploadError(
            "GCS upload failed: storage client not available. "
            f"environment={'cloud_run' if auth_info.get('is_cloud_run') else 'local_dev'}, "
            f"creds_type={auth_info.get('creds_type')}, "
            f"service_account_email={auth_info.get('service_account_email')}, "
            f"last_init_error={get_gcs_last_init_error() or 'unknown'}"
        )
    
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded file to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.exception(
            {
                "event": "gcs_upload_file_failed",
                "bucket": bucket_name,
                "object_name": object_name,
                "content_type": content_type,
                "local_path": local_path,
                "exc_type": type(e).__name__,
                "exc_message": str(e),
            }
        )
        raise GCSUploadError(f"GCS upload failed: {type(e).__name__}: {e} (bucket={bucket_name}, object={object_name})") from e


def list_objects(bucket_name: str, prefix: str = "") -> List[str]:
    """
    List all objects in a GCS bucket with the given prefix.
    
    Args:
        bucket_name: GCS bucket name
        prefix: Object name prefix to filter (default: empty string for all objects)
    
    Returns:
        List of object names (full paths within bucket)
    """
    client = get_gcs_client()
    if not client:
        logger.error("GCS client not available for listing")
        return []
    
    try:
        bucket = client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        object_names = [blob.name for blob in blobs]
        logger.debug(f"Listed {len(object_names)} objects from {bucket_name} with prefix '{prefix}'")
        return object_names
    except Exception as e:
        logger.error(f"Failed to list objects from {bucket_name} with prefix '{prefix}': {e}", exc_info=True)
        return []


def download_to_file(gcs_uri: str, dest_path: str) -> bool:
    """
    Download a GCS object to a local file.
    
    Args:
        gcs_uri: Full GCS URI (gs://bucket/object) or bucket/object path
        dest_path: Local file path to save to
    
    Returns:
        True if successful, False if error
    """
    bucket_name, blob_name = parse_gcs_path(gcs_uri)
    
    if not bucket_name or not blob_name:
        logger.error(f"Invalid GCS URI: {gcs_uri}")
        return False
    
    client = get_gcs_client()
    if not client:
        logger.error("GCS client not available for download")
        return False
    
    try:
        # Ensure destination directory exists
        dest_dir = os.path.dirname(dest_path)
        if dest_dir:
            os.makedirs(dest_dir, exist_ok=True)
        
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.download_to_filename(dest_path)
        logger.info(f"Successfully downloaded {gcs_uri} to {dest_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to download {gcs_uri} to {dest_path}: {e}", exc_info=True)
        return False


def object_exists(gcs_path: str) -> bool:
    """
    Check if a GCS object exists.
    
    Args:
        gcs_path: Full GCS path (gs://bucket/path) or bucket/path
    
    Returns:
        True if object exists, False if not found or error
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    if not bucket_name or not blob_name:
        return False
    
    return blob_exists(bucket_name, blob_name)


def delete_object(gcs_path: str) -> bool:
    """
    Delete an object from GCS (best-effort, logs warning on failure).
    
    Args:
        gcs_path: Full GCS path (gs://bucket/path) or bucket/path
    
    Returns:
        True if successful, False if error (but does not raise exception)
    """
    bucket_name, blob_name = parse_gcs_path(gcs_path)
    
    if not bucket_name or not blob_name:
        logger.warning(f"Invalid GCS path for deletion: {gcs_path}")
        return False
    
    client = get_gcs_client()
    if not client:
        logger.warning("GCS client not available for deletion")
        return False
    
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.delete()
        logger.info(f"Successfully deleted GCS object: {gcs_path}")
        return True
    except Exception as e:
        # Handle 404 (object not found) gracefully - this is expected for orphaned records
        error_str = str(e)
        if "404" in error_str or "NotFound" in str(type(e).__name__):
            logger.debug(f"GCS object not found (already deleted or never existed): {gcs_path}")
            return True  # Consider this success - object is gone
        # Log warning but don't fail - deletion is best-effort
        logger.warning(f"Failed to delete GCS object {gcs_path}: {e}")
        return False
















