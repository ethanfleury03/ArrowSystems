"""
Google Cloud Storage client utilities.

Provides helper functions for accessing files from Cloud Storage buckets.
"""

import os
import logging
from typing import Optional, BinaryIO, List
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy import of Google Cloud Storage
_gcs_client = None
_gcs_available = None


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


def get_gcs_client():
    """Get or create a GCS client. Returns None if not available."""
    global _gcs_client
    
    if not _check_gcs_available():
        logger.error("Google Cloud Storage library not installed. Install with: pip install google-cloud-storage")
        return None
    
    if _gcs_client is None:
        try:
            from google.cloud import storage
            # Check for credentials
            creds_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
            if not creds_path and not os.path.exists(os.path.expanduser("~/.config/gcloud/application_default_credentials.json")):
                logger.warning("GCS credentials not found. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'")
            _gcs_client = storage.Client()
        except Exception as e:
            error_msg = str(e)
            if "Could not automatically determine credentials" in error_msg or "DefaultCredentialsError" in str(type(e).__name__):
                logger.error("GCS authentication failed. Set GOOGLE_APPLICATION_CREDENTIALS environment variable or run 'gcloud auth application-default login'")
            else:
                logger.error(f"Failed to initialize GCS client: {e}", exc_info=True)
            return None
    
    return _gcs_client


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
        logger.error("GCS client not available for upload. Check GCS credentials and configuration.")
        return None
    
    try:
        bucket = client.bucket(bucket_name)
        # Check if bucket exists
        if not bucket.exists():
            logger.error(f"GCS bucket does not exist: {bucket_name}. Please create the bucket or check the bucket name.")
            return None
        
        blob = bucket.blob(object_name)
        blob.upload_from_string(content, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        error_type = type(e).__name__
        error_msg = str(e)
        
        # Provide more specific error messages
        if "403" in error_msg or "Forbidden" in error_msg or "PermissionDenied" in error_type:
            logger.error(f"GCS permission denied. Service account needs 'storage.objects.create' permission on bucket {bucket_name}")
        elif "404" in error_msg or "NotFound" in error_type:
            logger.error(f"GCS bucket not found: {bucket_name}. Verify the bucket name and that it exists in your GCP project.")
        elif "Could not automatically determine credentials" in error_msg or "DefaultCredentialsError" in error_type:
            logger.error("GCS authentication failed. Set GOOGLE_APPLICATION_CREDENTIALS or run 'gcloud auth application-default login'")
        else:
        logger.error(f"Failed to upload {object_name} to {bucket_name}: {e}", exc_info=True)
        return None


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
        logger.error("GCS client not available for upload")
        return None
    
    try:
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(object_name)
        blob.upload_from_filename(local_path, content_type=content_type)
        gcs_uri = f"gs://{bucket_name}/{object_name}"
        logger.info(f"Successfully uploaded file to GCS: {gcs_uri}")
        return gcs_uri
    except Exception as e:
        logger.error(f"Failed to upload file {local_path} to {bucket_name}/{object_name}: {e}", exc_info=True)
        return None


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
















