"""
Google Cloud Storage client utilities.

Provides helper functions for accessing files from Cloud Storage buckets.
"""

import os
import logging
from typing import Optional, BinaryIO
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
        return None
    
    if _gcs_client is None:
        try:
            from google.cloud import storage
            _gcs_client = storage.Client()
        except Exception as e:
            logger.error(f"Failed to initialize GCS client: {e}")
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






