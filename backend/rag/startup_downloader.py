"""
RAG index downloader for production environments.

This module handles downloading RAG index files from Google Cloud Storage
into the local filesystem for use by the RAG pipeline.
"""

import os
from pathlib import Path
from typing import List
from backend.logging_config import get_logger

logger = get_logger(__name__)

# Configuration constants
BUCKET_NAME = "arrow-rag-support-prod-rag"
INDEX_PREFIX = "latest_model/"
LOCAL_DIR = "/app/latest_model"

REQUIRED_FILES = [
    "docstore.json",
    "index_store.json",
    "default__vector_store.json",
]

OPTIONAL_FILES = [
    "graph_store.json",
    "image__vector_store.json",
]


def download_index_from_gcs() -> bool:
    """
    Downloads all RAG index files from GCS into /app/latest_model.
    
    Downloads from: gs://arrow-rag-support-prod-rag/latest_model/
    Stores to: /app/latest_model/
    
    Returns:
        True if all required files were downloaded successfully, False otherwise.
    """
    try:
        from google.cloud import storage
    except ImportError:
        logger.error(
            "[RAG] google-cloud-storage not installed - cannot download index from GCS",
            exc_info=True
        )
        return False
    
    logger.info("[RAG] Starting GCS index download...", 
                bucket=BUCKET_NAME, 
                prefix=INDEX_PREFIX,
                local_dir=LOCAL_DIR)
    
    # Ensure local directory exists
    local_path = Path(LOCAL_DIR)
    try:
        local_path.mkdir(parents=True, exist_ok=True)
        logger.info("[RAG] Created/verified local directory", local_dir=LOCAL_DIR)
    except Exception as e:
        logger.error(
            "[RAG] Failed to create local directory",
            local_dir=LOCAL_DIR,
            error=str(e),
            exc_info=True
        )
        return False
    
    # Initialize GCS client
    try:
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        logger.info("[RAG] GCS client initialized", bucket=BUCKET_NAME)
    except Exception as e:
        logger.error(
            "[RAG] Failed to initialize GCS client",
            bucket=BUCKET_NAME,
            error=str(e),
            exc_info=True
        )
        return False
    
    # Track download results
    required_success = []
    required_failures = []
    optional_results = {}
    
    # Download required files
    logger.info("[RAG] Downloading required index files...", files=REQUIRED_FILES)
    for filename in REQUIRED_FILES:
        gcs_path = f"{INDEX_PREFIX}{filename}"
        local_file_path = local_path / filename
        
        try:
            blob = bucket.blob(gcs_path)
            
            # Check if blob exists
            if not blob.exists():
                logger.error(
                    "[RAG] Required file not found in GCS",
                    gcs_path=gcs_path,
                    filename=filename
                )
                required_failures.append(filename)
                continue
            
            # Download file
            logger.info("[RAG] Downloading file...", filename=filename, gcs_path=gcs_path)
            blob.download_to_filename(str(local_file_path))
            
            # Verify download
            if not local_file_path.exists():
                logger.error(
                    "[RAG] Download completed but file not found locally",
                    filename=filename,
                    local_path=str(local_file_path)
                )
                required_failures.append(filename)
                continue
            
            file_size = local_file_path.stat().st_size
            logger.info(
                f"[RAG] Downloaded: {filename} ({file_size} bytes)",
                filename=filename,
                size=file_size,
                gcs_path=gcs_path,
                local_path=str(local_file_path)
            )
            required_success.append(filename)
            
        except Exception as e:
            logger.error(
                f"[RAG] Download failed for: {filename}",
                filename=filename,
                gcs_path=gcs_path,
                error=str(e),
                exc_info=True
            )
            required_failures.append(filename)
    
    # Download optional files (non-blocking)
    logger.info("[RAG] Downloading optional index files...", files=OPTIONAL_FILES)
    for filename in OPTIONAL_FILES:
        gcs_path = f"{INDEX_PREFIX}{filename}"
        local_file_path = local_path / filename
        
        try:
            blob = bucket.blob(gcs_path)
            
            if not blob.exists():
                logger.debug(
                    "[RAG] Optional file not found in GCS (skipping)",
                    filename=filename,
                    gcs_path=gcs_path
                )
                optional_results[filename] = "not_found"
                continue
            
            blob.download_to_filename(str(local_file_path))
            
            if local_file_path.exists():
                file_size = local_file_path.stat().st_size
                logger.info(
                    f"[RAG] Downloaded optional file: {filename} ({file_size} bytes)",
                    filename=filename,
                    size=file_size
                )
                optional_results[filename] = "success"
            else:
                logger.warning(
                    "[RAG] Optional file download failed (non-critical)",
                    filename=filename
                )
                optional_results[filename] = "failed"
                
        except Exception as e:
            logger.warning(
                "[RAG] Optional file download error (non-critical)",
                filename=filename,
                error=str(e)
            )
            optional_results[filename] = "error"
    
    # Validate results
    if required_failures:
        logger.error(
            "[RAG] Index download failed — missing required files",
            required_failures=required_failures,
            required_success=required_success,
            message=f"Failed to download {len(required_failures)} required file(s): {', '.join(required_failures)}"
        )
        return False
    
    logger.info(
        "[RAG] Index download complete — validating...",
        required_files=len(required_success),
        optional_files=len([v for v in optional_results.values() if v == "success"]),
        message="All required files downloaded successfully"
    )
    
    # Verify all required files are present locally
    missing_locally = []
    for filename in REQUIRED_FILES:
        if not (local_path / filename).exists():
            missing_locally.append(filename)
    
    if missing_locally:
        logger.error(
            "[RAG] Validation failed — files missing after download",
            missing_files=missing_locally,
            message=f"Files not found locally after download: {', '.join(missing_locally)}"
        )
        return False
    
    logger.info(
        "[RAG] Index download and validation complete",
        local_dir=LOCAL_DIR,
        required_files=REQUIRED_FILES,
        message="Ready to load RAG index"
    )
    
    return True

