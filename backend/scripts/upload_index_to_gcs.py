"""
Upload RAG index directory to Google Cloud Storage bucket.

This script uploads the local latest_model/ directory to the GCS bucket
so it can be mounted in Cloud Run at /app/latest_model.
"""

import argparse
import os
from pathlib import Path
from google.cloud import storage
from google.cloud.exceptions import NotFound
from google.api_core import exceptions


def clear_gcs_prefix(bucket_name: str, gcs_prefix: str) -> int:
    """
    Clear all objects under a GCS prefix before uploading.
    
    This ensures no stale files remain if the index structure changes.
    
    Args:
        bucket_name: GCS bucket name
        gcs_prefix: Prefix/path in bucket to clear
    
    Returns:
        Number of objects deleted
    """
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to GCS bucket '{bucket_name}': {e}. "
            "Ensure you have gcloud auth configured and the bucket exists."
        )
    
    # List all blobs under the prefix
    if gcs_prefix:
        # Ensure prefix ends with / for proper matching
        prefix_to_clear = gcs_prefix.rstrip('/') + '/'
    else:
        # If no prefix, we should NOT clear the entire bucket - this is a safety check
        # However, if the user explicitly wants to clear root, they can set prefix to "/"
        raise ValueError(
            "Cannot clear bucket root (gcs_prefix is empty). "
            "This is a safety measure. If you really want to clear the root, "
            "set gcs_prefix to '/' explicitly."
        )
    
    blobs = list(bucket.list_blobs(prefix=prefix_to_clear))
    deleted_count = 0
    
    if blobs:
        print(f"[CLEAR] Clearing {len(blobs)} objects from gs://{bucket_name}/{prefix_to_clear}")
        for blob in blobs:
            try:
                blob.delete()
                deleted_count += 1
            except Exception as e:
                print(f"   [WARN] Failed to delete {blob.name}: {e}")
        print(f"   [OK] Deleted {deleted_count} objects")
    else:
        print(f"[CLEAR] No objects found at gs://{bucket_name}/{prefix_to_clear}")
    
    return deleted_count


def upload_directory_to_gcs(
    local_dir: str,
    bucket_name: str,
    gcs_prefix: str = "latest_model",
    overwrite: bool = True,
    clear_before_upload: bool = False
) -> None:
    """
    Upload a local directory to a GCS bucket.
    
    Args:
        local_dir: Local directory path to upload
        bucket_name: GCS bucket name
        gcs_prefix: Prefix/path in bucket (default: "latest_model")
        overwrite: Whether to overwrite existing files
        clear_before_upload: If True, clear all objects under gcs_prefix before uploading
    """
    local_path = Path(local_dir)
    
    if not local_path.exists():
        raise FileNotFoundError(f"Local directory not found: {local_dir}")
    
    if not local_path.is_dir():
        raise ValueError(f"Path is not a directory: {local_dir}")
    
    # Verify it's a valid index directory
    docstore_path = local_path / "docstore.json"
    if not docstore_path.exists():
        raise ValueError(
            f"Directory does not appear to be a valid RAG index "
            f"(missing docstore.json): {local_dir}"
        )
    
    print(f"[UPLOAD] Uploading RAG index from: {local_dir}")
    print(f"         To GCS bucket: {bucket_name}/{gcs_prefix}/")
    print()
    
    # Initialize GCS client
    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)
    except Exception as e:
        raise RuntimeError(
            f"Failed to connect to GCS bucket '{bucket_name}': {e}. "
            "Ensure you have gcloud auth configured and the bucket exists."
        )
    
    # Check if bucket exists (skip if we don't have permission to check)
    try:
        bucket.reload()
    except NotFound:
        raise RuntimeError(
            f"Bucket '{bucket_name}' does not exist. "
            "Create it with: gsutil mb gs://{bucket_name}"
        )
    except exceptions.Forbidden:
        # Don't have permission to check bucket metadata, but we can still try to upload
        print(f"[WARN] Cannot verify bucket '{bucket_name}' exists (missing storage.buckets.get permission)")
        print("         Attempting to upload anyway...")
        print()
    
    # Clear prefix before upload if requested
    if clear_before_upload:
        try:
            deleted_count = clear_gcs_prefix(bucket_name, gcs_prefix)
            print()
        except Exception as e:
            print(f"[WARN] Failed to clear prefix before upload: {e}")
            print("         Continuing with upload anyway...")
            print()
    
    # Upload all files in the directory
    uploaded_count = 0
    skipped_count = 0
    
    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = Path(root) / file
            # Get relative path from local_dir
            relative_path = local_file_path.relative_to(local_path)
            # Construct GCS blob path
            if gcs_prefix:
                gcs_blob_path = f"{gcs_prefix}/{relative_path}".replace("\\", "/")
            else:
                # Upload to bucket root (no prefix)
                gcs_blob_path = str(relative_path).replace("\\", "/")
            
            blob = bucket.blob(gcs_blob_path)
            
            # Check if file already exists
            if blob.exists() and not overwrite:
                print(f"   [SKIP] Skipping (exists): {gcs_blob_path}")
                skipped_count += 1
                continue
            
            # Upload file
            try:
                blob.upload_from_filename(str(local_file_path))
                print(f"   [OK] Uploaded: {gcs_blob_path}")
                uploaded_count += 1
            except Exception as e:
                print(f"   [ERROR] Failed to upload {gcs_blob_path}: {e}")
                raise
    
    print()
    print("=" * 70)
    print(f"[SUCCESS] Upload complete!")
    print(f"         Uploaded: {uploaded_count} files")
    if skipped_count > 0:
        print(f"         Skipped: {skipped_count} files (already exist)")
    print()
    if gcs_prefix:
        print(f"[INFO] Index is now available at: gs://{bucket_name}/{gcs_prefix}/")
        print(f"       Cloud Run mounts bucket root to /app/latest_model/, so files will appear at: /app/latest_model/{gcs_prefix}/")
    else:
        print(f"[INFO] Index is now available at: gs://{bucket_name}/")
        print(f"       Cloud Run mounts bucket root to /app/latest_model/, so files will appear at: /app/latest_model/")
    print("=" * 70)


def main():
    """Main entry point for the upload script."""
    parser = argparse.ArgumentParser(
        description="Upload RAG index directory to GCS bucket"
    )
    parser.add_argument(
        "--dir",
        type=str,
        default="latest_model",
        help="Local directory path containing the RAG index (default: latest_model)"
    )
    parser.add_argument(
        "--bucket",
        type=str,
        default="arrow-rag-support-prod-rag",
        help="GCS bucket name (default: arrow-rag-support-prod-rag)"
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="",  # Default to bucket root - bucket root mounts to /app/latest_model
        help="GCS prefix/path in bucket (default: empty string for root). "
             "Files at gs://bucket/ will appear at /app/latest_model/ when bucket root mounts to /app/latest_model/"
    )
    parser.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Skip files that already exist in bucket"
    )
    parser.add_argument(
        "--clear-before-upload",
        action="store_true",
        help="Clear all objects under the prefix before uploading (ensures no stale files)"
    )
    
    args = parser.parse_args()
    
    # Resolve local directory path
    local_dir = Path(args.dir).resolve()
    
    if not local_dir.exists():
        # Try relative to script location
        script_dir = Path(__file__).parent.parent.parent
        local_dir = (script_dir / args.dir).resolve()
    
    upload_directory_to_gcs(
        local_dir=str(local_dir),
        bucket_name=args.bucket,
        gcs_prefix=args.prefix,
        overwrite=not args.no_overwrite,
        clear_before_upload=args.clear_before_upload
    )


if __name__ == "__main__":
    main()

