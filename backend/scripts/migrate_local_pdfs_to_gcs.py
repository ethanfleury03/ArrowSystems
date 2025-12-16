"""
Migration script to upload local PDFs to Google Cloud Storage.

This script:
1. Queries the database for documents with local file_path but no gcs_path
2. Uploads each local PDF to GCS using the same object naming convention
3. Updates Document.gcs_path accordingly
4. Logs per-document success/failure

Safe to rerun: skips documents that already have gcs_path set.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.utils.gcs_client import upload_file, parse_gcs_path, blob_exists
from backend.config.env import settings

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for GCS (remove special chars, spaces)."""
    import re
    sanitized = re.sub(r'[^\w\s.-]', '_', filename)
    sanitized = sanitized.replace(' ', '_')
    return sanitized


def get_gcs_object_name(metadata_id: str, filename: str, prefix: str) -> str:
    """
    Generate GCS object name using the same convention as upload endpoint:
    {prefix}{metadata_id}/{sanitized_filename}
    """
    sanitized = sanitize_filename(filename)
    return f"{prefix}{metadata_id}/{sanitized}"


def migrate_document(metadata: DocumentIngestionMetadata, session) -> Dict[str, Any]:
    """
    Migrate a single document from local storage to GCS.
    
    Returns:
        Dict with success status, gcs_path, and any error message
    """
    result = {
        "metadata_id": metadata.id,
        "filename": metadata.filename,
        "success": False,
        "gcs_path": None,
        "error": None,
        "skipped": False,
    }
    
    # Check if document already has gcs_path in Document table
    doc_record = session.query(Document).filter(Document.file_name == metadata.filename).first()
    if doc_record and doc_record.gcs_path:
        # Verify the GCS object exists
        bucket_name, blob_name = parse_gcs_path(doc_record.gcs_path)
        if bucket_name and blob_name and blob_exists(bucket_name, blob_name):
            logger.info(
                f"Document {metadata.filename} already has GCS path: {doc_record.gcs_path}. Skipping."
            )
            result["skipped"] = True
            result["success"] = True
            result["gcs_path"] = doc_record.gcs_path
            return result
    
    # Check if local file exists
    if not metadata.file_path or not os.path.exists(metadata.file_path):
        result["error"] = f"Local file not found: {metadata.file_path}"
        logger.warning(f"⚠️  {metadata.filename}: {result['error']}")
        return result
    
    # Check if bucket is configured
    if not settings.DOCS_GCS_BUCKET:
        result["error"] = "DOCS_GCS_BUCKET not configured"
        logger.error(f"❌ {metadata.filename}: {result['error']}")
        return result
    
    # Generate GCS object name
    gcs_object_name = get_gcs_object_name(
        metadata.id,
        metadata.filename,
        settings.DOCS_GCS_PREFIX
    )
    
    # Check if object already exists in GCS
    if blob_exists(settings.DOCS_GCS_BUCKET, gcs_object_name):
        logger.info(
            f"GCS object already exists: gs://{settings.DOCS_GCS_BUCKET}/{gcs_object_name}. "
            f"Updating database record."
        )
        gcs_path = f"gs://{settings.DOCS_GCS_BUCKET}/{gcs_object_name}"
    else:
        # Upload to GCS
        logger.info(
            f"Uploading {metadata.filename} to gs://{settings.DOCS_GCS_BUCKET}/{gcs_object_name}..."
        )
        gcs_path = upload_file(
            bucket_name=settings.DOCS_GCS_BUCKET,
            object_name=gcs_object_name,
            local_path=metadata.file_path,
            content_type="application/pdf"
        )
        
        if not gcs_path:
            result["error"] = "Failed to upload to GCS"
            logger.error(f"❌ {metadata.filename}: {result['error']}")
            return result
    
    # Update Document table with gcs_path
    try:
        if doc_record:
            doc_record.gcs_path = gcs_path
            doc_record.updated_at = metadata.updated_at
        else:
            # Create new Document record
            doc_record = Document(
                file_name=metadata.filename,
                gcs_path=gcs_path,
                display_name=metadata.filename,
                machine_model=metadata.machine_model,
                file_size_bytes=metadata.file_size_bytes,
                is_active=True,
                requires_admin_review=False,
            )
            session.add(doc_record)
        
        session.commit()
        result["success"] = True
        result["gcs_path"] = gcs_path
        logger.info(f"✅ {metadata.filename}: Migrated to {gcs_path}")
        
    except Exception as e:
        session.rollback()
        result["error"] = f"Failed to update database: {str(e)}"
        logger.error(f"❌ {metadata.filename}: {result['error']}", exc_info=True)
    
    return result


def main(dry_run: bool = False):
    """
    Main migration function.
    
    Args:
        dry_run: If True, only show what would be migrated without making changes
    """
    if not settings.DOCS_GCS_BUCKET:
        logger.error("❌ DOCS_GCS_BUCKET environment variable not set. Cannot proceed.")
        sys.exit(1)
    
    logger.info("=" * 70)
    logger.info("PDF Migration to GCS")
    logger.info("=" * 70)
    logger.info(f"GCS Bucket: {settings.DOCS_GCS_BUCKET}")
    logger.info(f"GCS Prefix: {settings.DOCS_GCS_PREFIX}")
    logger.info(f"Dry Run: {dry_run}")
    logger.info("")
    
    session = SessionLocal()
    try:
        # Find all documents with local file_path but potentially missing gcs_path
        # Query DocumentIngestionMetadata for documents with file_path
        metadata_list = session.query(DocumentIngestionMetadata).filter(
            DocumentIngestionMetadata.file_path.isnot(None)
        ).all()
        
        logger.info(f"Found {len(metadata_list)} documents with local file_path")
        logger.info("")
        
        if dry_run:
            logger.info("DRY RUN MODE - No changes will be made")
            logger.info("")
        
        results = {
            "total": len(metadata_list),
            "success": 0,
            "skipped": 0,
            "failed": 0,
            "errors": [],
        }
        
        for metadata in metadata_list:
            if dry_run:
                # In dry run, just check what would happen
                doc_record = session.query(Document).filter(
                    Document.file_name == metadata.filename
                ).first()
                
                if doc_record and doc_record.gcs_path:
                    logger.info(f"✓ {metadata.filename}: Already has GCS path (would skip)")
                    results["skipped"] += 1
                elif metadata.file_path and os.path.exists(metadata.file_path):
                    gcs_object_name = get_gcs_object_name(
                        metadata.id,
                        metadata.filename,
                        settings.DOCS_GCS_PREFIX
                    )
                    logger.info(
                        f"→ {metadata.filename}: Would upload to "
                        f"gs://{settings.DOCS_GCS_BUCKET}/{gcs_object_name}"
                    )
                    results["success"] += 1
                else:
                    logger.warning(f"✗ {metadata.filename}: Local file not found (would fail)")
                    results["failed"] += 1
                    results["errors"].append({
                        "filename": metadata.filename,
                        "error": "Local file not found"
                    })
            else:
                # Actually migrate
                result = migrate_document(metadata, session)
                
                if result["skipped"]:
                    results["skipped"] += 1
                elif result["success"]:
                    results["success"] += 1
                else:
                    results["failed"] += 1
                    results["errors"].append({
                        "filename": result["filename"],
                        "error": result["error"]
                    })
        
        # Print summary
        logger.info("")
        logger.info("=" * 70)
        logger.info("Migration Summary")
        logger.info("=" * 70)
        logger.info(f"Total documents: {results['total']}")
        logger.info(f"✅ Successfully migrated: {results['success']}")
        logger.info(f"⏭️  Skipped (already migrated): {results['skipped']}")
        logger.info(f"❌ Failed: {results['failed']}")
        
        if results["errors"]:
            logger.info("")
            logger.info("Errors:")
            for error in results["errors"]:
                logger.info(f"  - {error['filename']}: {error['error']}")
        
        logger.info("=" * 70)
        
    finally:
        session.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Migrate local PDFs to Google Cloud Storage")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be migrated without making changes"
    )
    
    args = parser.parse_args()
    main(dry_run=args.dry_run)


