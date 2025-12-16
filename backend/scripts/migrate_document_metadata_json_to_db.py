"""
Migration Helper Script: Move document_metadata.json to Database

This is a ONE-OFF migration script to move data from data/document_metadata.json
into the documents table in PostgreSQL.

Usage:
    python -m backend.scripts.migrate_document_metadata_json_to_db

This script should ONLY be run once during the migration to GCP.
It is NOT used at runtime.

DO NOT import or use this script in production code.
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

# Add parent directory to path to allow imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from backend.utils.db import SessionLocal, Document
from backend.utils.document_metadata import infer_machine_model_from_filename

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METADATA_FILE = Path("data/document_metadata.json")


def migrate_json_to_db(
    json_file: Path = METADATA_FILE,
    gcs_bucket: str = None,
    dry_run: bool = False
) -> Dict[str, Any]:
    """
    Migrate document metadata from JSON file to database.
    
    Args:
        json_file: Path to document_metadata.json file
        gcs_bucket: Optional GCS bucket name for gcs_path generation
        dry_run: If True, don't actually write to database
    
    Returns:
        Dictionary with migration statistics
    """
    if not json_file.exists():
        logger.warning(f"JSON file not found: {json_file}")
        logger.info("No existing metadata to migrate. This is OK if starting fresh.")
        return {
            "success": True,
            "migrated": 0,
            "skipped": 0,
            "errors": 0,
            "message": "No JSON file found - starting fresh"
        }
    
    logger.info(f"Loading metadata from {json_file}")
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            metadata_dict = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON file: {e}")
        return {
            "success": False,
            "migrated": 0,
            "skipped": 0,
            "errors": 1,
            "message": f"Failed to load JSON: {e}"
        }
    
    logger.info(f"Found {len(metadata_dict)} documents in JSON file")
    
    session = SessionLocal()
    stats = {
        "migrated": 0,
        "skipped": 0,
        "errors": 0,
        "error_details": []
    }
    
    try:
        for filename, meta in metadata_dict.items():
            try:
                # Check if document already exists
                existing_doc = session.query(Document).filter(
                    Document.file_name == filename
                ).first()
                
                if existing_doc:
                    logger.debug(f"Skipping {filename} - already exists in database")
                    stats["skipped"] += 1
                    continue
                
                # Parse machine_model
                machine_model = meta.get("machine_model")
                machine_model_str = None
                
                if machine_model is not None:
                    if isinstance(machine_model, list):
                        if len(machine_model) > 1:
                            machine_model_str = json.dumps(machine_model)
                        elif len(machine_model) == 1:
                            machine_model_str = machine_model[0]
                    elif isinstance(machine_model, str):
                        machine_model_str = machine_model
                
                # If no machine_model, try to infer from filename
                if not machine_model_str:
                    inferred = infer_machine_model_from_filename(filename)
                    if inferred:
                        if len(inferred) > 1:
                            machine_model_str = json.dumps(inferred)
                        else:
                            machine_model_str = inferred[0]
                
                # Parse last_ingestion_date
                last_ingestion_date = None
                ingestion_date_str = meta.get("last_ingestion_date")
                if ingestion_date_str:
                    try:
                        last_ingestion_date = datetime.fromisoformat(
                            ingestion_date_str.replace('Z', '+00:00')
                        )
                    except (ValueError, AttributeError):
                        pass
                
                # Generate GCS path if bucket is provided
                gcs_path = None
                if gcs_bucket:
                    # Remove any existing gs:// prefix from bucket name
                    bucket = gcs_bucket.replace('gs://', '').replace('/', '')
                    gcs_path = f"gs://{bucket}/{filename}"
                
                # Create document record
                doc = Document(
                    file_name=filename,
                    gcs_path=gcs_path,
                    display_name=filename,  # Default to filename
                    machine_model=machine_model_str,
                    category=meta.get("category"),
                    product_family=meta.get("product_family"),
                    is_active=meta.get("is_active", True),
                    requires_admin_review=meta.get("requires_admin_review", False),
                    last_ingestion_date=last_ingestion_date or datetime.utcnow()
                )
                
                if not dry_run:
                    session.add(doc)
                    session.commit()
                    logger.info(f"Migrated: {filename}")
                else:
                    logger.info(f"[DRY RUN] Would migrate: {filename}")
                
                stats["migrated"] += 1
                
            except Exception as e:
                logger.error(f"Error migrating {filename}: {e}", exc_info=True)
                stats["errors"] += 1
                stats["error_details"].append({
                    "filename": filename,
                    "error": str(e)
                })
                session.rollback()
        
        if not dry_run:
            session.commit()
        
        logger.info(f"Migration complete: {stats['migrated']} migrated, {stats['skipped']} skipped, {stats['errors']} errors")
        
        return {
            "success": stats["errors"] == 0,
            **stats
        }
        
    finally:
        session.close()


def main():
    """Main entry point for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Migrate document_metadata.json to PostgreSQL documents table"
    )
    parser.add_argument(
        "--json-file",
        type=Path,
        default=METADATA_FILE,
        help=f"Path to document_metadata.json (default: {METADATA_FILE})"
    )
    parser.add_argument(
        "--gcs-bucket",
        type=str,
        default=None,
        help="GCS bucket name for gcs_path generation (e.g., rag-postgres-prod-docs)"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Don't actually write to database, just show what would be migrated"
    )
    
    args = parser.parse_args()
    
    # Get GCS bucket from environment if not provided
    gcs_bucket = args.gcs_bucket or os.getenv("DOCS_BUCKET_NAME")
    
    if not args.dry_run:
        logger.info("Starting migration from JSON to database...")
        if gcs_bucket:
            logger.info(f"Using GCS bucket: {gcs_bucket}")
    else:
        logger.info("[DRY RUN MODE] No changes will be made to the database")
    
    result = migrate_json_to_db(
        json_file=args.json_file,
        gcs_bucket=gcs_bucket,
        dry_run=args.dry_run
    )
    
    if result["success"]:
        logger.info("✅ Migration completed successfully")
        sys.exit(0)
    else:
        logger.error(f"❌ Migration completed with {result['errors']} errors")
        if result.get("error_details"):
            for detail in result["error_details"]:
                logger.error(f"  - {detail['filename']}: {detail['error']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

















