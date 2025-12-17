"""
Find and optionally delete orphaned document records.

Orphaned records are DocumentIngestionMetadata records that:
- Have no GCS path (file_path is NULL or not gs://)
- Have no Document record with gcs_path
- Should not exist (documents must be in GCS)

This script helps identify and clean up these orphaned records.
"""

import os
import sys
import argparse
from pathlib import Path

# Add parent directory to path for backend imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

from sqlalchemy.orm import Session
from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.logging_config import get_logger

logger = get_logger(__name__)


def find_orphaned_documents(dry_run: bool = True) -> list:
    """
    Find orphaned document records (no GCS path).
    
    Args:
        dry_run: If True, only report findings. If False, delete orphaned records.
    
    Returns:
        List of orphaned record dictionaries
    """
    session: Session = None
    orphaned_records = []
    
    try:
        session = SessionLocal()
        
        # Find all DocumentIngestionMetadata records
        all_metadata = session.query(DocumentIngestionMetadata).all()
        
        for meta in all_metadata:
            # Check if it has a valid GCS path
            has_gcs_path = False
            
            # Check Document table first
            doc = session.query(Document).filter(
                Document.file_name == meta.filename
            ).first()
            
            if doc and doc.gcs_path and doc.gcs_path.startswith('gs://'):
                has_gcs_path = True
            
            # Check metadata.file_path
            if not has_gcs_path and meta.file_path and meta.file_path.startswith('gs://'):
                has_gcs_path = True
            
            # If no GCS path, it's orphaned
            if not has_gcs_path:
                orphaned_records.append({
                    'metadata_id': meta.id,
                    'filename': meta.filename,
                    'status': meta.status,
                    'file_path': meta.file_path,
                    'document_id': doc.id if doc else None,
                    'document_gcs_path': doc.gcs_path if doc else None,
                    'created_at': meta.created_at.isoformat() if meta.created_at else None,
                })
        
        # Report findings
        print("=" * 70)
        print("ORPHANED DOCUMENT RECORDS")
        print("=" * 70)
        print()
        
        if not orphaned_records:
            print("✅ No orphaned records found. All documents have valid GCS paths.")
            return []
        
        print(f"⚠️  Found {len(orphaned_records)} orphaned records (no GCS path):")
        print()
        
        for i, record in enumerate(orphaned_records, 1):
            print(f"{i}. {record['filename']}")
            print(f"   Metadata ID: {record['metadata_id']}")
            print(f"   Status: {record['status']}")
            print(f"   Metadata file_path: {record['file_path'] or 'NULL'}")
            print(f"   Document ID: {record['document_id'] or 'None'}")
            print(f"   Document gcs_path: {record['document_gcs_path'] or 'NULL'}")
            print(f"   Created: {record['created_at']}")
            print()
        
        # Delete if not dry run
        if not dry_run:
            confirm = input(f"Delete these {len(orphaned_records)} orphaned records? Type 'yes' to confirm: ")
            if confirm.lower() == 'yes':
                deleted_count = 0
                for record in orphaned_records:
                    try:
                        # Delete Document record if exists
                        if record['document_id']:
                            doc = session.query(Document).filter(
                                Document.id == record['document_id']
                            ).first()
                            if doc:
                                session.delete(doc)
                                print(f"   Deleted Document record: {record['filename']}")
                        
                        # Delete DocumentIngestionMetadata record
                        meta = session.query(DocumentIngestionMetadata).filter(
                            DocumentIngestionMetadata.id == record['metadata_id']
                        ).first()
                        if meta:
                            session.delete(meta)
                            print(f"   Deleted DocumentIngestionMetadata record: {record['filename']}")
                        
                        deleted_count += 1
                    except Exception as e:
                        logger.error(f"Failed to delete {record['filename']}: {e}")
                        session.rollback()
                        continue
                
                session.commit()
                print()
                print(f"✅ Deleted {deleted_count} orphaned records.")
            else:
                print("❌ Deletion cancelled.")
        else:
            print("=" * 70)
            print("DRY RUN MODE - No changes made")
            print("=" * 70)
            print()
            print("To actually delete these records, run:")
            print("  python backend/scripts/find_orphaned_documents.py --delete")
            print()
        
        return orphaned_records
        
    except Exception as e:
        logger.error(f"Error finding orphaned documents: {e}", exc_info=True)
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find and optionally delete orphaned document records.")
    parser.add_argument("--delete", action="store_true", help="Actually delete orphaned records (default: dry run)")
    args = parser.parse_args()
    
    find_orphaned_documents(dry_run=not args.delete)

