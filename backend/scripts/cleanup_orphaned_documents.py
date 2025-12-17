#!/usr/bin/env python3
"""
Cleanup Orphaned Document Records

Finds and fixes/removes document records that have no GCS paths.
Useful for cleaning up after failed uploads or incomplete deletions.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.config.env import settings
from sqlalchemy import or_


def find_orphaned_records(dry_run=True):
    """
    Find document records that have no GCS paths.
    
    Args:
        dry_run: If True, only show what would be fixed/deleted without making changes
    """
    session = SessionLocal()
    
    try:
        # Find all DocumentIngestionMetadata records
        all_metadata = session.query(DocumentIngestionMetadata).all()
        
        # Find records that won't be picked up by ingest.py
        # (no GCS path in either Document.gcs_path or Metadata.file_path)
        orphaned = []
        
        for meta in all_metadata:
            # Check if Document record exists
            doc = session.query(Document).filter(Document.file_name == meta.filename).first()
            
            # Check if it has a GCS path
            has_gcs_in_doc = doc and doc.gcs_path and doc.gcs_path.startswith('gs://')
            has_gcs_in_meta = meta.file_path and meta.file_path.startswith('gs://')
            
            if not has_gcs_in_doc and not has_gcs_in_meta:
                orphaned.append({
                    'metadata': meta,
                    'document': doc,
                    'reason': 'No GCS path in either table'
                })
        
        print("=" * 70)
        print("Orphaned Document Records")
        print("=" * 70)
        print()
        print(f"Found {len(orphaned)} orphaned records:")
        print()
        
        for i, item in enumerate(orphaned, 1):
            meta = item['metadata']
            doc = item['document']
            
            print(f"{i}. {meta.filename}")
            print(f"   Metadata ID: {meta.id}")
            print(f"   Status: {meta.status}")
            print(f"   Metadata file_path: {meta.file_path or 'NULL'}")
            print(f"   Document exists: {doc is not None}")
            if doc:
                print(f"   Document gcs_path: {doc.gcs_path or 'NULL'}")
                print(f"   Document is_active: {doc.is_active}")
            print(f"   Reason: {item['reason']}")
            print()
        
        if not orphaned:
            print("✅ No orphaned records found!")
            return []
        
        if dry_run:
            print("=" * 70)
            print("DRY RUN MODE - No changes made")
            print("=" * 70)
            print()
            print("To actually fix/delete these records, run:")
            print("  python backend/scripts/cleanup_orphaned_documents.py --fix")
            print("  OR")
            print("  python backend/scripts/cleanup_orphaned_documents.py --delete")
            print()
            return orphaned
        
        return orphaned
        
    finally:
        session.close()


def fix_orphaned_records(orphaned, action='delete'):
    """
    Fix or delete orphaned records.
    
    Args:
        orphaned: List of orphaned record info
        action: 'delete' to remove records, 'mark_inactive' to mark as inactive
    """
    session = SessionLocal()
    
    try:
        fixed = 0
        deleted = 0
        
        for item in orphaned:
            meta = item['metadata']
            doc = item['document']
            
            if action == 'delete':
                # Delete both metadata and document records
                if doc:
                    session.delete(doc)
                    print(f"   Deleted Document record: {meta.filename}")
                
                session.delete(meta)
                print(f"   Deleted DocumentIngestionMetadata record: {meta.filename}")
                deleted += 1
                
            elif action == 'mark_inactive':
                # Mark document as inactive instead of deleting
                if doc:
                    doc.is_active = False
                    session.add(doc)
                    print(f"   Marked Document as inactive: {meta.filename}")
                
                # Update metadata status
                meta.status = "DELETED"
                session.add(meta)
                print(f"   Marked DocumentIngestionMetadata as DELETED: {meta.filename}")
                fixed += 1
        
        session.commit()
        
        print()
        if action == 'delete':
            print(f"✅ Deleted {deleted} orphaned record(s)")
        else:
            print(f"✅ Marked {fixed} orphaned record(s) as inactive/deleted")
        
    except Exception as e:
        session.rollback()
        print(f"❌ Error: {e}")
        raise
    finally:
        session.close()


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Cleanup orphaned document records')
    parser.add_argument('--fix', action='store_true', help='Mark orphaned records as inactive (instead of deleting)')
    parser.add_argument('--delete', action='store_true', help='Delete orphaned records permanently')
    parser.add_argument('--dry-run', action='store_true', default=True, help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Find orphaned records
    orphaned = find_orphaned_records(dry_run=(not args.fix and not args.delete))
    
    if not orphaned:
        return 0
    
    # If fix or delete specified, do it
    if args.delete:
        print()
        print("⚠️  WARNING: This will PERMANENTLY DELETE the orphaned records!")
        response = input("Are you sure? (yes/no): ")
        if response.lower() == 'yes':
            fix_orphaned_records(orphaned, action='delete')
        else:
            print("Cancelled.")
            return 0
    elif args.fix:
        print()
        print("⚠️  This will mark orphaned records as inactive/deleted (not permanently delete)")
        response = input("Continue? (yes/no): ")
        if response.lower() == 'yes':
            fix_orphaned_records(orphaned, action='mark_inactive')
        else:
            print("Cancelled.")
            return 0
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


