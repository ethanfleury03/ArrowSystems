#!/usr/bin/env python3
"""
Document diagnostic script.

Reads DB and GCS and prints:
- DB metadata count
- DB documents count
- GCS object count under prefix
- Orphan metadata rows (metadata_id, gcs_path)
- GCS objects without DB row (object key)

Usage:
    python -m backend.scripts.doc_diagnose
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir.parent))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.utils.gcs_client import list_objects, object_exists, get_gcs_client
from backend.config.env import settings


def main():
    """Run document diagnostics."""
    print("=" * 70)
    print("Document Count Verification")
    print("=" * 70)
    print()
    
    # Database counts
    print("📊 Database Counts:")
    session = SessionLocal()
    try:
        count_metadata = session.query(DocumentIngestionMetadata).count()
        count_documents = session.query(Document).count()
        
        print(f"   DocumentIngestionMetadata records: {count_metadata}")
        print(f"   Document records: {count_documents}")
        print()
    except Exception as e:
        print(f"   ❌ Error querying database: {e}")
        return
    finally:
        session.close()
    
    # GCS counts
    print("☁️  GCS Storage:")
    if not settings.DOCS_GCS_BUCKET:
        print("   ⚠️  DOCS_GCS_BUCKET not configured")
        return
    
    gcs_client = get_gcs_client()
    if not gcs_client:
        print("   ❌ GCS client not available. Check credentials.")
        return
    
    try:
        prefix = settings.DOCS_GCS_PREFIX or ""
        gcs_objects = list_objects(settings.DOCS_GCS_BUCKET, prefix)
        print(f"   Bucket: {settings.DOCS_GCS_BUCKET}")
        print(f"   Prefix: {prefix}")
        print(f"   Objects found: {len(gcs_objects)}")
        print()
    except Exception as e:
        print(f"   ❌ Error listing GCS objects: {e}")
        return
    
    # Find orphans
    print("🔍 Orphan Detection:")
    session = SessionLocal()
    try:
        all_metadata = session.query(DocumentIngestionMetadata).all()
        all_documents = session.query(Document).all()
        
        # Build set of GCS paths in DB
        gcs_paths_in_db = set()
        for doc in all_documents:
            if doc.gcs_path:
                gcs_paths_in_db.add(doc.gcs_path)
        for meta in all_metadata:
            if meta.file_path and meta.file_path.startswith('gs://'):
                gcs_paths_in_db.add(meta.file_path)
        
        # Find orphaned metadata (DB says exists, GCS missing)
        orphan_metadata = []
        for meta in all_metadata:
            gcs_path = None
            doc = session.query(Document).filter(
                Document.file_name == meta.filename
            ).first()
            
            if doc and doc.gcs_path:
                gcs_path = doc.gcs_path
            elif meta.file_path and meta.file_path.startswith('gs://'):
                gcs_path = meta.file_path
            
            if gcs_path:
                if not object_exists(gcs_path):
                    orphan_metadata.append({
                        "metadata_id": meta.id,
                        "filename": meta.filename,
                        "gcs_path": gcs_path,
                    })
        
        # Find GCS objects without DB
        gcs_objects_without_db = []
        for obj_name in gcs_objects:
            gcs_path = f"gs://{settings.DOCS_GCS_BUCKET}/{obj_name}"
            if gcs_path not in gcs_paths_in_db:
                gcs_objects_without_db.append(obj_name)
        
        print(f"   Orphaned metadata records (GCS missing): {len(orphan_metadata)}")
        if orphan_metadata:
            print("   Orphan details:")
            for orphan in orphan_metadata[:10]:  # Show first 10
                print(f"      - {orphan['metadata_id']}: {orphan['filename']} ({orphan['gcs_path']})")
            if len(orphan_metadata) > 10:
                print(f"      ... and {len(orphan_metadata) - 10} more")
        
        print(f"   GCS objects without DB records: {len(gcs_objects_without_db)}")
        if gcs_objects_without_db:
            print("   GCS objects without DB:")
            for obj_name in gcs_objects_without_db[:10]:  # Show first 10
                print(f"      - {obj_name}")
            if len(gcs_objects_without_db) > 10:
                print(f"      ... and {len(gcs_objects_without_db) - 10} more")
        
        print()
        
        # Summary
        print("=" * 70)
        print("Summary:")
        print(f"   DB metadata: {count_metadata}")
        print(f"   DB documents: {count_documents}")
        print(f"   GCS objects: {len(gcs_objects)}")
        print(f"   Orphaned metadata: {len(orphan_metadata)}")
        print(f"   GCS objects without DB: {len(gcs_objects_without_db)}")
        
        if len(orphan_metadata) > 0 or len(gcs_objects_without_db) > 0:
            print()
            print("   ⚠️  Data inconsistency detected!")
            print("   Use DELETE /admin/documents/orphans to clean up orphaned records")
        else:
            print()
            print("   ✅ All records are consistent")
        
    except Exception as e:
        print(f"   ❌ Error finding orphans: {e}")
        import traceback
        traceback.print_exc()
    finally:
        session.close()
    
    print("=" * 70)


if __name__ == "__main__":
    main()

