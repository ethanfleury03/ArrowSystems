#!/usr/bin/env python3
"""
Find documents that exist in database but are missing from GCS bucket.

This helps identify the mismatch between DB count (58) and GCS count (55).
"""

import os
import sys
from pathlib import Path

# Add backend to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir.parent))

from backend.utils.db import SessionLocal, Document
from backend.utils.gcs_client import object_exists, list_object_names
from backend.config.env import settings


def find_missing_documents():
    """Find documents in DB that are missing from GCS."""
    session = SessionLocal()
    try:
        # Get all active documents
        all_docs = session.query(Document).filter(Document.is_active == True).all()
        
        print(f"📊 Found {len(all_docs)} active documents in database")
        print()
        
        # Get GCS object names
        gcs_objects = []
        if settings.DOCS_GCS_BUCKET:
            print(f"☁️  Listing objects from bucket: {settings.DOCS_GCS_BUCKET}")
            print(f"   Prefix: {settings.DOCS_GCS_PREFIX or '(root)'}")
            gcs_objects = list_object_names(settings.DOCS_GCS_BUCKET, settings.DOCS_GCS_PREFIX or "")
            print(f"   Found {len(gcs_objects)} objects in GCS")
        else:
            print("⚠️  DOCS_GCS_BUCKET not configured")
            return
        
        print()
        print("=" * 70)
        print("Checking for missing documents...")
        print("=" * 70)
        print()
        
        missing_docs = []
        docs_without_gcs_path = []
        
        for doc in all_docs:
            filename = doc.file_name
            
            # Check if document has GCS path
            if not doc.gcs_path:
                docs_without_gcs_path.append({
                    "filename": filename,
                    "doc_id": doc.id,
                    "reason": "No gcs_path in Document table"
                })
                continue
            
            # Check if GCS object exists
            if not object_exists(doc.gcs_path):
                missing_docs.append({
                    "filename": filename,
                    "doc_id": doc.id,
                    "gcs_path": doc.gcs_path,
                    "reason": "GCS object not found"
                })
        
        # Also check documents by filename (in case gcs_path is wrong but file exists)
        gcs_filenames = set()
        for obj_name in gcs_objects:
            # Extract just the filename from the object name
            filename = os.path.basename(obj_name)
            gcs_filenames.add(filename)
        
        # Check if any missing docs might exist under different paths
        print(f"🔍 Documents missing from GCS: {len(missing_docs)}")
        if missing_docs:
            print()
            for missing in missing_docs:
                filename = missing["filename"]
                gcs_path = missing["gcs_path"]
                
                # Check if filename exists in GCS (maybe under different path)
                filename_exists = filename in gcs_filenames
                
                print(f"  ❌ {filename}")
                print(f"     Document ID: {missing['doc_id']}")
                print(f"     Expected GCS path: {gcs_path}")
                print(f"     Filename exists in bucket: {'✅ YES' if filename_exists else '❌ NO'}")
                if filename_exists:
                    # Find the actual object name
                    matching_objects = [obj for obj in gcs_objects if os.path.basename(obj) == filename]
                    if matching_objects:
                        print(f"     Actual GCS path(s):")
                        for obj in matching_objects:
                            print(f"       - gs://{settings.DOCS_GCS_BUCKET}/{obj}")
                print()
        
        print(f"📋 Documents without gcs_path: {len(docs_without_gcs_path)}")
        if docs_without_gcs_path:
            print()
            for doc in docs_without_gcs_path:
                filename = doc["filename"]
                filename_exists = filename in gcs_filenames
                print(f"  ⚠️  {filename}")
                print(f"     Document ID: {doc['doc_id']}")
                print(f"     Filename exists in bucket: {'✅ YES' if filename_exists else '❌ NO'}")
                if filename_exists:
                    matching_objects = [obj for obj in gcs_objects if os.path.basename(obj) == filename]
                    if matching_objects:
                        print(f"     Found at:")
                        for obj in matching_objects:
                            print(f"       - gs://{settings.DOCS_GCS_BUCKET}/{obj}")
                print()
        
        # Summary
        print("=" * 70)
        print("Summary:")
        print(f"   Database documents (active): {len(all_docs)}")
        print(f"   GCS objects: {len(gcs_objects)}")
        print(f"   Missing from GCS: {len(missing_docs)}")
        print(f"   Without gcs_path: {len(docs_without_gcs_path)}")
        print()
        
        # Check specifically for anyjet_user_guide_v1.1.pdf
        print("=" * 70)
        print("Checking for 'anyjet_user_guide_v1.1.pdf'...")
        print("=" * 70)
        print()
        
        anyjet_doc = session.query(Document).filter(
            Document.file_name.like('%anyjet%')
        ).first()
        
        if anyjet_doc:
            print(f"✅ Found in database:")
            print(f"   Filename: {anyjet_doc.file_name}")
            print(f"   Document ID: {anyjet_doc.id}")
            print(f"   GCS Path: {anyjet_doc.gcs_path or '(not set)'}")
            print(f"   Is Active: {anyjet_doc.is_active}")
            print()
            
            if anyjet_doc.gcs_path:
                exists = object_exists(anyjet_doc.gcs_path)
                print(f"   GCS object exists: {'✅ YES' if exists else '❌ NO'}")
                if not exists:
                    print(f"   Expected path: {anyjet_doc.gcs_path}")
            else:
                print(f"   ⚠️  No gcs_path set")
            
            # Check if filename exists in bucket under any path
            anyjet_in_bucket = any('anyjet' in obj.lower() for obj in gcs_objects)
            if anyjet_in_bucket:
                matching = [obj for obj in gcs_objects if 'anyjet' in obj.lower()]
                print(f"   Found in bucket at:")
                for obj in matching:
                    print(f"     - gs://{settings.DOCS_GCS_BUCKET}/{obj}")
            else:
                print(f"   ❌ Not found in bucket under any path")
        else:
            print("❌ 'anyjet_user_guide_v1.1.pdf' not found in database")
        
    finally:
        session.close()


if __name__ == "__main__":
    find_missing_documents()
