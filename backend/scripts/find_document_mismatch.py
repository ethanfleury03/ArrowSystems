#!/usr/bin/env python3
"""
Find documents that exist in Document table but not in DocumentIngestionMetadata table,
or vice versa, to explain the count mismatch between /documents (58) and /admin/documents (55).
"""

import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

# Load .env file if it exists
try:
    from dotenv import load_dotenv
    env_path = project_root / '.env'
    if env_path.exists():
        load_dotenv(env_path)
except ImportError:
    pass  # dotenv not available, rely on environment variables

from backend.utils.db import SessionLocal, Document, DocumentIngestionMetadata
from sqlalchemy import and_

def main():
    """Find documents causing the mismatch."""
    session = SessionLocal()
    try:
        # Get all active Document records
        active_docs = session.query(Document).filter(Document.is_active == True).all()
        print(f"Active Document records: {len(active_docs)}")
        
        # Get all DocumentIngestionMetadata records
        all_metadata = session.query(DocumentIngestionMetadata).all()
        print(f"DocumentIngestionMetadata records: {len(all_metadata)}")
        
        # Create sets of filenames for comparison
        doc_filenames = {doc.file_name for doc in active_docs}
        metadata_filenames = {meta.filename for meta in all_metadata}
        
        # Find documents in Document but not in DocumentIngestionMetadata
        docs_without_metadata = doc_filenames - metadata_filenames
        print(f"\nDocuments in Document table (active) but NOT in DocumentIngestionMetadata: {len(docs_without_metadata)}")
        if docs_without_metadata:
            print("Missing metadata records:")
            for filename in sorted(docs_without_metadata):
                doc = next((d for d in active_docs if d.file_name == filename), None)
                if doc:
                    print(f"  - {filename}")
                    print(f"    Document ID: {doc.id}")
                    print(f"    GCS Path: {doc.gcs_path}")
                    print(f"    Is Active: {doc.is_active}")
                    print(f"    File Size: {doc.file_size_bytes}")
                    print()
        
        # Find metadata records without corresponding active Document records
        metadata_without_docs = metadata_filenames - doc_filenames
        print(f"\nDocumentIngestionMetadata records without corresponding active Document: {len(metadata_without_docs)}")
        if metadata_without_docs:
            print("Orphaned metadata records:")
            for filename in sorted(metadata_without_docs):
                meta = next((m for m in all_metadata if m.filename == filename), None)
                if meta:
                    print(f"  - {filename}")
                    print(f"    Metadata ID: {meta.id}")
                    print(f"    Status: {meta.status}")
                    print(f"    File Path: {meta.file_path}")
                    print()
        
        # Check for "anyjet_user_guide_v1.1.pdf" specifically
        target_filename = "anyjet_user_guide_v1.1.pdf"
        print(f"\n=== Checking for '{target_filename}' ===")
        doc_record = session.query(Document).filter(Document.file_name == target_filename).first()
        meta_record = session.query(DocumentIngestionMetadata).filter(DocumentIngestionMetadata.filename == target_filename).first()
        
        if doc_record:
            print(f"Document record found:")
            print(f"  ID: {doc_record.id}")
            print(f"  Is Active: {doc_record.is_active}")
            print(f"  GCS Path: {doc_record.gcs_path}")
            print(f"  File Size: {doc_record.file_size_bytes}")
        else:
            print("Document record NOT found")
        
        if meta_record:
            print(f"DocumentIngestionMetadata record found:")
            print(f"  ID: {meta_record.id}")
            print(f"  Status: {meta_record.status}")
            print(f"  File Path: {meta_record.file_path}")
        else:
            print("DocumentIngestionMetadata record NOT found")
        
        # Summary
        print(f"\n=== Summary ===")
        print(f"Active Document records: {len(active_docs)}")
        print(f"DocumentIngestionMetadata records: {len(all_metadata)}")
        print(f"Difference: {len(active_docs) - len(all_metadata)}")
        print(f"\nThis explains why:")
        print(f"  - /documents endpoint (sidebar) shows: {len(active_docs)} (queries Document table)")
        print(f"  - /admin/documents endpoint shows: {len(all_metadata)} (queries DocumentIngestionMetadata table)")
        
    finally:
        session.close()

if __name__ == "__main__":
    # Ensure DATABASE_URL is set
    if not os.getenv("DATABASE_URL"):
        print("ERROR: DATABASE_URL environment variable is required")
        print("Set it in your .env file or export it before running this script")
        sys.exit(1)
    
    main()
