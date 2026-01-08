#!/usr/bin/env python3
"""
Quick script to mark all FAILED documents as COMPLETE.
Run this after successful full ingestion.
"""

import sys
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata

def main():
    print("=" * 60)
    print("Updating FAILED documents to COMPLETE")
    print("=" * 60)
    print()
    
    try:
        with SessionLocal() as session:
            # Find all FAILED documents
            failed_docs = (
                session.query(DocumentIngestionMetadata)
                .filter(DocumentIngestionMetadata.status == "FAILED")
                .order_by(DocumentIngestionMetadata.filename)
                .all()
            )
            
            if not failed_docs:
                print("✅ No failed documents found. All documents are already processed.")
                return 0
            
            print(f"Found {len(failed_docs)} failed document(s):")
            for doc in failed_docs:
                print(f"  - {doc.filename}")
            print()
            
            # Update all to COMPLETE
            updated_count = 0
            for doc in failed_docs:
                old_status = doc.status
                doc.status = "COMPLETE"
                doc.error_message = None  # Clear error message
                updated_count += 1
                print(f"  ✅ {doc.filename}: {old_status} -> COMPLETE")
            
            # Commit changes
            session.commit()
            
            print()
            print("=" * 60)
            print(f"✅ Successfully updated {updated_count} document(s) to COMPLETE")
            print("=" * 60)
            print()
            print("Please refresh your UI to see the updated status.")
            
            return 0
            
    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

