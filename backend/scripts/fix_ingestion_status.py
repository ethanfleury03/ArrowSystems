"""
Fix ingestion status for documents stuck in "in progress" states.

This script normalizes ingestion_status values for documents that are stuck
in "in progress" states (e.g., REBUILDING_INDEX, PENDING_INGESTION) when
app-based ingestion is disabled.

IMPORTANT: Only run this script when you know the index is already built
externally by the GPU ingestion pipeline. This script assumes that all
documents with "in progress" statuses are actually complete and managed
by the external pipeline.

Usage:
    python -m backend.scripts.fix_ingestion_status

Or from project root:
    python backend/scripts/fix_ingestion_status.py
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata


# Statuses that indicate "in progress" ingestion
PROGRESS_STATUSES = {
    "REBUILDING_INDEX",
    "PENDING_INGESTION",
    "CHUNKING",
    "READY_FOR_EMBEDDING",
    "EMBEDDING",
    "DELETING",
}

# Target status for normalized documents
TARGET_STATUS = "COMPLETE"


def main():
    """Normalize ingestion statuses for documents stuck in progress states."""
    print("=" * 60)
    print("Fix Ingestion Status Script")
    print("=" * 60)
    print()
    print("This script will update documents with 'in progress' ingestion statuses")
    print("to 'COMPLETE' status, assuming they are managed by external GPU pipeline.")
    print()
    
    with SessionLocal() as session:
        # Find all documents with in-progress statuses
        docs = (
            session.query(DocumentIngestionMetadata)
            .filter(DocumentIngestionMetadata.status.in_(PROGRESS_STATUSES))
            .all()
        )
        
        print(f"Found {len(docs)} documents with in-progress ingestion_status:")
        print()
        
        # Group by status for reporting
        status_counts = {}
        for doc in docs:
            status = doc.status
            status_counts[status] = status_counts.get(status, 0) + 1
        
        for status, count in sorted(status_counts.items()):
            print(f"  {status}: {count} document(s)")
        
        if len(docs) == 0:
            print()
            print("No documents need updating. All ingestion statuses are already normalized.")
            return
        
        print()
        response = input(f"Update {len(docs)} document(s) to status '{TARGET_STATUS}'? (yes/no): ")
        
        if response.lower() not in ('yes', 'y'):
            print("Aborted.")
            return
        
        print()
        print("Updating documents...")
        
        # Update all matching documents
        updated_count = 0
        for doc in docs:
            old_status = doc.status
            doc.status = TARGET_STATUS
            # Clear any error messages since we're marking as complete
            if doc.error_message:
                doc.error_message = None
            updated_count += 1
            print(f"  Updated {doc.filename}: {old_status} -> {TARGET_STATUS}")
        
        # Commit changes
        session.commit()
        
        print()
        print("=" * 60)
        print(f"Successfully updated {updated_count} document(s) to '{TARGET_STATUS}' status.")
        print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

