"""
Mark failed documents as COMPLETE if they exist in the current index.

After a successful full ingestion, this script checks which FAILED documents
are now in the index and updates their status to COMPLETE.

Usage:
    python backend/scripts/mark_failed_as_complete.py [--dry-run]
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata
from backend.logging_config import get_logger

logger = get_logger(__name__)


def check_document_in_index(filename: str, index_dir: str = "/workspace/latest_model") -> bool:
    """
    Check if a document exists in the index by looking for its filename in docstore.
    
    Args:
        filename: The filename to check
        index_dir: Path to the index directory
        
    Returns:
        True if document found in index, False otherwise
    """
    try:
        import json
        
        docstore_path = Path(index_dir) / "docstore.json"
        if not docstore_path.exists():
            logger.warning(f"Index docstore not found at {docstore_path}")
            return False
        
        with open(docstore_path, 'r', encoding='utf-8') as f:
            docstore = json.load(f)
        
        # Check if filename exists in any node's metadata
        nodes = docstore.get("docstore", {}).get("data", {})
        for node_id, node_data in nodes.items():
            metadata = node_data.get("metadata", {})
            if metadata.get("file_name") == filename:
                return True
        
        return False
    except Exception as e:
        logger.error(f"Error checking index for {filename}: {e}")
        return False


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Mark failed documents as COMPLETE if in index")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be updated without actually updating"
    )
    parser.add_argument(
        "--index-dir",
        default="/workspace/latest_model",
        help="Path to index directory (default: /workspace/latest_model)"
    )
    parser.add_argument(
        "--skip-index-check",
        action="store_true",
        help="Skip index check and mark all FAILED as COMPLETE (use with caution)"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Mark Failed Documents as COMPLETE")
    print("=" * 60)
    print()
    
    with SessionLocal() as session:
        # Find all documents with FAILED status
        failed_docs = (
            session.query(DocumentIngestionMetadata)
            .filter(DocumentIngestionMetadata.status == "FAILED")
            .order_by(DocumentIngestionMetadata.filename)
            .all()
        )
        
        if not failed_docs:
            print("✅ No failed documents found.")
            return 0
        
        print(f"Found {len(failed_docs)} failed document(s)")
        print()
        
        if not args.skip_index_check:
            print(f"Checking index at: {args.index_dir}")
            print("Verifying which documents exist in the index...")
            print()
        
        # Check each document
        to_update = []
        not_in_index = []
        
        for doc in failed_docs:
            if args.skip_index_check:
                to_update.append(doc)
            else:
                in_index = check_document_in_index(doc.filename, args.index_dir)
                if in_index:
                    to_update.append(doc)
                    print(f"  ✅ {doc.filename} - Found in index")
                else:
                    not_in_index.append(doc)
                    print(f"  ❌ {doc.filename} - NOT in index (will skip)")
        
        print()
        
        if not to_update:
            print("No documents to update (none found in index).")
            if not_in_index:
                print(f"\n{len(not_in_index)} document(s) not found in index:")
                for doc in not_in_index[:10]:  # Show first 10
                    print(f"  - {doc.filename}")
                if len(not_in_index) > 10:
                    print(f"  ... and {len(not_in_index) - 10} more")
            return 0
        
        print(f"Will update {len(to_update)} document(s) to COMPLETE:")
        for doc in to_update[:10]:  # Show first 10
            error_preview = doc.error_message[:50] + "..." if doc.error_message and len(doc.error_message) > 50 else (doc.error_message or "No error")
            print(f"  - {doc.filename}")
            print(f"    Current error: {error_preview}")
        
        if len(to_update) > 10:
            print(f"  ... and {len(to_update) - 10} more")
        
        if args.dry_run:
            print()
            print("[DRY RUN] Would update these documents to COMPLETE.")
            print("Run without --dry-run to actually update.")
            return 0
        
        print()
        response = input(f"Update {len(to_update)} document(s) to COMPLETE? (yes/no): ")
        
        if response.lower() not in ('yes', 'y'):
            print("Aborted.")
            return 0
        
        print()
        print("Updating documents...")
        
        updated_count = 0
        for doc in to_update:
            old_status = doc.status
            doc.status = "COMPLETE"
            doc.error_message = None  # Clear error message
            updated_count += 1
            print(f"  ✅ Updated {doc.filename}: {old_status} -> COMPLETE")
        
        # Commit changes
        session.commit()
        
        print()
        print("=" * 60)
        print(f"Successfully updated {updated_count} document(s) to COMPLETE.")
        print("=" * 60)
        
        if not_in_index:
            print()
            print(f"⚠️  Note: {len(not_in_index)} document(s) were NOT updated because")
            print("   they were not found in the index. These may need to be re-ingested.")
        
        return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nAborted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

