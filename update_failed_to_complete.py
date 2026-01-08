#!/usr/bin/env python3
"""
Quick script to mark all FAILED documents as COMPLETE.
Run this after successful full ingestion.
"""

import sys
import os
import subprocess
from pathlib import Path

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

def auto_fix_db_connection():
    """Automatically set up database connection if DATABASE_URL is missing."""
    if os.getenv("DATABASE_URL"):
        return False  # Already set
    
    print("⚠️  DATABASE_URL not set. Attempting auto-fix...")
    print()
    
    # Check if we're on RunPod/Linux
    if os.path.exists("/workspace"):
        print("Detected RunPod environment.")
        print("Quick fix options:")
        print()
        print("Option 1 (with proxy - recommended):")
        print("  bash fix_db_quick.sh YOUR_PASSWORD")
        print()
        print("Option 2 (direct connection - faster if IP is whitelisted):")
        print("  bash fix_db_direct.sh YOUR_PASSWORD CLOUD_SQL_IP")
        print()
        print("Option 3 (manual):")
        print("  export DATABASE_URL='postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app'")
        print("  # Then start proxy: ./cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres &")
        print()
    else:
        print("Set DATABASE_URL environment variable:")
        print("  export DATABASE_URL='postgresql://rag_user:YOUR_PASSWORD@HOST:5432/rag_app'")
        print()
    
    return True

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
        error_str = str(e).lower()
        
        # Check for missing DATABASE_URL
        if "database_url" in error_str or "required" in error_str:
            if auto_fix_db_connection():
                return 1
        
        # Check for connection refused
        if "connection refused" in error_str or "127.0.0.1" in error_str:
            print(f"\n❌ Database connection failed: {e}")
            print()
            print("🔧 QUICK FIX:")
            print()
            print("On RunPod, run ONE of these:")
            print()
            print("  bash fix_db_quick.sh YOUR_PASSWORD")
            print("  # OR")
            print("  bash fix_db_direct.sh YOUR_PASSWORD CLOUD_SQL_IP")
            print()
            print("Then run this script again.")
            print()
            return 1
        
        print(f"\n❌ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())

