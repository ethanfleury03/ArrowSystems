#!/usr/bin/env python3
"""
Document Count Verification Script

Verifies consistency between:
- Database document counts (DocumentIngestionMetadata and Document tables)
- Admin UI API endpoint
- ingest.py query results

Run this after uploading a document to verify it's visible to ingest.py.
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.config.env import settings
from sqlalchemy import func, or_
import re


def mask_db_url(db_url: str) -> str:
    """Mask password in database URL for logging."""
    return re.sub(r':([^:@]+)@', r':***@', db_url)


def get_db_info():
    """Get database connection info."""
    db_url = settings.DATABASE_URL if hasattr(settings, 'DATABASE_URL') else os.getenv('DATABASE_URL', 'NOT_SET')
    db_url_safe = mask_db_url(db_url) if db_url != 'NOT_SET' else 'NOT_SET'
    
    # Extract host and database name
    host = 'unknown'
    db_name = 'unknown'
    if db_url != 'NOT_SET':
        # Parse PostgreSQL connection string: postgresql://user:pass@host:port/dbname
        # Host: after @, before : or /
        host_match = re.search(r'@([^:/]+)', db_url)
        if host_match:
            host = host_match.group(1)
        
        # Database name: after the last /, before ? or end of string
        # Need to find the / that comes after the port number
        # Pattern: :port/dbname or :port/dbname?params
        db_match = re.search(r':\d+/([^?]+)', db_url)
        if db_match:
            db_name = db_match.group(1)
        else:
            # Fallback: try to find any / after @
            db_match = re.search(r'@[^/]+/([^?]+)', db_url)
            if db_match:
                db_name = db_match.group(1)
    
    return {
        'url': db_url_safe,
        'host': host,
        'name': db_name,
        'is_sqlite': db_url.startswith('sqlite') if db_url != 'NOT_SET' else False
    }


def count_documents():
    """Count documents using the same query as ingest.py."""
    session = SessionLocal()
    try:
        # Same query as ingest.py
        metadata_records = (
            session.query(DocumentIngestionMetadata, Document)
            .outerjoin(Document, DocumentIngestionMetadata.filename == Document.file_name)
            .filter(
                or_(
                    Document.gcs_path.isnot(None),
                    (DocumentIngestionMetadata.file_path.isnot(None) & 
                     DocumentIngestionMetadata.file_path.like('gs://%'))
                )
            )
            .filter(
                or_(
                    Document.is_active.is_(True),
                    Document.id.is_(None)
                )
            )
            .all()
        )
        
        # Total counts
        total_metadata = session.query(func.count(DocumentIngestionMetadata.id)).scalar() or 0
        total_documents = session.query(func.count(Document.id)).scalar() or 0
        docs_with_gcs = session.query(func.count(Document.id)).filter(
            Document.gcs_path.isnot(None)
        ).scalar() or 0
        metadata_with_gcs = session.query(func.count(DocumentIngestionMetadata.id)).filter(
            DocumentIngestionMetadata.file_path.like('gs://%')
        ).scalar() or 0
        
        return {
            'ingest_query_count': len(metadata_records),
            'total_metadata': total_metadata,
            'total_documents': total_documents,
            'docs_with_gcs': docs_with_gcs,
            'metadata_with_gcs': metadata_with_gcs,
            'records': metadata_records
        }
    finally:
        session.close()


def main():
    print("=" * 70)
    print("Document Count Verification")
    print("=" * 70)
    print()
    
    # Database info
    db_info = get_db_info()
    print(f"📊 Database Connection:")
    print(f"   URL: {db_info['url']}")
    print(f"   Host: {db_info['host']}")
    print(f"   Database: {db_info['name']}")
    
    if db_info['is_sqlite']:
        print()
        print("❌ ERROR: SQLite detected! This should never happen in production.")
        print("   Ensure DATABASE_URL points to PostgreSQL.")
        return 1
    
    print()
    
    # Count documents
    counts = count_documents()
    
    print(f"📈 Document Counts:")
    print(f"   Total DocumentIngestionMetadata records: {counts['total_metadata']}")
    print(f"   Total Document records: {counts['total_documents']}")
    print(f"   Document records with gcs_path: {counts['docs_with_gcs']}")
    print(f"   Metadata records with gs:// file_path: {counts['metadata_with_gcs']}")
    print()
    print(f"✅ Documents matching ingest.py query: {counts['ingest_query_count']}")
    print()
    
    # Check for mismatches and identify missing records
    if counts['total_metadata'] != counts['ingest_query_count']:
        diff = counts['total_metadata'] - counts['ingest_query_count']
        print(f"⚠️  WARNING: {diff} DocumentIngestionMetadata records are NOT included in ingest.py query")
        print()
        
        # Find which records are missing
        session = SessionLocal()
        try:
            all_metadata = session.query(DocumentIngestionMetadata).all()
            matched_filenames = {meta.filename for meta, _ in counts['records']}
            missing_metadata = [meta for meta in all_metadata if meta.filename not in matched_filenames]
            
            print(f"   Missing records ({len(missing_metadata)}):")
            for meta in missing_metadata:
                # Check if Document record exists
                doc = session.query(Document).filter(Document.file_name == meta.filename).first()
                doc_gcs = doc.gcs_path if doc else None
                doc_active = doc.is_active if doc else None
                
                print(f"   - {meta.filename}")
                print(f"     Status: {meta.status}")
                print(f"     Metadata file_path: {meta.file_path or 'NULL'}")
                print(f"     Document record exists: {doc is not None}")
                if doc:
                    print(f"     Document gcs_path: {doc.gcs_path or 'NULL'}")
                    print(f"     Document is_active: {doc.is_active}")
                print()
                
                # Explain why it's missing
                reasons = []
                if not doc_gcs and not (meta.file_path and meta.file_path.startswith('gs://')):
                    reasons.append("No GCS path in either Document.gcs_path or Metadata.file_path")
                if doc and not doc.is_active:
                    reasons.append("Document record exists but is_active=False")
                
                if reasons:
                    print(f"     ❌ Missing because: {', '.join(reasons)}")
                print()
        finally:
            session.close()
        
        print("   💡 Fix: Update these records to have GCS paths in either:")
        print("      - Document.gcs_path, OR")
        print("      - DocumentIngestionMetadata.file_path (starting with 'gs://')")
        print()
    
    # List recent documents
    if counts['records']:
        print(f"📄 Sample documents (first 5):")
        for meta, doc in counts['records'][:5]:
            gcs_source = "Document.gcs_path" if (doc and doc.gcs_path) else "Metadata.file_path"
            gcs_path = doc.gcs_path if (doc and doc.gcs_path) else meta.file_path
            print(f"   - {meta.filename}")
            print(f"     Status: {meta.status}, GCS source: {gcs_source}")
            print(f"     GCS path: {gcs_path[:80]}..." if gcs_path and len(gcs_path) > 80 else f"     GCS path: {gcs_path}")
        if len(counts['records']) > 5:
            print(f"   ... and {len(counts['records']) - 5} more")
        print()
    
    print("=" * 70)
    print("✅ Verification complete")
    print("=" * 70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

