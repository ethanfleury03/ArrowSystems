"""
Simple document deletion utility.

Deletes document metadata and related data without triggering index rebuild.
This allows CRUD operations on documents even when ingestion is disabled.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.utils.test_mode import get_chunks_dir, get_original_pdfs_dir
from backend.utils.gcs_client import delete_object, parse_gcs_path
from backend.logging_config import get_logger

logger = get_logger(__name__)


def delete_document_metadata_simple(metadata_id: str) -> dict:
    """
    Delete a document by metadata_id without triggering index rebuild.
    
    This function:
    1. Deletes DocumentIngestionMetadata row
    2. Deletes Document row (if exists, matching by filename)
    3. Deletes chunks JSON file
    4. Deletes GCS file (best-effort, logs warning on failure)
    5. Deletes local file (if exists)
    
    Does NOT:
    - Trigger index rebuild
    - Delete from vector store (that's handled by external pipeline)
    
    Args:
        metadata_id: The ID of the DocumentIngestionMetadata record to delete
    
    Returns:
        dict with deletion results:
        {
            "metadata_id": str,
            "filename": str,
            "deleted_metadata": bool,
            "deleted_document": bool,
            "deleted_chunks_file": bool,
            "deleted_gcs": bool,
            "deleted_local": bool,
        }
    """
    result = {
        "metadata_id": metadata_id,
        "filename": None,
        "deleted_metadata": False,
        "deleted_document": False,
        "deleted_chunks_file": False,
        "deleted_gcs": False,
        "deleted_local": False,
    }
    
    session: Optional[Session] = None
    try:
        session = SessionLocal()
        
        # Load metadata record
        metadata = session.query(DocumentIngestionMetadata).filter(
            DocumentIngestionMetadata.id == metadata_id
        ).first()
        
        if not metadata:
            logger.warning(f"Document metadata not found: {metadata_id}")
            return result
        
        result["filename"] = metadata.filename
        filename = metadata.filename
        gcs_path = None
        
        # Get GCS path from Document table if available
        doc_record = session.query(Document).filter(
            Document.file_name == filename
        ).first()
        if doc_record and doc_record.gcs_path:
            gcs_path = doc_record.gcs_path
        
        # Delete Document row (if exists)
        if doc_record:
            session.delete(doc_record)
            result["deleted_document"] = True
            logger.info(f"Deleted Document row for filename: {filename}")
        
        # Delete DocumentIngestionMetadata row
        session.delete(metadata)
        session.commit()
        result["deleted_metadata"] = True
        logger.info(f"Deleted DocumentIngestionMetadata row: {metadata_id}")
        
        # Delete chunks JSON file
        chunks_dir = get_chunks_dir()
        chunks_file = Path(chunks_dir) / f"{metadata_id}.json"
        if chunks_file.exists():
            try:
                chunks_file.unlink()
                result["deleted_chunks_file"] = True
                logger.info(f"Deleted chunks file: {chunks_file}")
            except Exception as e:
                logger.warning(f"Failed to delete chunks file {chunks_file}: {e}")
        
        # Delete GCS file (best-effort)
        if gcs_path:
            try:
                if delete_object(gcs_path):
                    result["deleted_gcs"] = True
            except Exception as e:
                logger.warning(f"Failed to delete GCS file {gcs_path}: {e}")
        
        # Also try to delete from metadata.file_path if it exists
        if metadata.file_path and os.path.exists(metadata.file_path):
            try:
                os.remove(metadata.file_path)
                result["deleted_local"] = True
                logger.info(f"Deleted local file: {metadata.file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete local file {metadata.file_path}: {e}")
        
        # Also check original_pdfs directory
        original_pdfs_dir = get_original_pdfs_dir()
        original_file_path = os.path.join(original_pdfs_dir, filename)
        if os.path.exists(original_file_path):
            try:
                os.remove(original_file_path)
                result["deleted_local"] = True
                logger.info(f"Deleted original PDF: {original_file_path}")
            except Exception as e:
                logger.warning(f"Failed to delete original PDF {original_file_path}: {e}")
        
        return result
        
    except Exception as e:
        logger.error(f"Error in simple delete for metadata_id {metadata_id}: {e}", exc_info=True)
        if session:
            session.rollback()
        raise
    finally:
        if session:
            session.close()

