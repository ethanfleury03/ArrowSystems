"""
Simple document deletion utility.

Deletes document metadata and related data without triggering full index rebuild.
Supports incremental deletion: removes chunks from vector index by ingestion_metadata_id.
This allows CRUD operations on documents with single-document ingestion enabled.
"""

import os
import logging
from pathlib import Path
from typing import Optional
from sqlalchemy.orm import Session

from backend.utils.db import SessionLocal, DocumentIngestionMetadata, Document
from backend.utils.test_mode import get_chunks_dir, get_original_pdfs_dir, get_index_dir
from backend.utils.gcs_client import delete_object, parse_gcs_path
from backend.logging_config import get_logger

logger = get_logger(__name__)


def delete_document_metadata_simple(metadata_id: str) -> dict:
    """
    Delete a document by metadata_id with incremental index deletion.
    
    This function:
    1. Deletes chunks from vector index by ingestion_metadata_id (if RAG pipeline available)
    2. Deletes DocumentIngestionMetadata row
    3. Deletes Document row (if exists, matching by filename)
    4. Deletes chunks JSON file
    5. Deletes GCS file (best-effort, logs warning on failure)
    6. Deletes local file (if exists)
    
    Does NOT:
    - Trigger full index rebuild (uses incremental deletion)
    - Delete from vector store if RAG pipeline is not initialized (that's handled by external pipeline)
    
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
            "deleted_index_nodes": int,
            "deleted_index_ref_docs": int,
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
        "deleted_index_nodes": 0,
        "deleted_index_ref_docs": 0,
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
        
        # Delete chunks from vector index (best-effort, never blocks deletion)
        # This happens BEFORE deleting database records so we can still access metadata for logging
        # If index cleanup fails, we continue with DB deletion anyway
        try:
            # Import here to avoid circular dependencies
            from backend.rag_pipeline import rag_pipeline
            
            if rag_pipeline and rag_pipeline.is_initialized():
                index = rag_pipeline.orchestrator.index if (
                    rag_pipeline.orchestrator and 
                    hasattr(rag_pipeline.orchestrator, 'index') and
                    rag_pipeline.orchestrator.index
                ) else None
                
                if index:
                    nodes_to_delete = []
                    ref_doc_ids_to_delete = set()
                    
                    # Method 1: Find nodes via retriever corpus_nodes by ingestion_metadata_id
                    if (hasattr(rag_pipeline.orchestrator, 'retriever') and 
                        rag_pipeline.orchestrator.retriever):
                        retriever = rag_pipeline.orchestrator.retriever
                        if hasattr(retriever, 'corpus_nodes') and retriever.corpus_nodes:
                            for node_wrapper in retriever.corpus_nodes:
                                try:
                                    if node_wrapper is None:
                                        continue
                                    node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
                                    if node is None:
                                        continue
                                    if not hasattr(node, 'metadata') or not node.metadata:
                                        continue
                                    # Match by ingestion_metadata_id (preferred) or metadata_id
                                    node_metadata_id = (
                                        node.metadata.get('ingestion_metadata_id') or 
                                        node.metadata.get('metadata_id')
                                    )
                                    if node_metadata_id == metadata_id:
                                        nodes_to_delete.append(node)
                                        # Track ref_doc_id if available
                                        if hasattr(node, 'ref_doc_id') and node.ref_doc_id:
                                            ref_doc_ids_to_delete.add(node.ref_doc_id)
                                except Exception as e:
                                    logger.debug(f"Error processing node_wrapper in corpus_nodes: {e}")
                                    continue
                    
                    # Method 2: Find nodes via docstore by ingestion_metadata_id
                    if hasattr(index, 'docstore') and index.docstore:
                        try:
                            doc_keys = list(index.docstore.docs.keys()) if hasattr(index.docstore, 'docs') and index.docstore.docs else []
                            for doc_id in doc_keys:
                                try:
                                    if doc_id is None:
                                        continue
                                    doc = index.docstore.get_document(doc_id)
                                    if doc is None:
                                        continue
                                    if not hasattr(doc, 'metadata') or not doc.metadata:
                                        continue
                                    doc_metadata_id = (
                                        doc.metadata.get('ingestion_metadata_id') or 
                                        doc.metadata.get('metadata_id')
                                    )
                                    if doc_metadata_id == metadata_id:
                                        ref_doc_ids_to_delete.add(doc_id)
                                except KeyError:
                                    # Doc ID doesn't exist in docstore - skip
                                    logger.debug(f"Doc ID {doc_id} not found in docstore")
                                    continue
                                except Exception as e:
                                    logger.debug(f"Error checking docstore doc {doc_id}: {e}")
                                    continue
                        except Exception as e:
                            logger.warning(f"Error accessing docstore: {e}")
                            # Continue with deletion even if docstore access fails
                    
                    # Delete nodes from index
                    for node in nodes_to_delete:
                        try:
                            if node is None:
                                continue
                            if not hasattr(node, 'node_id') or node.node_id is None:
                                logger.debug("Skipping node without node_id")
                                continue
                            index.delete(node.node_id)
                            result["deleted_index_nodes"] += 1
                            logger.debug(f"Deleted node {node.node_id} from index")
                        except Exception as e:
                            node_id = getattr(node, 'node_id', 'unknown') if node else 'None'
                            logger.warning(f"Failed to delete node {node_id}: {e}")
                    
                    # Delete reference documents (this removes associated nodes)
                    for ref_doc_id in ref_doc_ids_to_delete:
                        try:
                            if ref_doc_id is None:
                                continue
                            index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
                            result["deleted_index_ref_docs"] += 1
                            logger.debug(f"Deleted ref_doc {ref_doc_id} from index")
                        except KeyError:
                            # Ref doc doesn't exist - treat as success
                            logger.debug(f"Ref doc {ref_doc_id} not found in index (may already be deleted)")
                        except Exception as e:
                            logger.warning(f"Failed to delete ref_doc {ref_doc_id}: {e}")
                    
                    # Persist the index if we deleted anything
                    if result["deleted_index_nodes"] > 0 or result["deleted_index_ref_docs"] > 0:
                        # Find storage path
                        storage_path = None
                        possible_paths = [
                            get_index_dir(),
                            "latest_model",
                            "../latest_model",
                            "/workspace/latest_model",
                            "/workspace/ArrowSystems/latest_model",
                            "/workspace/storage",
                            "./storage"
                        ]
                        
                        for path in possible_paths:
                            if path and os.path.exists(path):
                                storage_path = path
                                break
                        
                        if storage_path:
                            try:
                                logger.info(f"Persisting index after deleting {result['deleted_index_nodes']} nodes and {result['deleted_index_ref_docs']} ref_docs...")
                                index.storage_context.persist(persist_dir=storage_path)
                                logger.info("✅ Index persisted with deletions")
                                
                                # Reload RAG pipeline to refresh in-memory state
                                logger.info("Reloading RAG pipeline after chunk deletion...")
                                rag_pipeline.orchestrator.load_index(storage_dir=storage_path)
                                logger.info("✅ RAG pipeline reloaded")
                            except Exception as e:
                                logger.warning(f"Failed to persist/reload index: {e}")
                        
                        logger.info(
                            f"Deleted {result['deleted_index_nodes']} nodes and {result['deleted_index_ref_docs']} ref_docs "
                            f"from index for metadata_id {metadata_id}"
                        )
        except Exception as e:
            # Never fail the entire deletion if index deletion fails - this is best-effort only
            logger.warning(
                f"Failed to delete chunks from vector index (index may not be available, continuing with DB deletion): {e}",
                exc_info=True
            )
        
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
        
        # Delete GCS file (best-effort, never blocks deletion)
        # Handle 404 gracefully - object may already be deleted or never existed (orphaned record)
        if gcs_path:
            try:
                if delete_object(gcs_path):
                    result["deleted_gcs"] = True
                    logger.info(f"Deleted GCS file: {gcs_path}")
                else:
                    # delete_object returns False on error, but 404 is handled as success
                    logger.debug(f"GCS file deletion returned False (may not exist): {gcs_path}")
            except Exception as e:
                # Never fail deletion due to GCS errors - log and continue
                error_str = str(e)
                if "404" in error_str or "NotFound" in str(type(e).__name__):
                    logger.debug(f"GCS object not found (already deleted or orphaned): {gcs_path}")
                    result["deleted_gcs"] = True  # Consider 404 as success
                else:
                    logger.warning(f"Failed to delete GCS file {gcs_path} (non-blocking): {e}")
        
        # Also try to delete from metadata.file_path if it's a GCS path
        if metadata.file_path and metadata.file_path.startswith('gs://'):
            if metadata.file_path != gcs_path:  # Avoid duplicate deletion
                try:
                    if delete_object(metadata.file_path):
                        result["deleted_gcs"] = True
                        logger.info(f"Deleted GCS file from metadata.file_path: {metadata.file_path}")
                except Exception as e:
                    error_str = str(e)
                    if "404" in error_str or "NotFound" in str(type(e).__name__):
                        logger.debug(f"GCS object from metadata.file_path not found: {metadata.file_path}")
                    else:
                        logger.warning(f"Failed to delete GCS file from metadata.file_path (non-blocking): {e}")
        
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


def _find_nodes_and_refdocs_by_document_id(index, retriever, document_id: int):
    """
    Helper function to find all nodes and ref_doc_ids for a given document_id.
    
    Args:
        index: LlamaIndex VectorStoreIndex instance
        retriever: Retriever instance with corpus_nodes
        document_id: Integer Document.id to search for
    
    Returns:
        tuple: (nodes_to_delete: list, ref_doc_ids_to_delete: set)
    """
    nodes_to_delete = []
    ref_doc_ids_to_delete = set()
    
    # Method 1: Find nodes via retriever corpus_nodes by document_id
    if retriever and hasattr(retriever, 'corpus_nodes') and retriever.corpus_nodes:
        for node_wrapper in retriever.corpus_nodes:
            node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
            if hasattr(node, 'metadata') and node.metadata:
                node_document_id = node.metadata.get('document_id')
                # Match by document_id (integer comparison)
                if node_document_id is not None and int(node_document_id) == int(document_id):
                    nodes_to_delete.append(node)
                    # Track ref_doc_id if available
                    if hasattr(node, 'ref_doc_id') and node.ref_doc_id:
                        ref_doc_ids_to_delete.add(node.ref_doc_id)
    
    # Method 2: Find nodes via docstore by document_id
    if hasattr(index, 'docstore') and index.docstore:
        for doc_id in list(index.docstore.docs.keys()):
            try:
                doc = index.docstore.get_document(doc_id)
                if hasattr(doc, 'metadata') and doc.metadata:
                    doc_document_id = doc.metadata.get('document_id')
                    if doc_document_id is not None and int(doc_document_id) == int(document_id):
                        ref_doc_ids_to_delete.add(doc_id)
            except Exception as e:
                logger.debug(f"Error checking docstore doc {doc_id}: {e}")
                continue
    
    return nodes_to_delete, ref_doc_ids_to_delete


def delete_document_chunks_by_document_id(document_id: int, force: bool = False) -> dict:
    """
    Delete all chunks/nodes for a document by document_id.
    
    This function finds all chunks in the vector index that belong to the given document_id
    and deletes them. It does NOT delete database records - only removes chunks from the index.
    
    This is useful when:
    - A document has multiple DocumentIngestionMetadata records (re-uploads)
    - You want to delete all chunks for a document in one operation
    - You need to clean up chunks before re-ingesting a document
    
    Args:
        document_id: Integer Document.id to delete chunks for
        force: If True, continue even if index is unavailable (returns partial result with error)
    
    Returns:
        dict with deletion results:
        {
            "document_id": int,
            "deleted_nodes": int,
            "deleted_ref_docs": int,
            "legacy_nodes_missing_document_id": int,  # Count of nodes that might exist but lack document_id
            "error": str (optional, only if force=True and error occurred)
        }
    """
    result = {
        "document_id": document_id,
        "deleted_nodes": 0,
        "deleted_ref_docs": 0,
        "legacy_nodes_missing_document_id": 0,
    }
    
    # Validate document_id exists in database
    session: Optional[Session] = None
    try:
        session = SessionLocal()
        db_doc = session.query(Document).filter(Document.id == document_id).first()
        
        if not db_doc:
            error_msg = f"Document with id {document_id} not found in database"
            if force:
                result["error"] = error_msg
                return result
            raise ValueError(error_msg)
        
        logger.info(f"Deleting chunks for document_id {document_id} (filename: {db_doc.file_name})")
        
    except Exception as e:
        if force:
            result["error"] = f"Database error: {str(e)}"
            return result
        raise
    finally:
        if session:
            session.close()
    
    # Delete chunks from vector index (best-effort, never blocks)
    try:
        # Import here to avoid circular dependencies
        from backend.rag_pipeline import rag_pipeline
        
        if not rag_pipeline or not rag_pipeline.is_initialized():
            error_msg = "RAG pipeline not initialized - cannot delete chunks from index"
            if force:
                result["error"] = error_msg
                logger.warning(error_msg)
                return result
            raise RuntimeError(error_msg)
        
        index = rag_pipeline.orchestrator.index if (
            rag_pipeline.orchestrator and 
            hasattr(rag_pipeline.orchestrator, 'index') and
            rag_pipeline.orchestrator.index
        ) else None
        
        if not index:
            error_msg = "Index not available - cannot delete chunks"
            if force:
                result["error"] = error_msg
                logger.warning(error_msg)
                return result
            raise RuntimeError(error_msg)
        
        retriever = None
        if (hasattr(rag_pipeline.orchestrator, 'retriever') and 
            rag_pipeline.orchestrator.retriever):
            retriever = rag_pipeline.orchestrator.retriever
        
        # Find nodes and ref_docs by document_id
        nodes_to_delete, ref_doc_ids_to_delete = _find_nodes_and_refdocs_by_document_id(
            index, retriever, document_id
        )
        
        # Count legacy nodes (nodes that might belong to this document but lack document_id)
        # This is informational only - we can't safely delete them without document_id
        if retriever and hasattr(retriever, 'corpus_nodes') and retriever.corpus_nodes:
            # Find all DocumentIngestionMetadata records for this document
            session = SessionLocal()
            try:
                metadata_records = session.query(DocumentIngestionMetadata).filter(
                    DocumentIngestionMetadata.filename == db_doc.file_name
                ).all()
                metadata_ids = {meta.id for meta in metadata_records}
                
                # Count nodes that have matching metadata_id but no document_id
                for node_wrapper in retriever.corpus_nodes:
                    node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
                    if hasattr(node, 'metadata') and node.metadata:
                        node_metadata_id = (
                            node.metadata.get('ingestion_metadata_id') or 
                            node.metadata.get('metadata_id')
                        )
                        node_document_id = node.metadata.get('document_id')
                        # If it matches metadata_id but lacks document_id, it's a legacy node
                        if node_metadata_id in metadata_ids and node_document_id is None:
                            result["legacy_nodes_missing_document_id"] += 1
            finally:
                session.close()
        
        # Delete nodes from index
        for node in nodes_to_delete:
            try:
                if hasattr(node, 'node_id'):
                    index.delete(node.node_id)
                    result["deleted_nodes"] += 1
                    logger.debug(f"Deleted node {node.node_id} from index (document_id={document_id})")
            except Exception as e:
                logger.warning(f"Failed to delete node {getattr(node, 'node_id', 'unknown')}: {e}")
        
        # Delete reference documents (this removes associated nodes)
        for ref_doc_id in ref_doc_ids_to_delete:
            try:
                index.delete_ref_doc(ref_doc_id, delete_from_docstore=True)
                result["deleted_ref_docs"] += 1
                logger.debug(f"Deleted ref_doc {ref_doc_id} from index (document_id={document_id})")
            except Exception as e:
                logger.warning(f"Failed to delete ref_doc {ref_doc_id}: {e}")
        
        # Persist the index if we deleted anything
        if result["deleted_nodes"] > 0 or result["deleted_ref_docs"] > 0:
            # Find storage path
            storage_path = None
            possible_paths = [
                get_index_dir(),
                "latest_model",
                "../latest_model",
                "/workspace/latest_model",
                "/workspace/ArrowSystems/latest_model",
                "/workspace/storage",
                "./storage"
            ]
            
            for path in possible_paths:
                if path and os.path.exists(path):
                    storage_path = path
                    break
            
            if storage_path:
                try:
                    logger.info(
                        f"Persisting index after deleting {result['deleted_nodes']} nodes and "
                        f"{result['deleted_ref_docs']} ref_docs for document_id {document_id}..."
                    )
                    index.storage_context.persist(persist_dir=storage_path)
                    logger.info("✅ Index persisted with deletions")
                    
                    # Reload RAG pipeline to refresh in-memory state
                    logger.info("Reloading RAG pipeline after chunk deletion...")
                    rag_pipeline.orchestrator.load_index(storage_dir=storage_path)
                    logger.info("✅ RAG pipeline reloaded")
                except Exception as e:
                    logger.warning(f"Failed to persist/reload index: {e}")
            
            logger.info(
                f"Deleted {result['deleted_nodes']} nodes and {result['deleted_ref_docs']} ref_docs "
                f"from index for document_id {document_id}"
            )
        else:
            logger.info(f"No chunks found in index for document_id {document_id}")
            
    except Exception as e:
        # Never fail if force=True - return partial result with error
        error_msg = f"Failed to delete chunks from vector index: {e}"
        if force:
            result["error"] = error_msg
            logger.warning(error_msg, exc_info=True)
        else:
            logger.error(error_msg, exc_info=True)
            raise
    
    return result

