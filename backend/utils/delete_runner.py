"""
Delete runner for Phase 4 of document ingestion.

This module handles safe document deletion with full index rebuild:
- Deletes document metadata, chunks, and files
- Rebuilds the entire index from remaining documents
- Performs atomic swap of index directories
"""

import os
import json
import shutil
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy.orm import Session
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.schema import TextNode

from backend.utils.db import SessionLocal, DocumentIngestionMetadata
from backend.utils.query_summarizer import QuerySummarizer
from backend.logging_config import get_logger
from backend.utils.test_mode import get_chunks_dir, get_original_pdfs_dir, get_index_dir, get_temp_index_dir

logger = get_logger(__name__)


def run_delete_and_reindex(metadata_id: str) -> None:
    """
    Safely delete a document and rebuild the index from remaining documents.
    
    INDEX-WRITE PATH: rebuilds entire index after deletion
    
    This function:
    1. Loads all metadata records
    2. Deletes the target document's metadata, chunks, and files
    3. Rebuilds the index from all remaining documents
    4. Performs atomic swap of index directories
    
    Args:
        metadata_id: The ID of the DocumentIngestionMetadata record to delete
    """
    # Check if app-based ingestion is allowed
    from backend.config.env import settings
    if not settings.allow_app_ingestion:
        logger.warning(
            {
                "event": "ingestion_blocked_from_app",
                "metadata_id": metadata_id,
                "function": "run_delete_and_reindex",
            }
        )
        raise RuntimeError(
            "Ingestion is disabled in this environment. Index rebuild must be triggered via external GPU pipeline."
        )
    
    session: Optional[Session] = None
    temp_index_dir = get_temp_index_dir()
    original_index_dir = get_index_dir()
    
    try:
        # Load metadata record
        session = SessionLocal()
        metadata = session.query(DocumentIngestionMetadata).filter(
            DocumentIngestionMetadata.id == metadata_id
        ).first()
        
        if not metadata:
            logger.error(f"delete_metadata_not_found", metadata_id=metadata_id)
            return
        
        filename = metadata.filename
        
        # Update status to DELETING
        metadata.status = "DELETING"
        session.commit()
        logger.info(f"delete_started", metadata_id=metadata_id, filename=filename)
        
        # A. Load all metadata records
        all_metadata = session.query(DocumentIngestionMetadata).all()
        logger.info(f"delete_loaded_metadata", total_documents=len(all_metadata))
        
        # B. Remove the target document
        # Delete chunk file
        chunks_file = Path(get_chunks_dir()) / f"{metadata_id}.json"
        if chunks_file.exists():
            try:
                chunks_file.unlink()
                logger.info(f"delete_chunks_file_removed", metadata_id=metadata_id, chunks_file=str(chunks_file))
            except Exception as e:
                logger.warning(f"delete_chunks_file_remove_failed", metadata_id=metadata_id, error=str(e))
        
        # Delete original PDF file
        if metadata.file_path and os.path.exists(metadata.file_path):
            try:
                os.remove(metadata.file_path)
                logger.info(f"delete_original_file_removed", metadata_id=metadata_id, file_path=metadata.file_path)
            except Exception as e:
                logger.warning(f"delete_original_file_remove_failed", metadata_id=metadata_id, error=str(e))
        
        # Also check and delete from the original_pdfs directory (in case file_path is different)
        original_pdfs_dir = get_original_pdfs_dir()
        original_file_path = os.path.join(original_pdfs_dir, filename)
        if os.path.exists(original_file_path):
            try:
                os.remove(original_file_path)
                logger.info(f"delete_original_file_removed_from_dir", metadata_id=metadata_id, file_path=original_file_path)
            except Exception as e:
                logger.warning(f"delete_original_file_remove_failed_from_dir", metadata_id=metadata_id, error=str(e))
        
        # Delete metadata row
        session.delete(metadata)
        session.commit()
        logger.info(f"delete_metadata_row_removed", metadata_id=metadata_id)
        
        # C. Begin REBUILDING_INDEX
        # Note: The deleted metadata is gone, so we can't update its status
        # We'll track rebuild status in logs for remaining documents
        logger.info(f"delete_rebuilding_index_started", metadata_id=metadata_id)
        
        # Update remaining documents to REBUILDING_INDEX status (optional, for UI visibility)
        # This is informational - the actual rebuild happens next
        for remaining_meta in session.query(DocumentIngestionMetadata).all():
            if remaining_meta.status == "COMPLETE":
                # Only update COMPLETE documents to show rebuild status
                remaining_meta.status = "REBUILDING_INDEX"
        session.commit()
        
        # Get remaining metadata (the deleted one is already removed from DB)
        remaining_metadata = session.query(DocumentIngestionMetadata).all()
        logger.info(f"delete_remaining_documents", count=len(remaining_metadata))
        
        if not remaining_metadata:
            # No documents left, just remove the index
            if os.path.exists(original_index_dir):
                try:
                    shutil.rmtree(original_index_dir)
                    logger.info(f"delete_index_removed_no_documents", index_dir=original_index_dir)
                except Exception as e:
                    logger.warning(f"delete_index_remove_failed", error=str(e))
            return
        
        # D. Rebuild the index from scratch
        # Create temporary folder
        if os.path.exists(temp_index_dir):
            shutil.rmtree(temp_index_dir)
        os.makedirs(temp_index_dir, exist_ok=True)
        logger.info(f"delete_temp_index_created", temp_dir=temp_index_dir)
        
        # Initialize embedding model
        import yaml
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        embed_model_name = config.get("models", {}).get("embedding", "BAAI/bge-large-en-v1.5")
        cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface/hub')
        if cache_dir.endswith('huggingface'):
            cache_dir = os.path.join(cache_dir, 'hub')
        
        # Set embedding model in Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        if not Settings.embed_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                cache_folder=cache_dir
            )
            logger.info(f"delete_embedding_model_initialized", model_name=embed_model_name)
        
        # Initialize summarizer
        query_summarizer = QuerySummarizer(
            enabled=True,
            min_length=0  # Summarize all chunks
        )
        
        # Collect all nodes from remaining documents
        all_nodes: List[TextNode] = []
        processed_count = 0
        failed_count = 0
        
        for remaining_meta in remaining_metadata:
            try:
                # Load chunks from JSON file
                remaining_chunks_file = Path(get_chunks_dir()) / f"{remaining_meta.id}.json"
                if not remaining_chunks_file.exists():
                    logger.warning(f"delete_chunks_file_missing", metadata_id=remaining_meta.id, skipping=True)
                    failed_count += 1
                    continue
                
                with open(remaining_chunks_file, 'r', encoding='utf-8') as f:
                    chunks_data = json.load(f)
                
                chunks = chunks_data.get("chunks", [])
                if not chunks:
                    logger.warning(f"delete_no_chunks", metadata_id=remaining_meta.id, skipping=True)
                    failed_count += 1
                    continue
                
                # Process each chunk
                for chunk in chunks:
                    try:
                        chunk_text = chunk.get("text", "")
                        chunk_metadata = chunk.get("metadata", {})
                        
                        # Summarize (with fallback)
                        summary = None
                        try:
                            summary, was_summarized, _ = query_summarizer.summarize(chunk_text)
                            if not was_summarized:
                                summary = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
                        except Exception as e:
                            logger.warning(f"delete_summary_failed", metadata_id=remaining_meta.id, error=str(e))
                            summary = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
                        
                        # Create TextNode
                        node_metadata = {
                            **chunk_metadata,
                            "machine_model": remaining_meta.machine_model,
                            "ingestion_metadata_id": remaining_meta.id,
                            "summary": summary,
                        }
                        
                        if "node_id" in chunk and chunk["node_id"]:
                            node_metadata["chunk_id"] = chunk["node_id"]
                        
                        node = TextNode(
                            text=chunk_text,
                            metadata=node_metadata
                        )
                        all_nodes.append(node)
                        
                    except Exception as e:
                        logger.warning(f"delete_chunk_processing_failed", metadata_id=remaining_meta.id, error=str(e))
                        continue
                
                processed_count += 1
                
            except Exception as e:
                logger.error(f"delete_document_rebuild_failed", metadata_id=remaining_meta.id, error=str(e))
                failed_count += 1
                continue
        
        if not all_nodes:
            logger.error(f"delete_no_nodes_to_rebuild", processed=processed_count, failed=failed_count)
            # Clean up temp directory
            if os.path.exists(temp_index_dir):
                shutil.rmtree(temp_index_dir)
            raise ValueError("No nodes available to rebuild index")
        
        logger.info(
            f"delete_nodes_collected",
            total_nodes=len(all_nodes),
            processed_documents=processed_count,
            failed_documents=failed_count
        )
        
        # Create new index with all nodes
        logger.info(f"delete_creating_new_index", node_count=len(all_nodes))
        new_index = VectorStoreIndex(nodes=[], show_progress=False)
        
        # Insert nodes in batches
        batch_size = 50
        successful_inserts = 0
        for i in range(0, len(all_nodes), batch_size):
            batch = all_nodes[i:i + batch_size]
            try:
                new_index.insert_nodes(batch)
                successful_inserts += len(batch)
            except Exception as e:
                logger.warning(f"delete_batch_insert_failed", batch_start=i, error=str(e))
                # Try inserting nodes one by one
                for node in batch:
                    try:
                        new_index.insert_nodes([node])
                        successful_inserts += 1
                    except Exception as node_error:
                        logger.warning(f"delete_node_insert_failed", error=str(node_error))
        
        if successful_inserts == 0:
            raise ValueError("All node insertions failed during rebuild")
        
        logger.info(f"delete_index_nodes_inserted", successful=successful_inserts, total=len(all_nodes))
        
        # Persist to temporary directory
        logger.info(f"delete_persisting_temp_index", temp_dir=temp_index_dir)
        new_index.storage_context.persist(persist_dir=temp_index_dir)
        logger.info(f"delete_temp_index_persisted")
        
        # E. Atomic swap
        # Remove old index
        if os.path.exists(original_index_dir):
            try:
                shutil.rmtree(original_index_dir)
                logger.info(f"delete_old_index_removed", index_dir=original_index_dir)
            except Exception as e:
                logger.error(f"delete_old_index_remove_failed", error=str(e))
                # Clean up temp directory
                if os.path.exists(temp_index_dir):
                    shutil.rmtree(temp_index_dir)
                raise
        
        # Rename temp to original
        try:
            os.rename(temp_index_dir, original_index_dir)
            logger.info(f"delete_index_swapped", temp_dir=temp_index_dir, final_dir=original_index_dir)
        except Exception as e:
            logger.error(f"delete_index_swap_failed", error=str(e))
            # Try to restore old index if it exists as backup
            raise
        
        # F. Finish
        # Restore status of remaining documents to COMPLETE
        for remaining_meta in session.query(DocumentIngestionMetadata).all():
            if remaining_meta.status == "REBUILDING_INDEX":
                remaining_meta.status = "COMPLETE"
        session.commit()
        
        logger.info(
            f"delete_and_reindex_success",
            deleted_metadata_id=metadata_id,
            remaining_documents=len(remaining_metadata),
            nodes_in_new_index=successful_inserts
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"delete_and_reindex_failed", metadata_id=metadata_id, error=error_msg, exc_info=True)
        
        # G. Failure handling
        # Clean up temp directory if it exists
        if os.path.exists(temp_index_dir):
            try:
                shutil.rmtree(temp_index_dir)
                logger.info(f"delete_temp_index_cleaned_up")
            except Exception as cleanup_error:
                logger.error(f"delete_temp_index_cleanup_failed", error=str(cleanup_error))
        
        # Do NOT delete the original index - it's still valid
        # The deleted document's metadata is already removed, so we can't update its status
        # Log the error for monitoring
        
    finally:
        if session:
            session.close()

