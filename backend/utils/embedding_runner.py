"""
Embedding runner for Phase 3 of document ingestion.

This module handles the embedding phase of document ingestion:
- Loads chunks from JSON files
- Summarizes chunks (with fallback)
- Converts to LlamaIndex nodes
- Inserts into vector index
- Persists index
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from llama_index.core import VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.schema import TextNode

from backend.utils.db import SessionLocal, DocumentIngestionMetadata
from backend.utils.query_summarizer import QuerySummarizer
from backend.logging_config import get_logger
from backend.utils.test_mode import get_chunks_dir, get_index_dir

logger = get_logger(__name__)


def run_embedding(metadata_id: str, request_id: Optional[str] = None) -> None:
    """
    Run embedding for a document ingestion metadata record.
    
    This function:
    1. Sets status to EMBEDDING
    2. Loads chunks from JSON file
    3. Summarizes each chunk (with fallback)
    4. Converts to LlamaIndex TextNode objects
    5. Loads or creates vector index
    6. Inserts nodes into index (embeddings generated automatically)
    7. Persists index
    8. Sets status to COMPLETE on success
    9. Sets status to FAILED on error
    
    Args:
        metadata_id: The ID of the DocumentIngestionMetadata record
        request_id: Optional request ID for tracing
    """
    session: Optional[Session] = None
    document = None
    try:
        # Load metadata record
        session = SessionLocal()
        metadata = session.query(DocumentIngestionMetadata).filter(
            DocumentIngestionMetadata.id == metadata_id
        ).first()
        
        if not metadata:
            logger.error(
                {
                    "event": "embedding_metadata_not_found",
                    "metadata_id": metadata_id,
                    "request_id": request_id,
                }
            )
            return
        
        document = metadata  # Store for error handling
        
        # Update status to EMBEDDING
        metadata.status = "EMBEDDING"
        session.commit()
        
        # Load chunks from JSON file
        chunks_file = Path(get_chunks_dir()) / f"{metadata_id}.json"
        if not chunks_file.exists():
            raise FileNotFoundError(f"Chunks file not found: {chunks_file}")
        
        with open(chunks_file, 'r', encoding='utf-8') as f:
            chunks_data = json.load(f)
        
        chunks = chunks_data.get("chunks", [])
        if not chunks:
            raise ValueError("No chunks found in chunks file")
        
        # Update the embedding_started log with chunk count
        logger.info(
            {
                "event": "document_embedding_started",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "num_chunks": len(chunks),
                "request_id": request_id,
            }
        )
        
        # Initialize summarizer (same as used in ingestion pipeline)
        query_summarizer = QuerySummarizer(
            enabled=True,
            min_length=0  # Summarize all chunks
        )
        
        # Summarize chunks (with fallback)
        nodes: List[TextNode] = []
        summarized_count = 0
        fallback_count = 0
        
        for chunk_idx, chunk in enumerate(chunks):
            try:
                chunk_text = chunk.get("text", "")
                chunk_metadata = chunk.get("metadata", {})
                
                # Try to summarize
                summary = None
                try:
                    summary, was_summarized, _ = query_summarizer.summarize(chunk_text)
                    if was_summarized:
                        summarized_count += 1
                    else:
                        # If summarizer didn't summarize (e.g., too short), use fallback
                        summary = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
                        fallback_count += 1
                except Exception as e:
                    # Fallback on any error
                    logger.warning(f"embedding_summary_failed", metadata_id=metadata_id, chunk_idx=chunk_idx, error=str(e))
                    summary = chunk_text[:200] if len(chunk_text) > 200 else chunk_text
                    fallback_count += 1
                
                # Create TextNode with metadata
                node_metadata = {
                    **chunk_metadata,
                    "machine_model": metadata.machine_model,
                    "ingestion_metadata_id": metadata_id,
                    "summary": summary,
                }
                
                # Preserve chunk_id if available
                if "node_id" in chunk and chunk["node_id"]:
                    node_metadata["chunk_id"] = chunk["node_id"]
                
                node = TextNode(
                    text=chunk_text,
                    metadata=node_metadata
                )
                nodes.append(node)
                
            except Exception as e:
                logger.warning(f"embedding_chunk_processing_failed", metadata_id=metadata_id, chunk_idx=chunk_idx, error=str(e))
                # Continue processing other chunks
                continue
        
        if not nodes:
            raise ValueError("No valid nodes created from chunks")
        
        logger.info(
            f"embedding_summarization_complete",
            metadata_id=metadata_id,
            total_chunks=len(chunks),
            summarized=summarized_count,
            fallback=fallback_count,
            valid_nodes=len(nodes)
        )
        
        # Load or create vector index
        storage_dir = get_index_dir()
        
        # Ensure embedding model is set in Settings
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
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
        if not Settings.embed_model:
            Settings.embed_model = HuggingFaceEmbedding(
                model_name=embed_model_name,
                cache_folder=cache_dir
            )
            logger.info(f"embedding_model_initialized", model_name=embed_model_name)
        
        # Load existing index or create new one
        index = None
        if os.path.exists(storage_dir):
            try:
                storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
                index = load_index_from_storage(storage_context)
                logger.info(f"embedding_index_loaded", metadata_id=metadata_id, storage_dir=storage_dir)
            except Exception as e:
                logger.warning(f"embedding_index_load_failed", metadata_id=metadata_id, error=str(e), creating_new=True)
                # Create new index if loading fails
                index = VectorStoreIndex(nodes=[], show_progress=False)
                logger.info(f"embedding_index_created_new", metadata_id=metadata_id)
        else:
            # Create new index if directory doesn't exist
            os.makedirs(storage_dir, exist_ok=True)
            index = VectorStoreIndex(nodes=[], show_progress=False)
            logger.info(f"embedding_index_created_new", metadata_id=metadata_id, storage_dir=storage_dir)
        
        # Insert nodes into index (embeddings generated automatically)
        batch_size = 50
        successful_inserts = 0
        failed_inserts = 0
        
        for i in range(0, len(nodes), batch_size):
            batch = nodes[i:i + batch_size]
            batch_index = i // batch_size
            try:
                index.insert_nodes(batch)
                successful_inserts += len(batch)
                # Log batch progress at DEBUG level
                logger.debug(
                    {
                        "event": "document_embedding_batch",
                        "document_id": metadata_id,
                        "batch_index": batch_index,
                        "batch_size": len(batch),
                        "request_id": request_id,
                    }
                )
            except Exception as e:
                logger.warning(
                    {
                        "event": "embedding_batch_insert_failed",
                        "metadata_id": metadata_id,
                        "batch_start": i,
                        "error": str(e),
                        "request_id": request_id,
                    }
                )
                # Try inserting nodes one by one
                for node in batch:
                    try:
                        index.insert_nodes([node])
                        successful_inserts += 1
                    except Exception as node_error:
                        logger.warning(
                            {
                                "event": "embedding_node_insert_failed",
                                "metadata_id": metadata_id,
                                "node_idx": i,
                                "error": str(node_error),
                                "request_id": request_id,
                            }
                        )
                        failed_inserts += 1
        
        if successful_inserts == 0:
            raise ValueError("All node insertions failed")
        
        # Log embedding completed
        logger.info(
            {
                "event": "document_embedding_completed",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "num_chunks": len(nodes),
                "successful_inserts": successful_inserts,
                "failed_inserts": failed_inserts,
                "request_id": request_id,
            }
        )
        
        # Persist the index
        logger.info(
            {
                "event": "embedding_persisting_index",
                "metadata_id": metadata_id,
                "storage_dir": storage_dir,
                "request_id": request_id,
            }
        )
        index.storage_context.persist(persist_dir=storage_dir)
        logger.info(
            {
                "event": "embedding_index_persisted",
                "metadata_id": metadata_id,
                "storage_dir": storage_dir,
                "request_id": request_id,
            }
        )
        
        # Update status to COMPLETE
        metadata.status = "COMPLETE"
        session.commit()
        
        # Log ingestion completed
        logger.info(
            {
                "event": "document_ingestion_completed",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "final_status": metadata.status,
                "request_id": request_id,
            }
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.exception(
            {
                "event": "document_ingestion_failed",
                "document_id": metadata_id,
                "filename": getattr(document, "filename", None) if document else None,
                "request_id": request_id,
                "error": error_msg,
            }
        )
        
        # Update status to FAILED
        if session:
            try:
                metadata = session.query(DocumentIngestionMetadata).filter(
                    DocumentIngestionMetadata.id == metadata_id
                ).first()
                if metadata:
                    metadata.status = "FAILED"
                    metadata.error_message = error_msg
                    session.commit()
            except Exception as commit_error:
                logger.error(
                    {
                        "event": "embedding_failed_to_update_status",
                        "metadata_id": metadata_id,
                        "error": str(commit_error),
                        "request_id": request_id,
                    }
                )
    finally:
        if session:
            session.close()

