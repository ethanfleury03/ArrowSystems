"""
Chunking runner for Phase 2 of document ingestion.

This module handles the chunking phase of document ingestion:
- Loads documents from files
- Converts to text using existing loaders
- Chunks using SmartChunkSplitter
- Saves chunks to temporary storage (for Phase 3 embedding)
"""

import os
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from sqlalchemy.orm import Session
from llama_index.core import Document
from llama_index.core.schema import TextNode

from backend.utils.db import SessionLocal, DocumentIngestionMetadata
from backend.ingest import DocumentLoader, SmartChunkSplitter, TextPreprocessor
from backend.logging_config import get_logger
from backend.utils.test_mode import get_chunks_dir

logger = get_logger(__name__)


def run_chunking(metadata_id: str, request_id: Optional[str] = None) -> Optional[str]:
    """
    Run chunking for a SINGLE document ingestion metadata record.
    
    INDEX-WRITE PATH: creates chunks for embedding (single-document, incremental)
    
    This function processes ONE document at a time:
    1. Loads the DocumentIngestionMetadata record
    2. Sets status to CHUNKING
    3. Loads the file and converts to text
    4. Chunks the text using SmartChunkSplitter
    5. Saves chunks to temporary storage
    6. Sets status to READY_FOR_EMBEDDING on success
    7. Sets status to FAILED on error
    
    This is safe for Cloud Run CPU environments - processes one document at a time,
    not bulk ingestion of all documents.
    
    IMPORTANT: This function is always allowed. There are no ingestion gates.
    Single-document chunking runs automatically on upload.
    
    Args:
        metadata_id: The ID of the DocumentIngestionMetadata record
        request_id: Optional request ID for tracing
    
    Returns:
        metadata_id if successful, None on failure
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
                    "event": "chunking_metadata_not_found",
                    "metadata_id": metadata_id,
                    "request_id": request_id,
                }
            )
            return None
        
        document = metadata  # Store for error handling
        
        # Log ingestion started
        logger.info(
            {
                "event": "document_ingestion_started",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "request_id": request_id,
            }
        )
        
        # Update status to CHUNKING
        metadata.status = "CHUNKING"
        session.commit()
        logger.info(f"chunking_started", metadata_id=metadata_id, filename=metadata.filename)
        
        # Determine file source: prefer GCS path, fall back to local file_path
        # First, try to get GCS path from Document table
        from backend.utils.db import Document
        doc_record = session.query(Document).filter(Document.file_name == metadata.filename).first()
        gcs_path = doc_record.gcs_path if doc_record else None
        
        # If no GCS path in Document table, check if we have a local file_path
        local_file_path = metadata.file_path if metadata.file_path and os.path.exists(metadata.file_path) else None
        
        # Download from GCS if gcs_path is available
        temp_file_path = None
        file_path = None
        file_ext = None
        
        if gcs_path:
            # Download from GCS to temporary file
            import tempfile
            temp_dir = tempfile.gettempdir()
            temp_file_path = os.path.join(temp_dir, f"ingest_{metadata_id}_{metadata.filename}")
            
            from backend.utils.gcs_client import download_to_file
            logger.info(
                {
                    "event": "document_downloading_from_gcs",
                    "document_id": metadata_id,
                    "filename": metadata.filename,
                    "gcs_path": gcs_path,
                    "request_id": request_id,
                }
            )
            
            if not download_to_file(gcs_path, temp_file_path):
                raise FileNotFoundError(f"Failed to download file from GCS: {gcs_path}")
            
            file_path = Path(temp_file_path)
            file_ext = file_path.suffix.lower()
            file_size = os.path.getsize(temp_file_path)
            
            logger.info(
                {
                    "event": "document_file_loaded_from_gcs",
                    "document_id": metadata_id,
                    "filename": metadata.filename,
                    "gcs_path": gcs_path,
                    "temp_path": temp_file_path,
                    "file_size_bytes": file_size,
                    "request_id": request_id,
                }
            )
        elif local_file_path:
            # Use local file path (fallback for backward compatibility)
            file_path = Path(local_file_path)
            file_ext = file_path.suffix.lower()
            file_size = os.path.getsize(local_file_path)
            
            logger.info(
                {
                    "event": "document_file_loaded_from_local",
                    "document_id": metadata_id,
                    "filename": metadata.filename,
                    "file_path": local_file_path,
                    "file_size_bytes": file_size,
                    "request_id": request_id,
                }
            )
        else:
            raise FileNotFoundError(
                f"File not found for document {metadata_id} ({metadata.filename}). "
                f"Neither GCS path nor local file_path available."
            )
        
        # Log file loaded (for backward compatibility with existing logging)
        logger.debug(
            {
                "event": "document_file_loaded",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "file_size_bytes": file_size,
                "source": "gcs" if gcs_path else "local",
                "request_id": request_id,
            }
        )
        
        # Load document using existing DocumentLoader
        loader = DocumentLoader(str(file_path.parent))
        documents: List[Document] = []
        
        if file_ext == '.pdf':
            from llama_index.core import SimpleDirectoryReader
            pdf_docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
            for doc in pdf_docs:
                doc.metadata['file_name'] = file_path.name
                doc.metadata['file_type'] = 'pdf'
            documents.extend(pdf_docs)
        elif file_ext == '.docx':
            documents = loader._load_docx(file_path)
        elif file_ext in {'.md', '.markdown'}:
            documents = loader._load_markdown(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")
        
        if not documents:
            raise ValueError("No documents extracted from file")
        
        logger.info(f"chunking_documents_loaded", metadata_id=metadata_id, document_count=len(documents))
        
        # Preprocess documents
        text_preprocessor = TextPreprocessor()
        preprocessed_docs = []
        for doc in documents:
            original_text = doc.text or ""
            cleaned_text = text_preprocessor.clean_text(original_text, metadata=doc.metadata)
            if not text_preprocessor.is_low_content_page(cleaned_text) and cleaned_text:
                new_doc = Document(
                    text=cleaned_text,
                    metadata=doc.metadata
                )
                preprocessed_docs.append(new_doc)
        
        logger.info(f"chunking_preprocessed", metadata_id=metadata_id, preprocessed_count=len(preprocessed_docs))
        
        # Check if we have any documents to chunk - if not, fail early
        if len(preprocessed_docs) == 0:
            error_msg = "No content extracted from document. Document may be empty, contain only images, or all content was filtered out as low-quality."
            logger.warning(f"chunking_no_preprocessed_docs", metadata_id=metadata_id, filename=metadata.filename, error=error_msg)
            
            # Update status to FAILED
            metadata.status = "FAILED"
            metadata.error_message = error_msg
            session.commit()
            return None
        
        # Load chunking config
        import yaml
        config_path = "config.yaml"
        if not os.path.exists(config_path):
            config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
        
        config = {}
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
        
        chunk_size = config.get("chunking", {}).get("chunk_size", 350)
        chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 88)
        
        # Create chunk splitter
        smart_splitter = SmartChunkSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preprocessor=text_preprocessor
        )
        
        # Generate chunks
        text_nodes = smart_splitter.get_nodes_from_documents(preprocessed_docs, show_progress=False)
        
        # Filter nodes
        filtered_nodes = []
        # Get document_id from Document table if available
        document_id = doc_record.id if doc_record else None
        # Machine models (canonical): Document↔MachineModel join table
        machine_model_ids: list[int] = []
        machine_model_names: list[str] = []
        try:
            if doc_record and hasattr(doc_record, "machine_models") and doc_record.machine_models:
                machine_model_ids = [int(m.id) for m in doc_record.machine_models]
                machine_model_names = [m.name for m in doc_record.machine_models if getattr(m, "name", None)]
        except Exception:
            machine_model_ids = []
            machine_model_names = []

        # Fallback: resolve single name from ingestion metadata into an ID (best-effort)
        if not machine_model_ids and metadata.machine_model:
            try:
                from backend.utils.db import MachineModel
                from sqlalchemy import func
                mm = session.query(MachineModel).filter(
                    func.upper(MachineModel.name) == " ".join(metadata.machine_model.upper().split())
                ).first()
                if mm:
                    machine_model_ids = [int(mm.id)]
                    machine_model_names = [mm.name]
            except Exception:
                pass
        
        for node in text_nodes:
            should_skip, _ = text_preprocessor.should_skip_node(node.text, metadata=node.metadata)
            if not should_skip:
                # Preserve document_id from upstream doc metadata if present, otherwise set from DB
                if 'document_id' not in node.metadata and document_id is not None:
                    node.metadata['document_id'] = document_id
                # Machine model metadata MUST reflect the document's current join-table values.
                # Overwrite any stale values that may already exist on the node.
                node.metadata["machine_model_ids"] = machine_model_ids
                node.metadata["machine_model_names"] = machine_model_names
                # Backward-compat key used elsewhere for filtering (list[str] preferred)
                node.metadata["machine_model"] = machine_model_names if machine_model_names else metadata.machine_model
                node.metadata['ingestion_metadata_id'] = metadata_id
                filtered_nodes.append(node)
        
        # Log chunking completed
        logger.info(
            {
                "event": "document_chunking_completed",
                "document_id": metadata_id,
                "filename": metadata.filename,
                "num_chunks": len(filtered_nodes),
                "request_id": request_id,
            }
        )
        
        # Check if we have any chunks - if not, fail the ingestion
        if len(filtered_nodes) == 0:
            error_msg = "No chunks generated from document. Document may be empty, contain only images, or all content was filtered out as low-quality."
            logger.warning(f"chunking_no_chunks", metadata_id=metadata_id, filename=metadata.filename, error=error_msg)
            
            # Update status to FAILED
            metadata.status = "FAILED"
            metadata.error_message = error_msg
            session.commit()
            return None
        
        # Save chunks to temporary storage (JSON file)
        # This will be loaded in Phase 3 for embedding
        chunks_dir = Path(get_chunks_dir())
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / f"{metadata_id}.json"
        
        chunks_data = {
            "metadata_id": metadata_id,
            "filename": metadata.filename,
            "machine_model": machine_model_names if machine_model_names else metadata.machine_model,
            "machine_model_ids": machine_model_ids,
            "machine_model_names": machine_model_names,
            "created_at": datetime.utcnow().isoformat(),
            "chunks": [
                {
                    "text": node.text,
                    "metadata": node.metadata,
                    "node_id": getattr(node, 'node_id', None),
                }
                for node in filtered_nodes
            ]
        }
        
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"chunks_saved", metadata_id=metadata_id, chunks_file=str(chunks_file), chunk_count=len(filtered_nodes))
        
        # Update status to READY_FOR_EMBEDDING
        metadata.status = "READY_FOR_EMBEDDING"
        session.commit()
        logger.info(f"chunking_success", metadata_id=metadata_id, filename=metadata.filename, chunk_count=len(filtered_nodes))
        
        # Trigger embedding runner (Phase 3) - schedule in background
        # Note: This will be handled by the upload endpoint's background task system
        # We return the metadata_id so the caller can schedule embedding
        return metadata_id
        
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
                        "event": "chunking_failed_to_update_status",
                        "metadata_id": metadata_id,
                        "error": str(commit_error),
                        "request_id": request_id,
                    }
                )
        return None
    finally:
        # Clean up temporary file if downloaded from GCS
        if 'temp_file_path' in locals() and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.debug(
                    {
                        "event": "temp_file_cleaned_up",
                        "document_id": metadata_id,
                        "temp_path": temp_file_path,
                    }
                )
            except Exception as cleanup_error:
                logger.warning(
                    {
                        "event": "temp_file_cleanup_failed",
                        "document_id": metadata_id,
                        "temp_path": temp_file_path,
                        "error": str(cleanup_error),
                    }
                )
        if session:
            session.close()

