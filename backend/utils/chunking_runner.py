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
from backend.utils.logging_context import get_logger

logger = get_logger(__name__)


def run_chunking(metadata_id: str) -> Optional[str]:
    """
    Run chunking for a document ingestion metadata record.
    
    This function:
    1. Loads the DocumentIngestionMetadata record
    2. Sets status to CHUNKING
    3. Loads the file and converts to text
    4. Chunks the text using SmartChunkSplitter
    5. Saves chunks to temporary storage
    6. Sets status to READY_FOR_EMBEDDING on success
    7. Sets status to FAILED on error
    
    Args:
        metadata_id: The ID of the DocumentIngestionMetadata record
    
    Returns:
        metadata_id if successful, None on failure
    """
    session: Optional[Session] = None
    try:
        # Load metadata record
        session = SessionLocal()
        metadata = session.query(DocumentIngestionMetadata).filter(
            DocumentIngestionMetadata.id == metadata_id
        ).first()
        
        if not metadata:
            logger.error(f"chunking_metadata_not_found", metadata_id=metadata_id)
            return None
        
        # Update status to CHUNKING
        metadata.status = "CHUNKING"
        session.commit()
        logger.info(f"chunking_started", metadata_id=metadata_id, filename=metadata.filename)
        
        # Validate file path exists
        if not metadata.file_path or not os.path.exists(metadata.file_path):
            raise FileNotFoundError(f"File not found: {metadata.file_path}")
        
        file_path = Path(metadata.file_path)
        file_ext = file_path.suffix.lower()
        
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
        for node in text_nodes:
            should_skip, _ = text_preprocessor.should_skip_node(node.text, metadata=node.metadata)
            if not should_skip:
                # Add machine_model and metadata_id to node metadata
                node.metadata['machine_model'] = metadata.machine_model
                node.metadata['ingestion_metadata_id'] = metadata_id
                filtered_nodes.append(node)
        
        logger.info(f"chunking_complete", metadata_id=metadata_id, chunk_count=len(filtered_nodes))
        
        # Save chunks to temporary storage (JSON file)
        # This will be loaded in Phase 3 for embedding
        chunks_dir = Path("data/chunks")
        chunks_dir.mkdir(parents=True, exist_ok=True)
        chunks_file = chunks_dir / f"{metadata_id}.json"
        
        chunks_data = {
            "metadata_id": metadata_id,
            "filename": metadata.filename,
            "machine_model": metadata.machine_model,
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
        logger.error(f"chunking_failed", metadata_id=metadata_id, error=error_msg, exc_info=True)
        
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
                logger.error(f"chunking_failed_to_update_status", metadata_id=metadata_id, error=str(commit_error))
        return None
    finally:
        if session:
            session.close()

