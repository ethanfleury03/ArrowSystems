"""
Single File Ingestion Utility

Ingests a single document file into the existing RAG index without reprocessing
the entire dataset. Used for admin-controlled document onboarding.
"""

import os
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from llama_index.core import VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.schema import Document
from llama_index.core.schema import TextNode

from ..ingest import (
    DocumentLoader,
    TextPreprocessor,
    SmartChunkSplitter,
    ClaudeSemanticRewriter,
    NonTextExtractor
)
from .query_summarizer import QuerySummarizer
from .filenames import ensure_node_has_filename
from ..logging_config import get_logger
from ..logging_context import get_user_id

logger = get_logger(__name__)


def ingest_single_file(
    file_path: str,
    storage_dir: str = "latest_model",
    cache_dir: str = "/root/.cache/huggingface/hub",
    config_path: str = "config.yaml",
    enable_rewriting: bool = False
) -> Dict[str, Any]:
    """
    INDEX-WRITE PATH: creates/updates embeddings
    
    Ingest a single file into the existing RAG index.
    
    NOTE: This function is intended for use by external GPU ingestion scripts,
    not from the web app. It does not check allow_app_ingestion flag because
    it's meant to be called directly from ingestion workers.
    
    Args:
        file_path: Path to the file to ingest (PDF, DOCX, or Markdown)
        storage_dir: Directory containing the existing vector index
        cache_dir: HuggingFace cache directory
        config_path: Path to config.yaml
        enable_rewriting: Whether to enable Claude semantic rewriting
        
    Returns:
        Dictionary with ingestion results:
        {
            "success": bool,
            "doc_id": str,
            "filename": str,
            "page_count": int,
            "chunk_count": int,
            "error": Optional[str]
        }
    """
    file_path = Path(file_path)
    user_id = get_user_id()
    start_time = time.time()
    
    # Log ingestion start
    logger.info(
        "ingestion_start",
        filename=file_path.name,
        file_path=str(file_path),
        storage_dir=storage_dir,
        enable_rewriting=enable_rewriting,
        user_id=user_id,
    )
    
    if not file_path.exists():
        logger.error("ingestion_file_not_found", filename=file_path.name, file_path=str(file_path))
        return {
            "success": False,
            "error": f"File not found: {file_path}",
            "doc_id": None,
            "filename": file_path.name,
            "page_count": 0,
            "chunk_count": 0
        }
    
    try:
        # Load existing index
        logger.info("ingestion_loading_index", storage_dir=storage_dir)
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        
        # Initialize components
        logger.info("ingestion_initializing_components")
        
        # Load config
        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Initialize text preprocessor
        text_preprocessor = TextPreprocessor()
        
        # Initialize chunk splitter
        chunk_size = config.get("chunking", {}).get("chunk_size", 1536)
        chunk_overlap = config.get("chunking", {}).get("chunk_overlap", 256)
        smart_splitter = SmartChunkSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preprocessor=text_preprocessor
        )
        
        # Initialize Claude rewriter (if enabled)
        claude_rewriter = ClaudeSemanticRewriter(enabled=enable_rewriting)
        
        # Initialize query summarizer for chunk summaries
        query_summarizer = QuerySummarizer(
            enabled=True,
            min_length=0  # Summarize all chunks
        )
        
        # Step 1: Load the single file
        logger.info("ingestion_loading_document", filename=file_path.name)
        load_start_time = time.time()
        
        # Create a temporary data directory with just this file
        temp_data_dir = file_path.parent
        loader = DocumentLoader(str(temp_data_dir))
        
        # Load only this specific file
        documents = []
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.pdf':
            # Use SimpleDirectoryReader for PDF (it handles single files)
            from llama_index.core import SimpleDirectoryReader
            reader = SimpleDirectoryReader(
                input_files=[str(file_path)],
                filename_as_id=True
            )
            loaded_docs = reader.load_data()
            for doc in loaded_docs:
                doc.metadata['file_name'] = file_path.name
                doc.metadata['file_type'] = 'pdf'
                documents.append(doc)
        elif file_ext == '.docx':
            documents = loader._load_docx(file_path)
        elif file_ext in {'.md', '.markdown'}:
            documents = loader._load_markdown(file_path)
        else:
            logger.error("ingestion_unsupported_file_type", filename=file_path.name, file_ext=file_ext)
            return {
                "success": False,
                "error": f"Unsupported file type: {file_ext}",
                "doc_id": None,
                "filename": file_path.name,
                "page_count": 0,
                "chunk_count": 0
            }
        
        load_time_ms = (time.time() - load_start_time) * 1000
        
        if not documents:
            logger.error("ingestion_no_documents_extracted", filename=file_path.name)
            return {
                "success": False,
                "error": "No documents extracted from file",
                "doc_id": None,
                "filename": file_path.name,
                "page_count": 0,
                "chunk_count": 0
            }
        
        # Count pages (for PDF) or sections (for DOCX/MD)
        page_count = len(documents)
        logger.info("ingestion_pages_extracted", filename=file_path.name, pages=page_count, load_time_ms=round(load_time_ms, 2))
        
        # Step 2: Preprocess documents
        logger.info("ingestion_preprocessing", filename=file_path.name)
        preprocess_start_time = time.time()
        preprocessed_docs = []
        skipped_pages = 0
        for doc in documents:
            original_text = doc.text or ""
            cleaned_text = text_preprocessor.clean_text(original_text, metadata=doc.metadata)
            
            if not text_preprocessor.is_low_content_page(cleaned_text):
                if cleaned_text:
                    new_doc = Document(
                        text=cleaned_text,
                        metadata=doc.metadata
                    )
                    preprocessed_docs.append(new_doc)
                else:
                    skipped_pages += 1
            else:
                skipped_pages += 1
        
        preprocess_time_ms = (time.time() - preprocess_start_time) * 1000
        logger.info("ingestion_preprocessing_complete", filename=file_path.name, preprocessed=len(preprocessed_docs), skipped=skipped_pages, latency_ms=round(preprocess_time_ms, 2))
        
        # Step 3: Extract non-text content (tables, captions) - only for PDF
        # Images are permanently disabled - never extracted
        non_text_nodes = []
        if file_ext == '.pdf':
            logger.info("ingestion_extracting_non_text", filename=file_path.name)
            extract_start_time = time.time()
            
            # Pass config to NonTextExtractor so it respects extract_images flag
            extractor = NonTextExtractor(config=config)
            
            tables = extractor.extract_tables_from_pdf(str(file_path))
            
            # Check if images are enabled (though they're permanently disabled in extract_images_from_pdf)
            extract_images_enabled = bool(config.get("non_text", {}).get("extract_images", False))
            if extract_images_enabled:
                # Even if config says enabled, extract_images_from_pdf() always returns empty (hard remove)
                images = extractor.extract_images_from_pdf(str(file_path))
                if len(images) > 0:
                    logger.warning(f"⚠️ extract_images_from_pdf() returned {len(images)} images but should return 0 (images permanently disabled)")
            else:
                # Skip image extraction entirely when disabled (avoid overhead)
                images = []
                logger.debug("Skipping image extraction (disabled in config)")
            
            # Fix: Use correct method name
            captions = extractor.extract_figure_captions(str(file_path))
            
            # Create non-text nodes (tables and captions only - images never processed)
            for table in tables:
                table_text = table.get('table_markdown', '')
                if table_text:
                    node = TextNode(
                        text=table_text,
                        metadata={
                            'file_name': file_path.name,
                            'page_label': str(table.get('page_number', '')),
                            'content_type': 'table',
                            'file_type': 'pdf'
                        }
                    )
                    non_text_nodes.append(node)
            
            # Create caption nodes
            for caption in captions:
                caption_text = caption.get('caption_text', '').strip()
                if caption_text:
                    node = TextNode(
                        text=caption_text,
                        metadata={
                            'file_name': file_path.name,
                            'page_label': str(caption.get('page_number', '')),
                            'content_type': 'figure_caption',
                            'file_type': 'pdf'
                        }
                    )
                    non_text_nodes.append(node)
            
            extract_time_ms = (time.time() - extract_start_time) * 1000
            # Log accurately: images are always 0 (permanently disabled)
            logger.info("ingestion_non_text_extracted", filename=file_path.name, tables=len(tables), images=0, captions=len(captions), nodes=len(non_text_nodes), latency_ms=round(extract_time_ms, 2))
        
        # Step 4: Smart chunking
        logger.info("ingestion_chunking_start", filename=file_path.name)
        chunk_start_time = time.time()
        text_nodes = smart_splitter.get_nodes_from_documents(preprocessed_docs, show_progress=False)
        
        # Filter nodes
        filtered_nodes = []
        skipped_nodes = 0
        for node in text_nodes:
            should_skip, _ = text_preprocessor.should_skip_node(node.text, metadata=node.metadata)
            if not should_skip:
                filtered_nodes.append(node)
            else:
                skipped_nodes += 1
        
        chunk_time_ms = (time.time() - chunk_start_time) * 1000
        logger.info("ingestion_chunking_complete", filename=file_path.name, chunks=len(filtered_nodes), skipped=skipped_nodes, latency_ms=round(chunk_time_ms, 2))
        
        # Step 5: Optional Claude rewriting
        if enable_rewriting and claude_rewriter.enabled:
            logger.info("ingestion_rewriting_start", filename=file_path.name)
            rewrite_start_time = time.time()
            rewritten_nodes, _ = claude_rewriter.rewrite_nodes(filtered_nodes, show_progress=False)
            filtered_nodes = rewritten_nodes
            rewrite_time_ms = (time.time() - rewrite_start_time) * 1000
            logger.info("ingestion_rewriting_complete", filename=file_path.name, rewritten=len(filtered_nodes), latency_ms=round(rewrite_time_ms, 2))
        
        # Step 6: Generate summaries for chunks
        logger.info("ingestion_summarizing_chunks", filename=file_path.name, chunks=len(filtered_nodes))
        summary_start_time = time.time()
        summarized_count = 0
        failed_summaries = 0
        for node in filtered_nodes:
            try:
                # Generate summary for this chunk
                summary, was_summarized, _ = query_summarizer.summarize(node.text)
                # Summary is cached automatically by QuerySummarizer
                if was_summarized:
                    summarized_count += 1
            except Exception as e:
                failed_summaries += 1
                logger.warning("ingestion_summary_failed", filename=file_path.name, error=str(e))
        
        summary_time_ms = (time.time() - summary_start_time) * 1000
        logger.info("ingestion_summarizing_complete", filename=file_path.name, summarized=summarized_count, failed=failed_summaries, latency_ms=round(summary_time_ms, 2))
        
        # Step 7: Generate embeddings and add to index
        logger.info("ingestion_embedding_start", filename=file_path.name, nodes=len(filtered_nodes) + len(non_text_nodes))
        embed_start_time = time.time()
        all_nodes = filtered_nodes + non_text_nodes
        
        if all_nodes:
            # Ensure embedding model is set in Settings
            from llama_index.core import Settings
            if not Settings.embed_model:
                # Initialize embedding model if not already set
                from llama_index.embeddings.huggingface import HuggingFaceEmbedding
                embed_model_name = config.get("models", {}).get("embedding", "BAAI/bge-large-en-v1.5")
                Settings.embed_model = HuggingFaceEmbedding(
                    model_name=embed_model_name,
                    cache_folder=cache_dir
                )
                logger.info("ingestion_embedding_model_initialized", model_name=embed_model_name)
            
            # CRITICAL: Validate and repair filename integrity before indexing
            logger.info(f"Validating filename integrity for {len(all_nodes)} nodes...")
            validated_nodes = []
            repaired_count = 0
            still_missing = 0
            
            for node in all_nodes:
                success, file_name = ensure_node_has_filename(node, strict=True)
                if success:
                    if file_name and not (hasattr(node, 'metadata') and node.metadata.get('file_name')):
                        repaired_count += 1
                    validated_nodes.append(node)
                else:
                    still_missing += 1
                    node_id = getattr(node, 'node_id', None) or getattr(node, 'id_', None) or 'unknown'
                    logger.warning(f"Node missing file_name and cannot repair: {node_id}")
            
            # Strict validation: fail if >0.5% missing
            missing_rate = still_missing / max(len(all_nodes), 1)
            if missing_rate > 0.005:  # 0.5% threshold
                error_msg = (
                    f"CRITICAL: {still_missing} nodes ({missing_rate:.1%}) missing file_name after repair. "
                    f"Exceeds 0.5% threshold. Ingestion aborted."
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            if repaired_count > 0:
                logger.info(f"Repaired {repaired_count} nodes with missing file_name")
            if still_missing > 0:
                logger.warning(f"Dropped {still_missing} nodes that could not be repaired (below threshold)")
            
            # Insert validated nodes into existing index (embeddings generated automatically)
            batch_size = 50
            for i in range(0, len(validated_nodes), batch_size):
                batch = validated_nodes[i:i + batch_size]
                index.insert_nodes(batch)
            
            # Persist the updated index
            index.storage_context.persist(persist_dir=storage_dir)
            
            embed_time_ms = (time.time() - embed_start_time) * 1000
            logger.info("ingestion_embedding_complete", filename=file_path.name, nodes=len(all_nodes), latency_ms=round(embed_time_ms, 2))
        
        # Get doc_id (use filename as doc_id)
        doc_id = file_path.name
        
        # Update metadata with ingestion date and ensure machine_model is set
        from ..utils.document_metadata import ensure_metadata_entry
        meta_entry = ensure_metadata_entry(file_path.name)
        
        # Log if review is needed
        if meta_entry.get("requires_admin_review"):
            logger.warning("ingestion_requires_review", filename=file_path.name, reason="missing machine_model")
        
        # Log ingestion complete
        total_time_ms = (time.time() - start_time) * 1000
        logger.info(
            "ingestion_complete",
            filename=file_path.name,
            page_count=page_count,
            chunk_count=len(all_nodes),
            text_chunks=len(filtered_nodes),
            non_text_chunks=len(non_text_nodes),
            total_latency_ms=round(total_time_ms, 2),
            user_id=user_id,
        )
        
        return {
            "success": True,
            "doc_id": doc_id,
            "filename": file_path.name,
            "page_count": page_count,
            "chunk_count": len(all_nodes),
            "text_chunks": len(filtered_nodes),
            "non_text_chunks": len(non_text_nodes),
            "error": None
        }
        
    except Exception as e:
        total_time_ms = (time.time() - start_time) * 1000
        logger.error(
            "ingestion_failed",
            filename=file_path.name,
            error=str(e),
            total_latency_ms=round(total_time_ms, 2),
            user_id=user_id,
            exc_info=True
        )
        return {
            "success": False,
            "error": str(e),
            "doc_id": None,
            "filename": file_path.name,
            "page_count": 0,
            "chunk_count": 0
        }

