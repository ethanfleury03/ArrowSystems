"""
Single File Ingestion Utility

Ingests a single document file into the existing RAG index without reprocessing
the entire dataset. Used for admin-controlled document onboarding.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
from llama_index.core import Document, VectorStoreIndex, load_index_from_storage, StorageContext
from llama_index.core.schema import TextNode

from ..ingest import (
    DocumentLoader,
    TextPreprocessor,
    SmartChunkSplitter,
    ClaudeSemanticRewriter,
    NonTextExtractor
)
from .query_summarizer import QuerySummarizer

logger = logging.getLogger(__name__)


def ingest_single_file(
    file_path: str,
    storage_dir: str = "latest_model",
    cache_dir: str = "/root/.cache/huggingface/hub",
    config_path: str = "config.yaml",
    enable_rewriting: bool = False
) -> Dict[str, Any]:
    """
    Ingest a single file into the existing RAG index.
    
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
    
    if not file_path.exists():
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
        logger.info(f"Loading existing index from {storage_dir}...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        index = load_index_from_storage(storage_context)
        
        # Initialize components
        logger.info("Initializing ingestion components...")
        
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
        logger.info(f"Loading document: {file_path.name}...")
        
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
            return {
                "success": False,
                "error": f"Unsupported file type: {file_ext}",
                "doc_id": None,
                "filename": file_path.name,
                "page_count": 0,
                "chunk_count": 0
            }
        
        if not documents:
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
        
        # Step 2: Preprocess documents
        logger.info("Preprocessing documents...")
        preprocessed_docs = []
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
        
        # Step 3: Extract non-text content (tables, images) - only for PDF
        non_text_nodes = []
        if file_ext == '.pdf':
            logger.info("Extracting non-text content...")
            extractor = NonTextExtractor()
            tables = extractor.extract_tables_from_pdf(str(file_path))
            images = extractor.extract_images_from_pdf(str(file_path))
            captions = extractor.extract_captions_from_pdf(str(file_path))
            
            # Create non-text nodes
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
        
        # Step 4: Smart chunking
        logger.info("Chunking documents...")
        text_nodes = smart_splitter.get_nodes_from_documents(preprocessed_docs, show_progress=False)
        
        # Filter nodes
        filtered_nodes = []
        for node in text_nodes:
            should_skip, _ = text_preprocessor.should_skip_node(node.text, metadata=node.metadata)
            if not should_skip:
                filtered_nodes.append(node)
        
        # Step 5: Optional Claude rewriting
        if enable_rewriting and claude_rewriter.enabled:
            logger.info("Rewriting chunks with Claude...")
            rewritten_nodes, _ = claude_rewriter.rewrite_nodes(filtered_nodes, show_progress=False)
            filtered_nodes = rewritten_nodes
        
        # Step 6: Generate summaries for chunks
        logger.info("Generating chunk summaries...")
        for node in filtered_nodes:
            try:
                # Generate summary for this chunk
                summary, was_summarized, _ = query_summarizer.summarize(node.text)
                # Summary is cached automatically by QuerySummarizer
            except Exception as e:
                logger.warning(f"Failed to generate summary for chunk: {e}")
        
        # Step 7: Generate embeddings and add to index
        logger.info("Generating embeddings and adding to index...")
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
                logger.info(f"Initialized embedding model: {embed_model_name}")
            
            # Insert nodes into existing index (embeddings generated automatically)
            batch_size = 50
            for i in range(0, len(all_nodes), batch_size):
                batch = all_nodes[i:i + batch_size]
                index.insert_nodes(batch)
            
            # Persist the updated index
            index.storage_context.persist(persist_dir=storage_dir)
            logger.info(f"Successfully added {len(all_nodes)} nodes to index")
        
        # Get doc_id (use filename as doc_id)
        doc_id = file_path.name
        
        # Update metadata with ingestion date
        from utils.document_metadata import update_ingestion_date
        update_ingestion_date(file_path.name)
        
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
        logger.error(f"Error ingesting file {file_path}: {e}", exc_info=True)
        return {
            "success": False,
            "error": str(e),
            "doc_id": None,
            "filename": file_path.name,
            "page_count": 0,
            "chunk_count": 0
        }

