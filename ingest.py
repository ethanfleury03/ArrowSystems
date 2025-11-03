"""
High-Performance RAG Pipeline for Technical Documents
Optimized for GPU rental with bge-large-en-v1.5 and re-ranking
Enhanced with non-text content extraction (tables, images, diagrams)
"""

import warnings
# Suppress annoying Pydantic warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")
# Suppress pypdf warnings for malformed PDFs
warnings.filterwarnings("ignore", message=".*wrong pointing object.*")

import os
import logging
import time
import json
import yaml
import re
from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import base64
from io import BytesIO

import fitz  # PyMuPDF
import pandas as pd
from PIL import Image
import numpy as np
from tqdm import tqdm

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore, TextNode, ImageNode, Document
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from sentence_transformers import CrossEncoder
import qdrant_client
import shutil
import tarfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress pypdf warnings for malformed PDFs
logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._reader").setLevel(logging.ERROR)


class TextPreprocessor:
    """
    Preprocesses text to remove boilerplate, normalize whitespace, and filter low-quality content.
    """
    
    def __init__(self):
        # Compile regex patterns for common boilerplate text
        # Using more specific patterns to avoid catastrophic backtracking
        self.boilerplate_patterns = [
            # Page numbers (various formats) - more specific to avoid hanging
            re.compile(r'^[ \t]*[Pp]age[ \t]+\d+[ \t]*$', re.MULTILINE),
            re.compile(r'^[ \t]*\d{1,4}[ \t]*$', re.MULTILINE),  # Standalone page numbers (limit digits)
            re.compile(r'^[ \t]*-[ \t]*\d+[ \t]*-[ \t]*$', re.MULTILINE),  # "- 5 -"
            
            # Confidential/Proprietary markings
            re.compile(r'\b[Mm]emjet[ \t]+[Cc]onfidential\b', re.IGNORECASE),
            re.compile(r'\b[Cc]onfidential\b', re.IGNORECASE),
            re.compile(r'\b[Pp]roprietary\b', re.IGNORECASE),
            
            # Copyright notices (various formats) - limit length to prevent hanging
            re.compile(r'©[ \t]*\d{4}[^\n]{0,100}$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'Copyright[ \t]+©?[ \t]*\d{4}[^\n]{0,100}$', re.MULTILINE | re.IGNORECASE),
            re.compile(r'All[ \t]+rights[ \t]+reserved\.?', re.IGNORECASE),
            
            # Common header/footer patterns
            re.compile(r'^[ \t]*(Document|Version|Rev|Revision)[ \t]*:[ \t]*[\w\.-]+[ \t]*$', re.MULTILINE | re.IGNORECASE),
            
            # Repeated section titles (if they appear multiple times)
            re.compile(r'^[ \t]*(Table[ \t]+of[ \t]+Contents|Contents|Index)[ \t]*$', re.MULTILINE | re.IGNORECASE),
            
            # Date stamps in headers/footers
            re.compile(r'^[ \t]*\d{1,2}[/-]\d{1,2}[/-]\d{2,4}[ \t]*$', re.MULTILINE),
        ]
        
        # Patterns for structured lines that should be preserved (for smart chunking)
        self.preserve_patterns = [
            re.compile(r'^(Usage|Command|Example|Syntax|Parameters?|Options?|Steps?|Procedure|Note|Warning|Important):\s*', re.MULTILINE | re.IGNORECASE),
            re.compile(r'^\s*\d+[\.\)]\s+', re.MULTILINE),  # Numbered lists: "1. ", "2) "
            re.compile(r'^[-*•]\s+', re.MULTILINE),  # Bullet points
            re.compile(r'^\s*[A-Z][a-z]+:\s*$', re.MULTILINE),  # Section headers ending with colon
        ]
    
    def remove_boilerplate(self, text: str) -> str:
        """Remove common boilerplate text patterns from the input."""
        cleaned = text
        
        # Apply each boilerplate pattern with error handling
        for pattern in self.boilerplate_patterns:
            try:
                cleaned = pattern.sub('', cleaned)
            except Exception as e:
                # If regex fails, skip this pattern and continue
                logger.debug(f"Regex pattern failed, skipping: {e}")
                continue
        
        return cleaned
    
    def normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace: collapse multiple spaces/tabs, normalize newlines."""
        try:
            # Replace tabs with spaces
            text = text.replace('\t', ' ')
            
            # Collapse multiple spaces into single space (limit to prevent hanging)
            text = re.sub(r' {2,}', ' ', text)
            
            # Normalize line breaks: multiple newlines -> double newline (paragraph break)
            text = re.sub(r'\n{3,}', '\n\n', text)
            
            # Remove leading/trailing whitespace from each line
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(lines)
            
            # Remove leading/trailing whitespace from entire text
            text = text.strip()
        except Exception as e:
            logger.debug(f"Whitespace normalization failed: {e}")
            # Return original text if normalization fails
            pass
        
        return text
    
    def should_preserve_line(self, line: str) -> bool:
        """Check if a line matches patterns that should be preserved as-is."""
        for pattern in self.preserve_patterns:
            if pattern.search(line):
                return True
        return False
    
    def clean_text(self, text: str) -> str:
        """Apply all cleaning steps: remove boilerplate and normalize whitespace."""
        cleaned = self.remove_boilerplate(text)
        cleaned = self.normalize_whitespace(cleaned)
        return cleaned
    
    def is_low_content_page(self, text: str, min_words: int = 15) -> bool:
        """Check if a page has too little content to be useful."""
        words = len(text.split())
        return words < min_words
    
    def should_skip_node(self, text: str, min_chars: int = 30) -> bool:
        """Check if a node should be skipped (too short or empty)."""
        if not text or len(text.strip()) < min_chars:
            return True
        
        # Check if it's mostly whitespace or special characters
        alpha_chars = len(re.findall(r'[a-zA-Z]', text))
        if alpha_chars < min_chars // 2:  # At least half should be alphabetic
            return True
        
        return False


class SmartChunkSplitter:
    """
    Wrapper around SentenceSplitter that preserves structured content like tables,
    code blocks, numbered steps, and command syntax.
    """
    
    def __init__(self, chunk_size: int = 350, chunk_overlap: int = 88, preprocessor: TextPreprocessor = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.base_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True
        )
        self.preprocessor = preprocessor or TextPreprocessor()
    
    def _is_table_content(self, text: str) -> bool:
        """Detect if text contains table-like content (markdown table or pipe-separated)."""
        # Check for markdown table pattern (| col1 | col2 |)
        if re.search(r'\|[^\|]+\|', text) and text.count('|') >= 6:
            return True
        # Check for tab-separated or multiple spaces (likely table)
        lines = text.split('\n')
        if len(lines) >= 3:
            tabs_or_spaces = sum(1 for line in lines if '\t' in line or re.search(r' {3,}', line))
            if tabs_or_spaces >= len(lines) * 0.6:  # 60% of lines have table-like spacing
                return True
        return False
    
    def _is_code_block(self, text: str) -> bool:
        """Detect if text contains code blocks."""
        # Check for code block markers
        if '```' in text or '`' in text:
            return True
        # Check for common code patterns
        code_keywords = ['def ', 'class ', 'import ', 'function', 'return', 'const ', 'var ', 'let ']
        if any(keyword in text for keyword in code_keywords):
            return True
        return False
    
    def _preserve_structured_chunks(self, text: str) -> List[str]:
        """
        Split text while preserving structured content as single units.
        Returns list of text chunks.
        """
        chunks = []
        lines = text.split('\n')
        current_chunk = []
        current_chunk_size = 0
        
        i = 0
        while i < len(lines):
            line = lines[i]
            line_stripped = line.strip()
            
            # Check if this line starts a structured block
            if self.preprocessor.should_preserve_line(line_stripped):
                # Try to collect the entire structured block
                structured_block = [line]
                block_size = len(line)
                j = i + 1
                
                # Collect lines until we hit a non-structured line or exceed chunk size
                while j < len(lines) and block_size + len(lines[j]) < self.chunk_size:
                    next_line = lines[j].strip()
                    # Continue if it's part of the structure (numbered, bullet, or continuation)
                    if (next_line.startswith((' ', '\t')) or  # Indented continuation
                        re.match(r'^\d+[\.\)]', next_line) or  # Next numbered item
                        re.match(r'^[-*•]', next_line) or  # Next bullet
                        re.match(r'^[A-Z][a-z]+:\s*$', next_line)):  # Next section header
                        structured_block.append(lines[j])
                        block_size += len(lines[j])
                        j += 1
                    else:
                        break
                
                # If we have accumulated text, save it first
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_chunk_size = 0
                
                # Add the structured block as a single chunk
                structured_text = '\n'.join(structured_block)
                # Check if it exceeds chunk size (rare, but handle it)
                if len(structured_text) > self.chunk_size:
                    # Split the structured block using base splitter, but preserve internal structure
                    sub_chunks = self.base_splitter.split_text(structured_text)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(structured_text)
                
                i = j
                continue
            
            # Regular line - add to current chunk
            line_size = len(line)
            if current_chunk_size + line_size <= self.chunk_size:
                current_chunk.append(line)
                current_chunk_size += line_size
                i += 1
            else:
                # Current chunk is full, save it
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    # Start new chunk with overlap
                    overlap_lines = []
                    overlap_size = 0
                    # Get last few lines for overlap (up to chunk_overlap chars)
                    for line_idx in range(len(current_chunk) - 1, -1, -1):
                        if overlap_size + len(current_chunk[line_idx]) <= self.chunk_overlap:
                            overlap_lines.insert(0, current_chunk[line_idx])
                            overlap_size += len(current_chunk[line_idx])
                        else:
                            break
                    current_chunk = overlap_lines + [line]
                    current_chunk_size = overlap_size + line_size
                    i += 1
        
        # Add remaining chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def split_text(self, text: str) -> List[str]:
        """Split text with smart chunking that preserves structured content."""
        # Check if entire text is a table or code block
        if self._is_table_content(text) or self._is_code_block(text):
            # Preserve as single chunk (may exceed chunk_size, but that's okay for structured content)
            return [text]
        
        # Use smart chunking
        return self._preserve_structured_chunks(text)
    
    def get_nodes_from_documents(self, documents: List[Document], show_progress: bool = False) -> List[TextNode]:
        """
        Split documents into nodes with smart chunking.
        This is a simplified version that processes documents one by one.
        """
        all_nodes = []
        
        for doc in tqdm(documents, desc="Splitting documents", disable=not show_progress):
            try:
                text = doc.text or ""
                doc_name = doc.metadata.get('file_name', 'unknown')
                
                # Clean the text first (with error handling)
                try:
                    text = self.preprocessor.clean_text(text)
                except Exception as e:
                    logger.warning(f"Error cleaning text for {doc_name}: {e}")
                    # Use original text if cleaning fails
                    text = doc.text or ""
                
                # Check if page/document should be skipped (low content)
                try:
                    if self.preprocessor.is_low_content_page(text):
                        logger.debug(f"Skipping low-content page: {doc_name}")
                        continue
                except Exception as e:
                    logger.debug(f"Error checking low-content for {doc_name}: {e}")
                    # Continue processing if check fails
                
                # Split into chunks (with error handling)
                try:
                    chunks = self.split_text(text)
                except Exception as e:
                    logger.error(f"Error splitting text for {doc_name}: {e}")
                    # Fallback to simple split if smart chunking fails
                    if text:
                        chunks = [text]
                    else:
                        chunks = []
                
                # Create nodes from chunks
                for chunk_idx, chunk_text in enumerate(chunks):
                    # Skip if chunk is too short
                    try:
                        if self.preprocessor.should_skip_node(chunk_text):
                            continue
                    except Exception as e:
                        logger.debug(f"Error checking skip node: {e}")
                        # Continue if check fails
                    
                    # Create node with metadata
                    try:
                        node = TextNode(
                            text=chunk_text,
                            metadata={
                                **doc.metadata,
                                "chunk_index": chunk_idx,
                                "total_chunks": len(chunks)
                            }
                        )
                        all_nodes.append(node)
                    except Exception as e:
                        logger.error(f"Error creating node for {doc_name}, chunk {chunk_idx}: {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Error processing document {doc.metadata.get('file_name', 'unknown')}: {e}")
                continue
        
        return all_nodes


class NonTextExtractor:
    """Extract and process non-text content from documents."""
    
    def __init__(self, output_dir="/workspace/extracted_content"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
    def extract_tables_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract tables from PDF using PyMuPDF."""
        tables = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            # Extract tables using PyMuPDF's table detection
            page_tables = page.find_tables()
            
            for table_idx, table in enumerate(page_tables):
                try:
                    # Extract table data
                    table_data = table.extract()
                    if table_data and len(table_data) > 1:  # Ensure we have headers and data
                        # Convert to pandas DataFrame for better structure
                        headers = table_data[0]
                        # Handle duplicate column names
                        unique_headers = []
                        for i, header in enumerate(headers):
                            if header in unique_headers:
                                unique_headers.append(f"{header}_{i}")
                            else:
                                unique_headers.append(header)
                        
                        df = pd.DataFrame(table_data[1:], columns=unique_headers)
                        
                        # Create table metadata
                        table_info = {
                            "source_path": pdf_path,
                            "page_number": page_num + 1,
                            "table_index": table_idx,
                            "table_data": df.to_dict('records'),
                            "table_markdown": df.to_markdown(index=False),
                            "table_json": df.to_json(orient='records'),
                            "row_count": len(df),
                            "column_count": len(df.columns),
                            "content_type": "table"
                        }
                        tables.append(table_info)
                        
                        # Save table as separate file
                        table_filename = f"{Path(pdf_path).stem}_page{page_num+1}_table{table_idx}.json"
                        table_path = self.output_dir / table_filename
                        with open(table_path, 'w', encoding='utf-8') as f:
                            json.dump(table_info, f, indent=2, ensure_ascii=False)
                            
                except Exception as e:
                    logger.warning(f"Failed to extract table {table_idx} from page {page_num + 1}: {e}")
                    continue
                    
        doc.close()
        return tables
    
    def extract_images_from_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract images and diagrams from PDF."""
        images = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_idx, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        # Convert to PIL Image
                        img_data = pix.tobytes("png")
                        pil_image = Image.open(BytesIO(img_data))
                        
                        # Get image metadata
                        img_rects = page.get_image_rects(xref)
                        img_rect = img_rects[0] if img_rects else None
                        
                        # Create image info (NO base64 embedding - only metadata)
                        image_info = {
                            "source_path": pdf_path,
                            "page_number": page_num + 1,
                            "image_index": img_idx,
                            # Removed: image_data base64 encoding (not needed for embeddings)
                            "width": pil_image.width,
                            "height": pil_image.height,
                            "format": "PNG",
                            "content_type": "image",
                            "caption": f"Image from {Path(pdf_path).stem}, page {page_num + 1}",
                            "bbox": str(img_rect) if img_rect else None
                        }
                        images.append(image_info)
                        
                        # Save image
                        img_filename = f"{Path(pdf_path).stem}_page{page_num+1}_img{img_idx}.png"
                        img_path = self.output_dir / img_filename
                        pil_image.save(img_path)
                        image_info["saved_path"] = str(img_path)
                        
                except Exception as e:
                    logger.warning(f"Failed to extract image {img_idx} from page {page_num + 1}: {e}")
                    continue
                    
        doc.close()
        return images
    
    def extract_figure_captions(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract figure captions and references."""
        captions = []
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Look for figure captions (simple pattern matching)
            lines = text.split('\n')
            for i, line in enumerate(lines):
                line_lower = line.lower().strip()
                if any(keyword in line_lower for keyword in ['figure', 'fig.', 'diagram', 'chart', 'graph']):
                    # Extract caption text
                    caption_text = line.strip()
                    if len(caption_text) > 10:  # Filter out very short matches
                        caption_info = {
                            "source_path": pdf_path,
                            "page_number": page_num + 1,
                            "caption_text": caption_text,
                            "content_type": "figure_caption",
                            "line_number": i + 1
                        }
                        captions.append(caption_info)
        
        doc.close()
        return captions


class TechnicalRAGPipeline:
    """High-performance RAG pipeline optimized for technical documentation with non-text content support."""
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub", config_path="config.yaml"):
        self.cache_dir = cache_dir
        self.embed_model = None
        self.reranker = None
        self.index = None
        self.non_text_extractor = NonTextExtractor()
        self.text_preprocessor = TextPreprocessor()
        self.config = self._load_config(config_path)
        
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file or use defaults."""
        default_config = {
            "qdrant": {
                "url": "http://localhost:6333",
                "collection_name": "technical_docs"
            },
            "models": {
                "embedding": "BAAI/bge-large-en-v1.5",
                "reranker": "BAAI/bge-reranker-large"
            },
            "chunking": {
                "chunk_size": 512,
                "chunk_overlap": 128
            }
        }
        
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
                return {**default_config, **config}
            except Exception as e:
                logger.warning(f"Failed to load config: {e}, using defaults")
        
        return default_config
        
    def initialize_models(self):
        """Initialize embedding and re-ranking models."""
        logger.info("🚀 Initializing embedding model...")
        
        # Disable hf_transfer if not installed (RunPod issue)
        import os
        import shutil
        if os.environ.get('HF_HUB_ENABLE_HF_TRANSFER') == '1':
            logger.info("Disabling HF_HUB_ENABLE_HF_TRANSFER (package not installed)")
            os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '0'
        
        # Detect GPU with fallback to CPU if CUDA is incompatible
        import torch
        device = "cpu"  # Default to CPU
        try:
            if torch.cuda.is_available():
                # Test if CUDA actually works with a real operation
                try:
                    test_tensor = torch.zeros(1).cuda()
                    # Try a simple operation that requires kernel execution
                    result = test_tensor + 1
                    result.item()  # Force execution
                    del test_tensor, result
                    torch.cuda.empty_cache()
                    device = "cuda"
                    logger.info(f"🖥️ Using device: {device}")
                    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
                    logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
                except RuntimeError as cuda_error:
                    if "CUDA" in str(cuda_error) or "kernel" in str(cuda_error).lower():
                        logger.warning(f"⚠️ CUDA not compatible (kernel error), falling back to CPU: {cuda_error}")
                        device = "cpu"
                        logger.info(f"🖥️ Using device: {device}")
                    else:
                        raise
            else:
                logger.info(f"🖥️ Using device: {device} (CUDA not available)")
        except Exception as e:
            logger.warning(f"⚠️ CUDA test failed, falling back to CPU: {e}")
            device = "cpu"
            logger.info(f"🖥️ Using device: {device}")
        
        cache_path = os.path.expanduser(self.cache_dir)
        
        # Try multiple approaches
        model_options = [
            ("BAAI/bge-large-en-v1.5", "BGE Large"),
            ("BAAI/bge-base-en-v1.5", "BGE Base"),
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet")
        ]
        
        for model_name, display_name in model_options:
            try:
                logger.info(f"Trying model: {display_name} ({model_name})")
                
                # Method 1: Direct load without sentence-transformers prefix
                try:
                    # Force CPU if CUDA is incompatible
                    if device == "cuda":
                        try:
                            # Test CUDA compatibility
                            import torch
                            test = torch.zeros(1).cuda()
                            del test
                        except Exception:
                            logger.warning(f"CUDA incompatible, forcing CPU for {display_name}")
                            device = "cpu"
                    
                    self.embed_model = HuggingFaceEmbedding(
                        model_name=model_name,
                        cache_folder=self.cache_dir,
                        trust_remote_code=True,
                        device=device
                    )
                    logger.info(f"✅ Successfully loaded: {display_name} on {device}")
                    break
                except Exception as e1:
                    logger.debug(f"Method 1 failed: {e1}")
                    
                    # Method 2: Try with full sentence-transformers path
                    if not model_name.startswith("sentence-transformers/"):
                        try:
                            full_name = f"sentence-transformers/{model_name}"
                            self.embed_model = HuggingFaceEmbedding(
                                model_name=full_name,
                                cache_folder=self.cache_dir,
                                trust_remote_code=True,
                                device=device
                            )
                            logger.info(f"✅ Successfully loaded: {display_name} on {device}")
                            break
                        except Exception as e2:
                            logger.debug(f"Method 2 failed: {e2}")
                            raise e1
                    else:
                        raise e1
                        
            except Exception as e:
                logger.warning(f"Failed to load {display_name}: {str(e)[:100]}")
                continue
        
        if not self.embed_model:
            logger.error("All model loading attempts failed. Trying emergency fallback...")
            # Emergency fallback - use any available model
            try:
                self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
                logger.info("✅ Loaded with emergency fallback")
            except:
                raise RuntimeError("Could not load any embedding model. Check internet connection and HuggingFace access.")
        
        # Try to initialize re-ranker (optional)
        try:
            logger.info("🎯 Initializing re-ranker...")
            reranker_model = self.config.get("models", {}).get("reranker", "BAAI/bge-reranker-large")
            # Use CPU if CUDA was incompatible
            reranker_device = device if device == "cpu" else device
            try:
                self.reranker = CrossEncoder(
                    reranker_model,
                    cache_folder=self.cache_dir,
                    device=reranker_device
                )
                logger.info(f"✅ Re-ranker loaded successfully on {reranker_device}")
            except RuntimeError as cuda_error:
                if "CUDA" in str(cuda_error) or "cuda" in str(cuda_error).lower():
                    logger.warning(f"⚠️ CUDA incompatible for reranker, using CPU: {cuda_error}")
                    self.reranker = CrossEncoder(
                        reranker_model,
                        cache_folder=self.cache_dir,
                        device="cpu"
                    )
                    logger.info(f"✅ Re-ranker loaded successfully on CPU")
                else:
                    raise
        except Exception as e:
            logger.warning(f"Re-ranker not available: {e}")
            self.reranker = None
        
        # Set global embedding model
        Settings.embed_model = self.embed_model
        logger.info("✅ Models initialized successfully")
    
    def process_non_text_content(self, data_dir: str) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Process non-text content (tables, images, captions) from documents."""
        logger.info("📊 Processing non-text content...")
        
        all_tables = []
        all_images = []
        all_captions = []
        
        # Find all PDF files
        pdf_files = list(Path(data_dir).glob("*.pdf"))
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_path in pdf_files:
            logger.info(f"Processing {pdf_path.name}...")
            
            try:
                # Extract tables
                tables = self.non_text_extractor.extract_tables_from_pdf(str(pdf_path))
                all_tables.extend(tables)
                logger.info(f"Extracted {len(tables)} tables from {pdf_path.name}")
                
                # Extract images
                images = self.non_text_extractor.extract_images_from_pdf(str(pdf_path))
                all_images.extend(images)
                logger.info(f"Extracted {len(images)} images from {pdf_path.name}")
                
                # Extract captions
                captions = self.non_text_extractor.extract_figure_captions(str(pdf_path))
                all_captions.extend(captions)
                logger.info(f"Extracted {len(captions)} captions from {pdf_path.name}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")
                continue
        
        logger.info(f"✅ Non-text processing complete: {len(all_tables)} tables, {len(all_images)} images, {len(all_captions)} captions")
        return all_tables, all_images, all_captions
    
    def create_non_text_nodes(self, tables: List[Dict], images: List[Dict], captions: List[Dict]) -> List[TextNode]:
        """Create TextNode objects for non-text content to be embedded."""
        nodes = []
        
        # Process tables
        for table in tables:
            # Create text representation of table
            table_text = f"Table from {Path(table['source_path']).name}, page {table['page_number']}:\n{table['table_markdown']}"
            
            node = TextNode(
                text=table_text,
                metadata={
                    "content_type": "table",
                    "source_path": table["source_path"],
                    "page_number": table["page_number"],
                    "table_index": table["table_index"],
                    "row_count": table["row_count"],
                    "column_count": table["column_count"],
                    "table_json": table["table_json"]
                }
            )
            nodes.append(node)
        
        # Process figure captions
        for caption in captions:
            # Clean caption text to remove boilerplate
            caption_text = self.text_preprocessor.clean_text(caption["caption_text"])
            if not self.text_preprocessor.should_skip_node(caption_text):
                node = TextNode(
                    text=caption_text,
                    metadata={
                        "content_type": "figure_caption",
                        "source_path": caption["source_path"],
                        "page_number": caption["page_number"],
                        "line_number": caption["line_number"]
                    }
                )
                nodes.append(node)
        
        # Process images (create text nodes for captions and metadata only - no base64)
        for image in images:
            image_text = f"Image from {Path(image['source_path']).name}, page {image['page_number']}: {image['caption']}"
            
            node = TextNode(
                text=image_text,
                metadata={
                    "content_type": "image",
                    "source_path": image["source_path"],
                    "page_number": image["page_number"],
                    "image_index": image["image_index"],
                    "width": image["width"],
                    "height": image["height"],
                    "saved_path": image.get("saved_path"),
                    "bbox": str(image.get("bbox")) if image.get("bbox") else None
                }
            )
            nodes.append(node)
        
        return nodes
    
    def setup_qdrant_storage(self) -> StorageContext:
        """Setup Qdrant vector store for hybrid search."""
        try:
            # Initialize Qdrant client
            qdrant_url = self.config["qdrant"]["url"]
            collection_name = self.config["qdrant"]["collection_name"]
            
            client = qdrant_client.QdrantClient(url=qdrant_url)
            
            # Create vector store
            vector_store = QdrantVectorStore(
                client=client,
                collection_name=collection_name
            )
            
            # Create storage context
            storage_context = StorageContext.from_defaults(
                vector_store=vector_store,
                docstore=SimpleDocumentStore(),
                index_store=SimpleIndexStore()
            )
            
            logger.info(f"✅ Qdrant storage configured: {qdrant_url}/{collection_name}")
            return storage_context
            
        except Exception as e:
            logger.warning(f"Qdrant not available, using local storage: {e}")
            return None
    
    def build_index(self, data_dir="data", storage_dir="latest_model", use_qdrant=False):
        """Build or load vector index with optimized chunking and non-text content."""
        
        # Initialize models
        self.initialize_models()
        
        # Setup storage context
        storage_context = None
        if use_qdrant:
            storage_context = self.setup_qdrant_storage()
        
        # For local storage: always rebuild (clear old index first)
        if not use_qdrant and os.path.exists(storage_dir):
            logger.info(f"🗑️  Clearing old index from {storage_dir} for fresh rebuild...")
            shutil.rmtree(storage_dir)
            logger.info("✅ Old index cleared - ready for fresh build")
        
        # Create storage directory if it doesn't exist
        if not use_qdrant:
            os.makedirs(storage_dir, exist_ok=True)
        
        print("\n" + "="*70)
        print("📥 BUILDING NEW RAG INDEX")
        print("="*70)
        
        # Step 1: Load Documents
        print("\n[Step 1/6] 📄 Loading PDF documents...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"   ✅ Loaded {len(documents)} PDF documents")
        logger.info(f"Loaded {len(documents)} text documents")
        
        # Step 2: Preprocess Documents (Remove boilerplate, normalize whitespace)
        print("\n[Step 2/6] 🧹 Preprocessing documents (removing boilerplate, normalizing text)...")
        preprocessed_docs = []
        skipped_pages = 0
        for doc in documents:
            original_text = doc.text or ""
            cleaned_text = self.text_preprocessor.clean_text(original_text)
            
            # Skip low-content pages
            if self.text_preprocessor.is_low_content_page(cleaned_text):
                skipped_pages += 1
                logger.debug(f"Skipping low-content page: {doc.metadata.get('file_name', 'unknown')}")
                continue
            
            # Create new document with cleaned text
            if cleaned_text:  # Only add if there's content left after cleaning
                new_doc = Document(
                    text=cleaned_text,
                    metadata=doc.metadata
                )
                preprocessed_docs.append(new_doc)
        
        print(f"   ✅ Preprocessed {len(preprocessed_docs)} documents ({skipped_pages} low-content pages skipped)")
        logger.info(f"Preprocessed {len(preprocessed_docs)} documents, skipped {skipped_pages} low-content pages")
        
        # Step 3: Extract Non-Text Content
        print("\n[Step 3/6] 🖼️  Extracting tables, images, and captions...")
        print("   This may take a few minutes...")
        tables, images, captions = self.process_non_text_content(data_dir)
        print(f"   ✅ Extracted {len(tables)} tables, {len(images)} images, {len(captions)} captions")
        
        # Step 4: Create Non-Text Nodes
        print("\n[Step 4/6] 📊 Creating searchable nodes from extracted content...")
        non_text_nodes = self.create_non_text_nodes(tables, images, captions)
        print(f"   ✅ Created {len(non_text_nodes)} non-text nodes")
        logger.info(f"Created {len(non_text_nodes)} non-text nodes")
        
        # Step 5: Smart Chunking with Text Nodes
        print("\n[Step 5/6] 🧠 Smart chunking and filtering...")
        chunk_size = self.config.get("chunking", {}).get("chunk_size", 350)
        chunk_overlap = self.config.get("chunking", {}).get("chunk_overlap", 88)
        
        smart_splitter = SmartChunkSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            preprocessor=self.text_preprocessor
        )
        
        print(f"   - Chunk size: {chunk_size} characters")
        print(f"   - Chunk overlap: {chunk_overlap} characters")
        print(f"   - Preserving structured content (tables, code blocks, numbered steps)...")
        
        # Get nodes from documents using smart chunking
        text_nodes = smart_splitter.get_nodes_from_documents(preprocessed_docs, show_progress=True)
        
        # Filter out short/low-quality nodes (only for text nodes, not tables/images)
        filtered_nodes = []
        skipped_nodes = 0
        for node in text_nodes:
            # Skip filtering for non-text content types (tables, images, captions are already handled separately)
            content_type = node.metadata.get("content_type", "text")
            if content_type != "text":
                filtered_nodes.append(node)  # Don't filter non-text content
            elif not self.text_preprocessor.should_skip_node(node.text):
                filtered_nodes.append(node)
            else:
                skipped_nodes += 1
        
        print(f"   ✅ Created {len(filtered_nodes)} text nodes ({skipped_nodes} low-quality nodes filtered)")
        logger.info(f"Created {len(filtered_nodes)} text nodes, filtered {skipped_nodes} low-quality nodes")
        
        # Step 6: Create Vector Embeddings (LONGEST STEP)
        print("\n[Step 6/6] 🧠 Generating embeddings and building vector index...")
        print(f"   - Processing {len(filtered_nodes)} text nodes + {len(non_text_nodes)} non-text nodes...")
        print(f"   - This is the LONGEST step (embedding generation)")
        print(f"   - Expected time: 5-15 minutes on GPU, 30-60 minutes on CPU")
        print(f"   - Watch for progress below...")
        print("")
        
        # Record start time
        import time
        start_time = time.time()
        
        # Combine all nodes
        all_nodes = filtered_nodes + non_text_nodes
        
        # Create index from nodes (LlamaIndex API)
        if storage_context:
            # Create index with storage context
            self.index = VectorStoreIndex(
                nodes=[],
                storage_context=storage_context,
                show_progress=True
            )
        else:
            # Create index without storage context (will use default)
            self.index = VectorStoreIndex(
                nodes=[],
                show_progress=True
            )
        
        # Insert all nodes into the index (batch insert for better performance)
        print(f"   Inserting {len(all_nodes)} nodes into index...")
        batch_size = 100  # Insert in batches
        for i in tqdm(range(0, len(all_nodes), batch_size), desc="   Inserting nodes", unit="batch"):
            batch = all_nodes[i:i + batch_size]
            try:
                self.index.insert_nodes(batch)
            except RuntimeError as e:
                if "CUDA" in str(e) or "cuda" in str(e).lower():
                    logger.error(f"CUDA error during embedding - device should have been set to CPU")
                    logger.error(f"Error: {e}")
                    raise RuntimeError("CUDA incompatible. The embedding model should use CPU. Check device detection.") from e
                else:
                    raise
        
        elapsed = time.time() - start_time
        print(f"\n   ✅ Vector index created in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   ⚡ Processing speed: {len(all_nodes) / elapsed:.2f} nodes/sec")
        
        # Persist the index (only for local storage)
        print(f"\n💾 Saving index to disk...")
        if not use_qdrant:
            self.index.storage_context.persist(persist_dir=storage_dir)
            print(f"   ✅ Index saved to: {storage_dir}")
            logger.info("✅ Index created and saved locally")
        else:
            print(f"   ✅ Index saved to: Qdrant")
            logger.info("✅ Index created and saved to Qdrant")
        
        # Final summary
        total_time = time.time() - start_time
        print("\n" + "="*70)
        print("✅ INGESTION COMPLETE!")
        print("="*70)
        print(f"📊 Documents loaded: {len(documents)}")
        print(f"📊 Documents after preprocessing: {len(preprocessed_docs)} ({skipped_pages} low-content pages skipped)")
        print(f"📊 Text nodes created: {len(filtered_nodes)} ({skipped_nodes} filtered)")
        print(f"📊 Non-text nodes: {len(non_text_nodes)}")
        print(f"📊 Total nodes indexed: {len(all_nodes)}")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        if not use_qdrant:
            print(f"📁 Storage location: {storage_dir}")
            print(f"\n💡 Two-Pod Workflow:")
            print(f"   1. git add {storage_dir}/")
            print(f"   2. git commit -m 'Update RAG index'")
            print(f"   3. git push")
            print(f"   4. Switch to cheap pod → git pull → streamlit run app.py")
        print(f"📂 Extracted content: extracted_content/")
        print(f"🔍 Ready to query!")
        print("="*70 + "\n")
        
        return self.index
    
    def hybrid_search(self, query: str, top_k: int = 10, content_types: List[str] = None) -> List[NodeWithScore]:
        """Perform hybrid search across text, tables, and images."""
        if not self.index:
            raise RuntimeError("Index not built. Call build_index() first.")
        
        # Default to search all content types
        if content_types is None:
            content_types = ["text", "table", "image", "figure_caption"]
        
        # Perform vector search
        retriever = self.index.as_retriever(similarity_top_k=top_k * 2)  # Get more for re-ranking
        nodes = retriever.retrieve(query)
        
        # Filter by content type if specified
        if content_types:
            filtered_nodes = []
            for node in nodes:
                content_type = node.metadata.get("content_type", "text")
                if content_type in content_types or content_type == "text":
                    filtered_nodes.append(node)
            nodes = filtered_nodes[:top_k]
        
        # Apply re-ranking if available
        if self.reranker and len(nodes) > 1:
            logger.info("🎯 Applying re-ranking...")
            try:
                # Prepare query-document pairs for re-ranking
                pairs = [(query, node.text) for node in nodes]
                scores = self.reranker.predict(pairs)
                
                # Sort by re-ranking scores
                scored_nodes = list(zip(nodes, scores))
                scored_nodes.sort(key=lambda x: x[1], reverse=True)
                nodes = [node for node, score in scored_nodes[:top_k]]
                
            except Exception as e:
                logger.warning(f"Re-ranking failed: {e}")
        
        return nodes[:top_k]
    
    def backup_storage(self, storage_dir: str, backup_name: str = None):
        """Create a backup of the storage directory."""
        if not os.path.exists(storage_dir):
            logger.warning(f"Storage directory {storage_dir} does not exist")
            return None
            
        if backup_name is None:
            backup_name = f"rag_backup_{int(time.time())}"
        
        backup_path = f"/workspace/{backup_name}.tar.gz"
        
        try:
            with tarfile.open(backup_path, "w:gz") as tar:
                tar.add(storage_dir, arcname=os.path.basename(storage_dir))
            
            logger.info(f"✅ Backup created: {backup_path}")
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            return None
    
    def restore_storage(self, backup_path: str, storage_dir: str):
        """Restore storage from backup."""
        try:
            with tarfile.open(backup_path, "r:gz") as tar:
                tar.extractall(os.path.dirname(storage_dir))
            logger.info(f"✅ Storage restored from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore backup: {e}")
            return False


def main():
    """Main function to build the RAG index with non-text content support."""
    
    # Initialize pipeline
    pipeline = TechnicalRAGPipeline()
    
    # Build or load index (set use_qdrant=True for Qdrant storage)
    use_qdrant = os.getenv("USE_QDRANT", "false").lower() == "true"
    index = pipeline.build_index(use_qdrant=use_qdrant)
    
    print("\n" + "="*60)
    print("✅ INGESTION COMPLETED SUCCESSFULLY")
    print("="*60)
    if use_qdrant:
        print("🗄️ Index saved to: Qdrant")
    else:
        print("📁 Index saved to: storage/")
    print("🔍 Use query.py to search the documents")
    print("📊 Non-text content extracted to: extracted_content/")
    print("="*60)


if __name__ == "__main__":
    main()