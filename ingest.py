"""
High-Performance RAG Pipeline for Technical Documents
Optimized for GPU rental with bge-large-en-v1.5 and re-ranking
Enhanced with non-text content extraction (tables, images, diagrams)
"""

import warnings
# Suppress annoying Pydantic warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import logging
import time
import json
import yaml
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
from llama_index.core.schema import NodeWithScore, TextNode, ImageNode
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
                        
                        # Create image info
                        image_info = {
                            "source_path": pdf_path,
                            "page_number": page_num + 1,
                            "image_index": img_idx,
                            "image_data": base64.b64encode(img_data).decode('utf-8'),
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
                "chunk_size": 350,
                "chunk_overlap": 88
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
        
        # Detect GPU
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"🖥️ Using device: {device}")
        if device == "cuda":
            logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        
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
            self.reranker = CrossEncoder(
                reranker_model,
                cache_folder=self.cache_dir,
                device=device
            )
            logger.info(f"✅ Re-ranker loaded successfully on {device}")
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
            node = TextNode(
                text=caption["caption_text"],
                metadata={
                    "content_type": "figure_caption",
                    "source_path": caption["source_path"],
                    "page_number": caption["page_number"],
                    "line_number": caption["line_number"]
                }
            )
            nodes.append(node)
        
        # Process images (create text nodes for captions and metadata)
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
    
    def build_index(self, data_dir="data", storage_dir="/workspace/storage", use_qdrant=False):
        """Build or load vector index with optimized chunking and non-text content."""
        
        # Initialize models
        self.initialize_models()
        
        # Setup storage context
        storage_context = None
        if use_qdrant:
            storage_context = self.setup_qdrant_storage()
        
        # Check if index already exists (only for local storage)
        if not use_qdrant and os.path.exists(storage_dir):
            logger.info("🔄 Loading existing index...")
            if storage_context is None:
                storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
            self.index = load_index_from_storage(storage_context)
            logger.info("✅ Index loaded successfully")
            return self.index
        
        print("\n" + "="*70)
        print("📥 BUILDING NEW RAG INDEX")
        print("="*70)
        
        # Step 1: Load Documents
        print("\n[Step 1/5] 📄 Loading PDF documents...")
        documents = SimpleDirectoryReader(data_dir).load_data()
        print(f"   ✅ Loaded {len(documents)} PDF documents")
        logger.info(f"Loaded {len(documents)} text documents")
        
        # Step 2: Extract Non-Text Content
        print("\n[Step 2/5] 🖼️  Extracting tables, images, and captions...")
        print("   This may take a few minutes...")
        tables, images, captions = self.process_non_text_content(data_dir)
        print(f"   ✅ Extracted {len(tables)} tables, {len(images)} images, {len(captions)} captions")
        
        # Step 3: Create Non-Text Nodes
        print("\n[Step 3/5] 📊 Creating searchable nodes from extracted content...")
        non_text_nodes = self.create_non_text_nodes(tables, images, captions)
        print(f"   ✅ Created {len(non_text_nodes)} non-text nodes")
        logger.info(f"Created {len(non_text_nodes)} non-text nodes")
        
        # Optimized text splitter for technical documents
        chunk_size = self.config.get("chunking", {}).get("chunk_size", 350)
        chunk_overlap = self.config.get("chunking", {}).get("chunk_overlap", 88)
        
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True
        )
        
        # Step 4: Create Vector Embeddings (LONGEST STEP)
        print("\n[Step 4/5] 🧠 Generating embeddings and building vector index...")
        print(f"   - Chunk size: {chunk_size} characters")
        print(f"   - Chunk overlap: {chunk_overlap} characters")
        print(f"   - Processing {len(documents)} documents...")
        print(f"   - This is the LONGEST step (embedding generation)")
        print(f"   - Expected time: 5-15 minutes on GPU, 30-60 minutes on CPU")
        print(f"   - Watch for progress below...")
        print("")
        
        # Record start time
        import time
        start_time = time.time()
        
        # Create index with text documents (with progress bar)
        if storage_context:
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=[text_splitter],
                storage_context=storage_context,
                show_progress=True  # Built-in LlamaIndex progress bar!
            )
        else:
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=[text_splitter],
                show_progress=True  # Built-in LlamaIndex progress bar!
            )
        
        elapsed = time.time() - start_time
        print(f"\n   ✅ Vector index created in {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
        print(f"   ⚡ Processing speed: {len(documents) / elapsed:.2f} docs/sec")
        
        # Step 5: Add Non-Text Nodes
        if non_text_nodes:
            print(f"\n[Step 5/5] 📎 Adding {len(non_text_nodes)} non-text items to index...")
            logger.info("📊 Adding non-text content to index...")
            
            # Progress bar for non-text nodes
            for node in tqdm(non_text_nodes, 
                           desc="   Adding items", 
                           unit="item",
                           ncols=80,
                           bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]'):
                self.index.insert_nodes([node])
            
            print(f"   ✅ Non-text content added to index")
        else:
            print(f"\n[Step 5/5] No non-text content to add")
        
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
        estimated_chunks = len(documents) * 10  # Rough estimate
        print("\n" + "="*70)
        print("✅ INGESTION COMPLETE!")
        print("="*70)
        print(f"📊 Documents processed: {len(documents)}")
        print(f"📊 Non-text items extracted: {len(non_text_nodes)}")
        print(f"📊 Estimated total chunks: ~{estimated_chunks}")
        print(f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        if not use_qdrant:
            print(f"📁 Storage location: {storage_dir}")
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