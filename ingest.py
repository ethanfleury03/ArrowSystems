"""
High-Performance RAG Pipeline for Technical Documents
Optimized for GPU rental with bge-large-en-v1.5 and re-ranking
Enhanced with non-text content extraction (tables, images, diagrams)
"""

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

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage, Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore, TextNode, ImageNode
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from sentence_transformers import CrossEncoder
import qdrant_client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NonTextExtractor:
    """Extract and process non-text content from documents."""
    
    def __init__(self, output_dir="extracted_content"):
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
                        df = pd.DataFrame(table_data[1:], columns=table_data[0])
                        
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
                        img_rect = page.get_image_rects(xref)[0] if page.get_image_rects(xref) else None
                        
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
                            "bbox": img_rect.get_rect() if img_rect else None
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
        
        # Use config or fallback to multiple model options
        model_options = [
            self.config.get("models", {}).get("embedding", "BAAI/bge-large-en-v1.5"),
            "BAAI/bge-large-en",  # BGE-large (7B) model
            "BAAI/bge-large-en-v1.5",
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2"
        ]
        
        for model_name in model_options:
            try:
                logger.info(f"Trying model: {model_name}")
                self.embed_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    cache_folder=self.cache_dir
                )
                logger.info(f"✅ Successfully loaded: {model_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {e}")
                continue
        
        if not self.embed_model:
            raise RuntimeError("Could not load any embedding model")
        
        # Try to initialize re-ranker (optional)
        try:
            logger.info("🎯 Initializing re-ranker...")
            reranker_model = self.config.get("models", {}).get("reranker", "BAAI/bge-reranker-large")
            self.reranker = CrossEncoder(
                reranker_model,
                cache_folder=self.cache_dir
            )
            logger.info("✅ Re-ranker loaded successfully")
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
    
    def build_index(self, data_dir="data", storage_dir="storage", use_qdrant=False):
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
        
        logger.info("📥 Creating new index with optimized chunking and non-text content...")
        
        # Load text documents
        documents = SimpleDirectoryReader(data_dir).load_data()
        logger.info(f"Loaded {len(documents)} text documents")
        
        # Process non-text content
        tables, images, captions = self.process_non_text_content(data_dir)
        
        # Create non-text nodes
        non_text_nodes = self.create_non_text_nodes(tables, images, captions)
        logger.info(f"Created {len(non_text_nodes)} non-text nodes")
        
        # Optimized text splitter for technical documents
        chunk_size = self.config.get("chunking", {}).get("chunk_size", 350)
        chunk_overlap = self.config.get("chunking", {}).get("chunk_overlap", 88)
        
        text_splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            include_metadata=True
        )
        
        # Create index with text documents
        if storage_context:
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=[text_splitter],
                storage_context=storage_context
            )
        else:
            self.index = VectorStoreIndex.from_documents(
                documents,
                transformations=[text_splitter]
            )
        
        # Add non-text nodes to the index
        if non_text_nodes:
            logger.info("📊 Adding non-text content to index...")
            for node in non_text_nodes:
                self.index.insert_nodes([node])
        
        # Persist the index (only for local storage)
        if not use_qdrant:
            self.index.storage_context.persist(persist_dir=storage_dir)
            logger.info("✅ Index created and saved locally")
        else:
            logger.info("✅ Index created and saved to Qdrant")
        
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