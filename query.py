"""
Query Interface for Technical RAG System
Separate file for searching the ingested documents
"""

import os
import logging
from typing import List
from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore
from sentence_transformers import CrossEncoder

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TechnicalRAGQuery:
    """Query interface for the technical RAG system."""
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub"):
        self.cache_dir = cache_dir
        self.embed_model = None
        self.reranker = None
        self.index = None
        
    def initialize_models(self):
        """Initialize embedding and re-ranking models."""
        logger.info("🚀 Initializing embedding model...")
        
        # Try multiple model options for better compatibility
        model_options = [
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
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-large",
                cache_folder=self.cache_dir
            )
            logger.info("✅ Re-ranker loaded successfully")
        except Exception as e:
            logger.warning(f"Re-ranker not available: {e}")
            self.reranker = None
        
        # Set global embedding model
        Settings.embed_model = self.embed_model
        logger.info("✅ Models initialized successfully")
    
    def load_index(self, storage_dir="storage"):
        """Load the existing index."""
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Index not found at {storage_dir}. Run ingest.py first.")
        
        logger.info("🔄 Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.index = load_index_from_storage(storage_context)
        logger.info("✅ Index loaded successfully")
        
        return self.index
    
    def search_with_reranking(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Perform search with re-ranking for maximum precision.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of scored nodes with re-ranking applied
        """
        logger.info(f"🔍 Searching for: {query}")
        
        # Use retriever instead of query engine to avoid LLM dependency
        retriever = self.index.as_retriever(similarity_top_k=top_k * 2)
        nodes = retriever.retrieve(query)
        
        if not nodes or not self.reranker:
            return nodes[:top_k]
        
        logger.info(f"🎯 Re-ranking {len(nodes)} results...")
        
        # Prepare query-document pairs for re-ranking
        query_doc_pairs = []
        for node in nodes:
            query_doc_pairs.append([query, node.text])
        
        # Get re-ranking scores
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
        # Sort by re-ranking scores
        reranked_nodes = sorted(
            zip(nodes, rerank_scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top-k re-ranked results
        return [node for node, _ in reranked_nodes[:top_k]]
    
    def search(self, query: str, top_k: int = 10, use_reranking: bool = True):
        """
        Main search function with optional re-ranking.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_reranking: Whether to use re-ranking for better precision
            
        Returns:
            Search results
        """
        if use_reranking:
            return self.search_with_reranking(query, top_k)
        else:
            # Simple search without re-ranking
            retriever = self.index.as_retriever(similarity_top_k=top_k)
            return retriever.retrieve(query)

def main():
    """Main function to run queries."""
    
    # Initialize query system
    query_system = TechnicalRAGQuery()
    
    # Initialize models
    query_system.initialize_models()
    
    # Load index
    query_system.load_index()
    
    # Test searches
    test_queries = [
        "DuraFlex printer troubleshooting",
        "printhead maintenance procedures", 
        "electrical connections setup",
        "software installation guide"
    ]
    
    print("\n" + "="*60)
    print("🔍 TESTING SEARCH FUNCTIONALITY")
    print("="*60)
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        print("-" * 50)
        
        # Search with re-ranking
        results = query_system.search(query, top_k=3, use_reranking=True)
        
        if isinstance(results, list):
            for i, result in enumerate(results, 1):
                print(f"\nResult {i}:")
                print(f"Score: {result.score:.3f}")
                print(f"Text: {result.text[:200]}...")
                if hasattr(result, 'metadata'):
                    print(f"Source: {result.metadata.get('file_name', 'Unknown')}")
        else:
            print(f"Response: {results}")
        
        print("\n" + "="*60)

if __name__ == "__main__":
    main()
