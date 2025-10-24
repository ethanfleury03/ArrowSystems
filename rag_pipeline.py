"""
RAG Pipeline Module - Core RAG Logic for Production Architecture
Extracted from orchestrator.py and query.py for reuse across Streamlit and FastAPI

This module contains the core RAG functionality that can be used by:
- Streamlit application (existing)
- FastAPI backend (new)
- Any other frontend interface

Version: 1.0.0
Author: Arrow Systems Inc
"""

import warnings
# Suppress annoying Pydantic warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import logging
from typing import List, Optional, Dict, Any
from orchestrator import RAGOrchestrator, StructuredResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Core RAG Pipeline - Reusable RAG logic for production architecture.
    
    This class encapsulates the core RAG functionality that can be used
    by both Streamlit and FastAPI applications.
    """
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub", db_manager=None):
        """
        Initialize RAG Pipeline.
        
        Args:
            cache_dir: HuggingFace cache directory
            db_manager: Optional database manager for validated Q&A fast-path
        """
        self.cache_dir = cache_dir
        self.db_manager = db_manager
        self.orchestrator = RAGOrchestrator(
            cache_dir=cache_dir, 
            db_manager=db_manager,
            enable_llm_evaluation=True,  # Enable LLM evaluation by default
            enable_llm_answers=True      # Enable LLM answer generation by default
        )
        self._initialized = False
        
    def initialize(self, storage_dir="latest_model"):
        """
        Initialize models and load index.
        
        Args:
            storage_dir: Directory containing the vector index
        """
        if self._initialized:
            logger.info("RAG Pipeline already initialized")
            return
            
        logger.info("🚀 Initializing RAG Pipeline...")
        
        # Initialize models
        self.orchestrator.initialize_models()
        
        # Load index
        self.orchestrator.load_index(storage_dir=storage_dir)
        
        self._initialized = True
        logger.info("✅ RAG Pipeline initialized successfully")
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True
    ) -> StructuredResponse:
        """
        Execute RAG query with full orchestration.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            alpha: Hybrid search weight (0.5 = equal dense/BM25, 1.0 = dense only)
            metadata_filters: Optional metadata filters
            dynamic_windowing: Enable dynamic context windowing
        
        Returns:
            StructuredResponse with answer, reasoning, and sources
        """
        if not self._initialized:
            raise RuntimeError("RAG Pipeline not initialized. Call initialize() first.")
        
        return self.orchestrator.orchestrate_query(
            query=query,
            top_k=top_k,
            alpha=alpha,
            metadata_filters=metadata_filters,
            dynamic_windowing=dynamic_windowing
        )
    
    def format_response(self, response: StructuredResponse) -> str:
        """
        Format structured response for display.
        
        Args:
            response: StructuredResponse object
            
        Returns:
            Formatted string for display
        """
        return self.orchestrator.format_response(response)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        if not self._initialized:
            return {"error": "Pipeline not initialized"}
        
        stats = {
            "query_cache": self.orchestrator.cache.stats(),
            "semantic_cache": {
                "enabled": self.orchestrator.semantic_cache is not None,
                "size": len(self.orchestrator.semantic_cache.entries) if self.orchestrator.semantic_cache else 0
            },
            "document_evaluator": self.orchestrator.document_evaluator.get_cache_stats() if self.orchestrator.document_evaluator else {"enabled": False},
            "answer_generator": self.orchestrator.answer_generator.get_cache_stats() if self.orchestrator.answer_generator else {"enabled": False}
        }
        
        return stats
    
    def clear_caches(self):
        """
        Clear all caches.
        """
        if not self._initialized:
            logger.warning("Pipeline not initialized, cannot clear caches")
            return
        
        # Clear query cache
        self.orchestrator.cache = type(self.orchestrator.cache)(max_size=1000)
        
        # Clear semantic cache
        if self.orchestrator.semantic_cache:
            self.orchestrator.semantic_cache.entries.clear()
        
        # Clear LLM caches
        if self.orchestrator.document_evaluator:
            self.orchestrator.document_evaluator.clear_cache()
        
        if self.orchestrator.answer_generator:
            self.orchestrator.answer_generator.clear_cache()
        
        logger.info("✅ All caches cleared")
    
    def is_initialized(self) -> bool:
        """
        Check if pipeline is initialized.
        
        Returns:
            True if initialized, False otherwise
        """
        return self._initialized


# Global pipeline instance for reuse
_pipeline_instance = None


def get_rag_pipeline(cache_dir="/root/.cache/huggingface/hub", db_manager=None) -> RAGPipeline:
    """
    Get or create global RAG pipeline instance.
    
    This function provides a singleton pattern for the RAG pipeline,
    ensuring that expensive model loading only happens once.
    
    Args:
        cache_dir: HuggingFace cache directory
        db_manager: Optional database manager
        
    Returns:
        RAGPipeline instance
    """
    global _pipeline_instance
    
    if _pipeline_instance is None:
        _pipeline_instance = RAGPipeline(cache_dir=cache_dir, db_manager=db_manager)
        logger.info("🔄 Created new RAG pipeline instance")
    
    return _pipeline_instance


def initialize_rag_pipeline(storage_dir="latest_model", cache_dir="/root/.cache/huggingface/hub", db_manager=None) -> RAGPipeline:
    """
    Initialize and return RAG pipeline instance.
    
    This is a convenience function that creates and initializes
    the RAG pipeline in one call.
    
    Args:
        storage_dir: Directory containing the vector index
        cache_dir: HuggingFace cache directory
        db_manager: Optional database manager
        
    Returns:
        Initialized RAGPipeline instance
    """
    pipeline = get_rag_pipeline(cache_dir=cache_dir, db_manager=db_manager)
    pipeline.initialize(storage_dir=storage_dir)
    return pipeline


# Legacy compatibility functions for existing code
def create_elite_rag_query(cache_dir="/root/.cache/huggingface/hub", db_manager=None):
    """
    Legacy compatibility function for existing code.
    
    This function maintains backward compatibility with the existing
    EliteRAGQuery class while using the new RAGPipeline.
    """
    return get_rag_pipeline(cache_dir=cache_dir, db_manager=db_manager)


if __name__ == "__main__":
    """
    Test the RAG pipeline standalone.
    """
    print("🧪 Testing RAG Pipeline...")
    
    # Initialize pipeline
    pipeline = initialize_rag_pipeline()
    
    # Test query
    test_query = "What is the DuraFlex printhead temperature range?"
    print(f"\n🔍 Test Query: {test_query}")
    
    response = pipeline.query(test_query, top_k=5)
    
    print("\n📋 Response:")
    print(pipeline.format_response(response))
    
    # Show cache stats
    print("\n📊 Cache Statistics:")
    stats = pipeline.get_cache_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
