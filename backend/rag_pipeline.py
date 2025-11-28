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
from typing import List, Optional, Dict, Any, Tuple
from .orchestrator import RAGOrchestrator, StructuredResponse
from .logging_config import get_logger

logger = get_logger(__name__)


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
            enable_llm_evaluation=False,  # Disabled: Let LLM filter irrelevant chunks instead of pre-evaluating
            enable_llm_answers=True      # Enable LLM answer generation by default
        )
        self._initialized = False
        
    def initialize(self, storage_dir="latest_model") -> bool:
        """
        Initialize models and load index.
        
        Args:
            storage_dir: Directory containing the vector index
        
        Returns:
            True if initialization succeeded (both model and index loaded), False otherwise
        
        Raises:
            RuntimeError: If model loading fails (in production, this should abort startup)
        """
        if self._initialized:
            logger.info("rag_pipeline_already_initialized", storage_dir=storage_dir)
            return True
            
        logger.info("rag_pipeline_initializing", storage_dir=storage_dir)
        
        try:
            # Initialize models (model loading is always allowed, even on Cloud Run)
            # This will raise RuntimeError if models cannot be loaded from cache
            logger.info("rag_pipeline_initializing_models", storage_dir=storage_dir)
            self.orchestrator.initialize_models()
            logger.info("rag_pipeline_models_initialized", storage_dir=storage_dir)
            
            # Load index (will handle missing index gracefully if ingestion is disabled)
            logger.info("rag_pipeline_loading_index", storage_dir=storage_dir)
            self.orchestrator.load_index(storage_dir=storage_dir)
            
            # Check if index was actually loaded (might be None if ingestion disabled)
            if self.orchestrator.index is None:
                logger.warning("rag_pipeline_index_not_loaded", 
                             storage_dir=storage_dir,
                             message="Index is None after load_index() call. Pipeline will not be functional.")
                self._initialized = False
                return False
            
            # Verify index is a valid object
            if not hasattr(self.orchestrator.index, 'storage_context'):
                logger.error("rag_pipeline_index_invalid",
                           storage_dir=storage_dir,
                           index_type=type(self.orchestrator.index).__name__,
                           message="Index object is missing storage_context attribute - may be corrupted")
                self._initialized = False
                return False
            
            self._initialized = True
            logger.info("rag_pipeline_initialized", 
                       storage_dir=storage_dir,
                       rag_enabled=True,
                       index_type=type(self.orchestrator.index).__name__,
                       message="RAG pipeline successfully initialized and ready for queries")
            return True
            
        except Exception as e:
            error_type = type(e).__name__
            error_message = str(e)
            
            logger.error("rag_pipeline_init_failed", 
                        storage_dir=storage_dir,
                        error=error_message,
                        error_type=error_type,
                        exc_info=True,
                        rag_enabled=False,
                        message=f"RAG pipeline initialization failed: {error_type}: {error_message}")
            
            self._initialized = False
            # Don't re-raise - let caller handle gracefully (non-RAG endpoints should still work)
            return False
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None,
        role: Optional[str] = None,  # User role (ADMIN, TECHNICIAN, CUSTOMER) for machine-based filtering
        user_machine_models: Optional[List[str]] = None,  # Machine models for document-level filtering
        machine_confirmation: bool = False  # Whether user has confirmed their machine list
    ) -> StructuredResponse:
        """
        Execute RAG query with full orchestration.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            alpha: Hybrid search weight (0.5 = equal dense/BM25, 1.0 = dense only)
            metadata_filters: Optional metadata filters
            dynamic_windowing: Enable dynamic context windowing
            role: User role (ADMIN, TECHNICIAN, CUSTOMER) for machine-based filtering
            user_machine_models: List of machine models for document-level filtering
        
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
            dynamic_windowing=dynamic_windowing,
            chat_history=chat_history,
            role=role,
            user_machine_models=user_machine_models,
            machine_confirmation=machine_confirmation
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


def initialize_rag_pipeline(storage_dir="latest_model", cache_dir="/root/.cache/huggingface/hub", db_manager=None) -> Tuple[RAGPipeline, bool]:
    """
    Initialize and return RAG pipeline instance.
    
    This is a convenience function that creates and initializes
    the RAG pipeline in one call.
    
    Args:
        storage_dir: Directory containing the vector index
        cache_dir: HuggingFace cache directory
        db_manager: Optional database manager
        
    Returns:
        Tuple of (RAGPipeline instance, success: bool)
        - success is True if both model and index loaded successfully
        - success is False if initialization failed (e.g., index missing)
    """
    pipeline = get_rag_pipeline(cache_dir=cache_dir, db_manager=db_manager)
    success = pipeline.initialize(storage_dir=storage_dir)
    return pipeline, success


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
