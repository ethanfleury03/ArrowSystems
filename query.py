"""
Elite RAG Query Interface with Hybrid Search & Structured Responses
Implements the complete RAG orchestration pipeline
"""

import os
import logging
from typing import List, Optional, Dict, Any
from orchestrator import RAGOrchestrator, StructuredResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EliteRAGQuery:
    """
    Elite RAG query interface with hybrid search, query orchestration,
    and structured response generation.
    """
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub"):
        self.cache_dir = cache_dir
        self.orchestrator = RAGOrchestrator(cache_dir=cache_dir)
        
    def initialize(self, storage_dir="/workspace/storage"):
        """Initialize models and load index."""
        self.orchestrator.initialize_models()
        self.orchestrator.load_index(storage_dir=storage_dir)
        logger.info("✅ Elite RAG system initialized")
    
    def query(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True
    ) -> StructuredResponse:
        """
        Execute elite RAG query with full orchestration.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            alpha: Hybrid search weight (0.5 = equal dense/BM25, 1.0 = dense only)
            metadata_filters: Optional metadata filters
            dynamic_windowing: Enable dynamic context windowing
        
        Returns:
            StructuredResponse with answer, reasoning, and sources
        """
        return self.orchestrator.orchestrate_query(
            query=query,
            top_k=top_k,
            alpha=alpha,
            metadata_filters=metadata_filters,
            dynamic_windowing=dynamic_windowing
        )
    
    def format_response(self, response: StructuredResponse) -> str:
        """Format structured response for display."""
        return self.orchestrator.format_response(response)


# Legacy compatibility wrapper
class TechnicalRAGQuery(EliteRAGQuery):
    """Legacy wrapper for backward compatibility."""
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub"):
        super().__init__(cache_dir=cache_dir)
        self.embed_model = None
        self.reranker = None
        self.llm = None
        self.index = None
    
    def initialize_models(self):
        """Legacy method - redirects to new initialize."""
        self.orchestrator.initialize_models()
        self.embed_model = self.orchestrator.embed_model
        self.reranker = self.orchestrator.reranker
    
    def load_index(self, storage_dir="/workspace/storage"):
        """Legacy method - redirects to new load."""
        self.orchestrator.load_index(storage_dir=storage_dir)
        self.index = self.orchestrator.index
        return self.index
    
    def search(self, query: str, top_k: int = 10, use_reranking: bool = True):
        """Legacy search method - uses new orchestrator."""
        response = self.query(
            query=query,
            top_k=top_k,
            alpha=0.7 if use_reranking else 1.0  # More weight on dense if reranking
        )
        # Return nodes for backward compatibility
        return response

def main():
    """Main function to run elite RAG queries."""
    
    # Initialize elite RAG system
    rag_system = EliteRAGQuery()
    
    print("\n" + "="*80)
    print("🧠 ELITE RAG ORCHESTRATOR - DuraFlex Technical Assistant")
    print("="*80)
    print("Hybrid Search: Dense Embeddings (BAAI/bge-large-en-v1.5) + BM25 Keyword Search")
    print("Features: Query Rewriting | Intent Classification | Dynamic Windowing | Citations")
    print("="*80)
    print()
    
    # Initialize
    print("Initializing models and index...")
    rag_system.initialize()
    
    print("\n✅ System ready!")
    print("Ask questions about DuraFlex printer systems, troubleshooting, setup, and maintenance.")
    print("Type 'quit' or 'exit' to stop.\n")
    print("Advanced options:")
    print("  - Prefix with 'alpha:0.3' to adjust hybrid search weight (0=BM25 only, 1=dense only)")
    print("  - Prefix with 'top:20' to retrieve more chunks")
    print("  Example: 'alpha:0.3 top:15 how to troubleshoot print quality?'\n")
    
    while True:
        try:
            # Get user input
            user_input = input("❓ Your question: ").strip()
            
            # Check for exit commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye! Thanks for using the Elite RAG Orchestrator.")
                break
            
            # Skip empty queries
            if not user_input:
                print("Please enter a question.\n")
                continue
            
            # Parse advanced options
            alpha = 0.5  # Default: equal weight
            top_k = 10
            query_text = user_input
            
            # Check for alpha parameter
            if 'alpha:' in user_input:
                parts = user_input.split()
                for i, part in enumerate(parts):
                    if part.startswith('alpha:'):
                        try:
                            alpha = float(part.split(':')[1])
                            alpha = max(0.0, min(1.0, alpha))  # Clamp to [0, 1]
                            parts.pop(i)
                            break
                        except:
                            pass
                query_text = ' '.join(parts)
            
            # Check for top_k parameter
            if 'top:' in query_text:
                parts = query_text.split()
                for i, part in enumerate(parts):
                    if part.startswith('top:'):
                        try:
                            top_k = int(part.split(':')[1])
                            top_k = max(1, min(50, top_k))  # Clamp to [1, 50]
                            parts.pop(i)
                            break
                        except:
                            pass
                query_text = ' '.join(parts)
            
            print(f"\n🔍 Processing query (alpha={alpha}, top_k={top_k})...")
            print("-" * 80)
            
            # Execute query with orchestration
            response = rag_system.query(
                query=query_text,
                top_k=top_k,
                alpha=alpha,
                dynamic_windowing=True
            )
            
            # Format and display response
            formatted = rag_system.format_response(response)
            print(formatted)
            
            print()  # Add spacing for next question
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye! Thanks for using the Elite RAG Orchestrator.")
            break
        except Exception as e:
            logger.exception("Error processing query")
            print(f"\n❌ Error: {e}")
            print("Please try again.\n")


def demo_mode():
    """Run demo queries to showcase capabilities."""
    
    rag_system = EliteRAGQuery()
    print("Initializing Elite RAG Orchestrator...")
    rag_system.initialize()
    
    demo_queries = [
        "What is the DuraFlex printhead temperature range?",
        "How to troubleshoot print quality issues?",
        "Compare inline degasser vs standard degasser",
        "PPU installation procedure steps"
    ]
    
    print("\n" + "="*80)
    print("DEMO MODE - Running sample queries")
    print("="*80 + "\n")
    
    for query in demo_queries:
        print(f"\n{'='*80}")
        print(f"Demo Query: {query}")
        print('='*80)
        
        response = rag_system.query(query, top_k=5, alpha=0.5)
        formatted = rag_system.format_response(response)
        print(formatted)
        print()
        input("Press Enter for next demo query...")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        demo_mode()
    else:
        main()
