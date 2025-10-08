"""
Elite RAG Orchestrator with Hybrid Search (Dense + BM25 + Metadata)
Implements query rewriting, intent classification, and structured response generation
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class QueryIntent:
    """Classified query intent with metadata."""
    intent_type: str  # definition, lookup, reasoning, comparison, troubleshooting
    confidence: float
    keywords: List[str]
    requires_subqueries: bool
    temporal_context: Optional[str] = None


@dataclass
class RetrievalContext:
    """Retrieved context with metadata and scoring."""
    nodes: List[NodeWithScore]
    source_ids: Dict[str, str]  # Maps node_id to source identifier like [1], [2]
    relevance_scores: Dict[str, float]
    metadata_priority: Dict[str, float]
    total_chunks: int


@dataclass
class StructuredResponse:
    """Structured RAG response with citations."""
    query: str
    answer: str
    reasoning: str
    sources: List[Dict[str, Any]]
    confidence: float
    intent: QueryIntent


class QueryRewriter:
    """Handles query cleaning, expansion, and reformulation."""
    
    def __init__(self):
        # Common acronyms in technical documentation
        self.acronym_map = {
            'ppu': 'printhead power unit',
            'cli': 'command line interface',
            'pdf': 'portable document format',
            'api': 'application programming interface',
            'gui': 'graphical user interface',
            'cpu': 'central processing unit',
            'ram': 'random access memory',
            'usb': 'universal serial bus',
            'ip': 'internet protocol',
            'tcp': 'transmission control protocol',
            'http': 'hypertext transfer protocol',
            'dpi': 'dots per inch',
            'rpm': 'revolutions per minute',
            'psi': 'pounds per square inch',
        }
    
    def clean_query(self, query: str) -> str:
        """Clean and normalize query."""
        # Remove extra whitespace
        query = ' '.join(query.split())
        
        # Fix common typos (simple version)
        query = query.replace('pritner', 'printer')
        query = query.replace('printeer', 'printer')
        query = query.replace('temprature', 'temperature')
        query = query.replace('seperator', 'separator')
        
        return query.strip()
    
    def expand_acronyms(self, query: str) -> str:
        """Expand known acronyms."""
        words = query.split()
        expanded = []
        
        for word in words:
            word_lower = word.lower().strip('.,!?;:')
            if word_lower in self.acronym_map:
                # Add both acronym and expansion
                expanded.append(f"{word} ({self.acronym_map[word_lower]})")
            else:
                expanded.append(word)
        
        return ' '.join(expanded)
    
    def extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query."""
        # Remove stop words
        stop_words = {'what', 'how', 'why', 'where', 'when', 'who', 'is', 'are', 'the', 'a', 'an', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        
        return keywords
    
    def rewrite_query(self, query: str, intent: QueryIntent) -> List[str]:
        """Generate query variations based on intent."""
        variations = [query]
        
        # Clean and expand
        cleaned = self.clean_query(query)
        expanded = self.expand_acronyms(cleaned)
        
        if expanded != query:
            variations.append(expanded)
        
        # Intent-specific rewrites
        if intent.intent_type == 'troubleshooting':
            variations.append(f"error {query}")
            variations.append(f"fix {query}")
            variations.append(f"solve {query}")
        
        elif intent.intent_type == 'definition':
            variations.append(f"what is {query}")
            variations.append(f"{query} definition")
        
        elif intent.intent_type == 'comparison':
            variations.append(f"{query} differences")
            variations.append(f"compare {query}")
        
        return list(set(variations))  # Remove duplicates


class IntentClassifier:
    """Classify query intent for optimal retrieval."""
    
    def classify(self, query: str) -> QueryIntent:
        """Classify query intent."""
        query_lower = query.lower()
        
        # Pattern matching for intent classification
        if any(word in query_lower for word in ['what is', 'define', 'definition', 'meaning of', 'explain']):
            intent_type = 'definition'
            confidence = 0.9
        
        elif any(word in query_lower for word in ['error', 'fix', 'troubleshoot', 'not working', 'issue', 'problem', 'failed']):
            intent_type = 'troubleshooting'
            confidence = 0.85
        
        elif any(word in query_lower for word in ['compare', 'difference', 'vs', 'versus', 'better', 'which']):
            intent_type = 'comparison'
            confidence = 0.8
            requires_subqueries = True
        
        elif any(word in query_lower for word in ['how to', 'steps', 'procedure', 'process', 'install', 'configure']):
            intent_type = 'reasoning'
            confidence = 0.85
            requires_subqueries = True
        
        elif any(word in query_lower for word in ['how many', 'how much', 'temperature', 'pressure', 'voltage', 'speed']):
            intent_type = 'lookup'
            confidence = 0.9
        
        else:
            intent_type = 'lookup'
            confidence = 0.6
        
        # Extract keywords
        rewriter = QueryRewriter()
        keywords = rewriter.extract_keywords(query)
        
        # Check if requires subqueries
        requires_subqueries = intent_type in ['comparison', 'reasoning'] or len(keywords) > 5
        
        return QueryIntent(
            intent_type=intent_type,
            confidence=confidence,
            keywords=keywords,
            requires_subqueries=requires_subqueries
        )


class HybridRetriever:
    """Combines dense embeddings, BM25, and metadata filtering."""
    
    def __init__(self, index, embed_model, reranker=None):
        self.index = index
        self.embed_model = embed_model
        self.reranker = reranker
        self.bm25 = None
        self.corpus_nodes = []
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 index from document corpus."""
        try:
            # Get all nodes from the index
            logger.info("🔧 Initializing BM25 index...")
            
            # Retrieve a large set of documents to build BM25 corpus
            retriever = self.index.as_retriever(similarity_top_k=1000)
            # Use a generic query to get diverse documents
            dummy_nodes = retriever.retrieve("technical documentation system")
            
            if dummy_nodes:
                self.corpus_nodes = dummy_nodes
                
                # Tokenize corpus for BM25
                tokenized_corpus = [node.text.lower().split() for node in self.corpus_nodes]
                self.bm25 = BM25Okapi(tokenized_corpus)
                
                logger.info(f"✅ BM25 initialized with {len(self.corpus_nodes)} documents")
            else:
                logger.warning("⚠️ No documents found for BM25 initialization")
        
        except Exception as e:
            logger.warning(f"BM25 initialization failed: {e}")
            self.bm25 = None
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[NodeWithScore, float]]:
        """Perform BM25 keyword search."""
        if not self.bm25 or not self.corpus_nodes:
            return []
        
        # Tokenize query
        tokenized_query = query.lower().split()
        
        # Get BM25 scores
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return nodes with scores
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                results.append((self.corpus_nodes[idx], float(scores[idx])))
        
        return results
    
    def dense_search(self, query: str, top_k: int = 20) -> List[NodeWithScore]:
        """Perform dense embedding search."""
        retriever = self.index.as_retriever(similarity_top_k=top_k)
        return retriever.retrieve(query)
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None
    ) -> List[NodeWithScore]:
        """
        Perform hybrid search combining BM25 and dense embeddings.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for BM25). 0.5 = equal weight
            metadata_filters: Optional metadata filters
        
        Returns:
            Ranked list of nodes
        """
        # Get dense results
        dense_results = self.dense_search(query, top_k=top_k * 2)
        
        # Get BM25 results
        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        
        # Combine results with scoring
        combined_scores = defaultdict(lambda: {'dense': 0.0, 'bm25': 0.0, 'node': None})
        
        # Normalize dense scores
        if dense_results:
            max_dense = max(node.score for node in dense_results) if dense_results else 1.0
            for node in dense_results:
                node_id = node.node_id
                combined_scores[node_id]['dense'] = node.score / max_dense
                combined_scores[node_id]['node'] = node
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            for node, score in bm25_results:
                node_id = node.node_id
                combined_scores[node_id]['bm25'] = score / max_bm25
                if combined_scores[node_id]['node'] is None:
                    combined_scores[node_id]['node'] = node
        
        # Calculate hybrid scores
        hybrid_results = []
        for node_id, scores in combined_scores.items():
            if scores['node'] is not None:
                hybrid_score = alpha * scores['dense'] + (1 - alpha) * scores['bm25']
                
                # Apply metadata filtering and boosting
                if metadata_filters:
                    node = scores['node']
                    if not self._matches_filters(node, metadata_filters):
                        continue
                
                # Create new NodeWithScore with hybrid score
                scored_node = NodeWithScore(
                    node=scores['node'].node,
                    score=hybrid_score
                )
                hybrid_results.append(scored_node)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply re-ranking if available
        if self.reranker and len(hybrid_results) > 1:
            hybrid_results = self._rerank(query, hybrid_results[:top_k * 2])
        
        return hybrid_results[:top_k]
    
    def _matches_filters(self, node: NodeWithScore, filters: Dict[str, Any]) -> bool:
        """Check if node matches metadata filters."""
        for key, value in filters.items():
            node_value = node.metadata.get(key)
            if node_value != value:
                return False
        return True
    
    def _rerank(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """Apply cross-encoder re-ranking."""
        try:
            pairs = [(query, node.text) for node in nodes]
            scores = self.reranker.predict(pairs)
            
            # Update scores and sort
            for node, score in zip(nodes, scores):
                node.score = float(score)
            
            nodes.sort(key=lambda x: x.score, reverse=True)
            return nodes
        
        except Exception as e:
            logger.warning(f"Re-ranking failed: {e}")
            return nodes


class ResponseGenerator:
    """Generate structured responses with citations."""
    
    def __init__(self):
        self.source_counter = 1
    
    def generate_structured_response(
        self,
        query: str,
        context: RetrievalContext,
        intent: QueryIntent
    ) -> StructuredResponse:
        """Generate structured response with answer, reasoning, and sources."""
        
        # Reset source counter
        self.source_counter = 1
        
        # Build answer from context
        answer = self._build_answer(query, context, intent)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(context, intent)
        
        # Compile sources
        sources = self._compile_sources(context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(context, intent)
        
        return StructuredResponse(
            query=query,
            answer=answer,
            reasoning=reasoning,
            sources=sources,
            confidence=confidence,
            intent=intent
        )
    
    def _build_answer(
        self,
        query: str,
        context: RetrievalContext,
        intent: QueryIntent
    ) -> str:
        """Build answer from retrieved context."""
        
        if not context.nodes:
            return "The provided context does not include information to answer this query."
        
        # Group nodes by source document
        source_groups = defaultdict(list)
        for node in context.nodes:
            source_name = node.metadata.get('file_name', 'Unknown')
            source_groups[source_name].append(node)
        
        # Build answer sections
        answer_parts = []
        
        for source_name, nodes in source_groups.items():
            # Get source ID for citation
            source_id = None
            for node in nodes:
                if node.node_id in context.source_ids:
                    source_id = context.source_ids[node.node_id]
                    break
            
            if not source_id:
                continue
            
            # Combine relevant text from this source
            text_parts = []
            for node in nodes[:3]:  # Limit to top 3 chunks per source
                text_parts.append(node.text.strip())
            
            combined_text = ' '.join(text_parts)
            
            # Add to answer with citation
            answer_parts.append(f"According to {source_name} {source_id}:\n{combined_text}")
        
        if not answer_parts:
            return "The provided context does not include sufficient information to answer this query."
        
        return '\n\n'.join(answer_parts)
    
    def _generate_reasoning(
        self,
        context: RetrievalContext,
        intent: QueryIntent
    ) -> str:
        """Generate reasoning summary."""
        
        if not context.nodes:
            return "No relevant documents were retrieved for this query."
        
        reasoning_parts = [
            f"Retrieved {context.total_chunks} relevant document chunks using hybrid search (dense embeddings + BM25).",
            f"Query intent classified as: {intent.intent_type} (confidence: {intent.confidence:.2%})."
        ]
        
        # Add metadata priority info
        if context.metadata_priority:
            high_priority = [k for k, v in context.metadata_priority.items() if v > 0.8]
            if high_priority:
                reasoning_parts.append(f"Prioritized {len(high_priority)} sources based on reliability and recency.")
        
        # Add relevance info
        if context.relevance_scores:
            avg_score = np.mean(list(context.relevance_scores.values()))
            reasoning_parts.append(f"Average relevance score: {avg_score:.3f}")
        
        return ' '.join(reasoning_parts)
    
    def _compile_sources(self, context: RetrievalContext) -> List[Dict[str, Any]]:
        """Compile source summary."""
        
        sources = []
        source_docs = {}
        
        for node in context.nodes:
            source_name = node.metadata.get('file_name', 'Unknown')
            page_num = node.metadata.get('page_label', 'N/A')
            
            if source_name not in source_docs:
                source_id = context.source_ids.get(node.node_id, f"[{len(source_docs) + 1}]")
                source_docs[source_name] = {
                    'id': source_id,
                    'name': source_name,
                    'pages': set(),
                    'content_type': node.metadata.get('content_type', 'text')
                }
            
            if page_num != 'N/A':
                source_docs[source_name]['pages'].add(str(page_num))
        
        # Convert to list
        for source_info in source_docs.values():
            pages = sorted(list(source_info['pages']), key=lambda x: int(x) if x.isdigit() else 0)
            sources.append({
                'id': source_info['id'],
                'name': source_info['name'],
                'pages': ', '.join(pages) if pages else 'N/A',
                'content_type': source_info['content_type']
            })
        
        return sources
    
    def _calculate_confidence(
        self,
        context: RetrievalContext,
        intent: QueryIntent
    ) -> float:
        """Calculate response confidence."""
        
        if not context.nodes:
            return 0.0
        
        # Factors: relevance scores, intent confidence, number of sources
        avg_relevance = np.mean([node.score for node in context.nodes])
        num_sources = len(set(node.metadata.get('file_name', '') for node in context.nodes))
        
        confidence = (
            0.5 * avg_relevance +
            0.3 * intent.confidence +
            0.2 * min(num_sources / 3, 1.0)  # Up to 3 sources
        )
        
        return min(confidence, 1.0)


class RAGOrchestrator:
    """
    Elite RAG orchestrator implementing hybrid search, query rewriting,
    and structured response generation.
    """
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub"):
        self.cache_dir = cache_dir
        self.embed_model = None
        self.reranker = None
        self.index = None
        self.retriever = None
        
        # Components
        self.query_rewriter = QueryRewriter()
        self.intent_classifier = IntentClassifier()
        self.response_generator = ResponseGenerator()
    
    def initialize_models(self):
        """Initialize embedding and re-ranking models."""
        logger.info("🚀 Initializing models for RAG orchestrator...")
        
        # Embedding model options (without sentence-transformers/ prefix)
        model_options = [
            ("BAAI/bge-large-en-v1.5", "BGE Large"),
            ("BAAI/bge-base-en-v1.5", "BGE Base"),
            ("all-MiniLM-L6-v2", "MiniLM"),
            ("all-mpnet-base-v2", "MPNet")
        ]
        
        for model_name, display_name in model_options:
            try:
                logger.info(f"Loading embedding model: {display_name} ({model_name})")
                
                import os
                self.embed_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    cache_folder=self.cache_dir,
                    trust_remote_code=True,
                    device="cuda" if os.path.exists("/dev/nvidia0") else "cpu"
                )
                logger.info(f"✅ Embedding model loaded: {display_name}")
                break
            except Exception as e:
                logger.warning(f"Failed to load {display_name}: {str(e)[:100]}")
                # Try with sentence-transformers prefix if not already
                if not model_name.startswith("sentence-transformers/"):
                    try:
                        full_name = f"sentence-transformers/{model_name}"
                        self.embed_model = HuggingFaceEmbedding(
                            model_name=full_name,
                            cache_folder=self.cache_dir,
                            trust_remote_code=True
                        )
                        logger.info(f"✅ Embedding model loaded: {display_name} (with prefix)")
                        break
                    except:
                        continue
        
        if not self.embed_model:
            # Emergency fallback
            try:
                self.embed_model = HuggingFaceEmbedding(model_name="all-MiniLM-L6-v2")
                logger.info("✅ Loaded with emergency fallback")
            except:
                raise RuntimeError("Could not load embedding model. Check internet connection.")
        
        # Re-ranker model
        try:
            logger.info("Loading re-ranker model...")
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-large",
                cache_folder=self.cache_dir
            )
            logger.info("✅ Re-ranker loaded")
        except Exception as e:
            logger.warning(f"Re-ranker not available: {e}")
            self.reranker = None
        
        # Set global settings
        Settings.embed_model = self.embed_model
        logger.info("✅ Models initialized successfully")
    
    def load_index(self, storage_dir="/workspace/storage"):
        """Load existing index."""
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(f"Index not found at {storage_dir}")
        
        logger.info("🔄 Loading index...")
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.index = load_index_from_storage(storage_context)
        
        # Initialize hybrid retriever
        self.retriever = HybridRetriever(
            index=self.index,
            embed_model=self.embed_model,
            reranker=self.reranker
        )
        
        logger.info("✅ Index and retriever initialized")
    
    def orchestrate_query(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True
    ) -> StructuredResponse:
        """
        Main orchestration method - handles complete RAG pipeline.
        
        Args:
            query: User query
            top_k: Number of chunks to retrieve
            alpha: Weight for dense vs BM25 (0.5 = equal)
            metadata_filters: Optional metadata filters
            dynamic_windowing: Enable dynamic context windowing
        
        Returns:
            StructuredResponse with answer, reasoning, and sources
        """
        
        logger.info(f"🎯 Orchestrating query: {query}")
        
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        logger.info(f"📋 Intent: {intent.intent_type} (confidence: {intent.confidence:.2%})")
        
        # Step 2: Rewrite query
        query_variations = self.query_rewriter.rewrite_query(query, intent)
        logger.info(f"🔄 Generated {len(query_variations)} query variations")
        
        # Step 3: Hybrid retrieval
        all_nodes = []
        for q_variant in query_variations[:3]:  # Use top 3 variations
            nodes = self.retriever.hybrid_search(
                query=q_variant,
                top_k=top_k,
                alpha=alpha,
                metadata_filters=metadata_filters
            )
            all_nodes.extend(nodes)
        
        # Deduplicate by node_id
        seen = set()
        unique_nodes = []
        for node in all_nodes:
            if node.node_id not in seen:
                seen.add(node.node_id)
                unique_nodes.append(node)
        
        # Sort by score and limit
        unique_nodes.sort(key=lambda x: x.score, reverse=True)
        
        # Step 4: Dynamic context windowing
        if dynamic_windowing:
            unique_nodes = self._apply_dynamic_windowing(unique_nodes, top_k)
        else:
            unique_nodes = unique_nodes[:top_k]
        
        logger.info(f"📚 Retrieved {len(unique_nodes)} unique chunks")
        
        # Step 5: Build retrieval context
        context = self._build_retrieval_context(unique_nodes)
        
        # Step 6: Generate structured response
        response = self.response_generator.generate_structured_response(
            query=query,
            context=context,
            intent=intent
        )
        
        logger.info(f"✅ Response generated (confidence: {response.confidence:.2%})")
        
        return response
    
    def _apply_dynamic_windowing(
        self,
        nodes: List[NodeWithScore],
        base_top_k: int
    ) -> List[NodeWithScore]:
        """Apply dynamic context windowing based on relevance scores."""
        
        if not nodes:
            return []
        
        # Calculate score threshold
        scores = [node.score for node in nodes]
        mean_score = np.mean(scores)
        std_score = np.std(scores) if len(scores) > 1 else 0
        
        threshold = mean_score - 0.5 * std_score
        
        # Include nodes above threshold, minimum base_top_k
        windowed_nodes = []
        for node in nodes:
            if node.score >= threshold or len(windowed_nodes) < base_top_k:
                windowed_nodes.append(node)
            
            # Cap at 2x base_top_k
            if len(windowed_nodes) >= base_top_k * 2:
                break
        
        logger.info(f"🪟 Dynamic windowing: {len(nodes)} → {len(windowed_nodes)} chunks (threshold: {threshold:.3f})")
        
        return windowed_nodes
    
    def _build_retrieval_context(self, nodes: List[NodeWithScore]) -> RetrievalContext:
        """Build retrieval context with metadata."""
        
        # Assign source IDs
        source_ids = {}
        source_counter = 1
        source_map = {}
        
        for node in nodes:
            source_name = node.metadata.get('file_name', 'Unknown')
            if source_name not in source_map:
                source_map[source_name] = f"[{source_counter}]"
                source_counter += 1
            source_ids[node.node_id] = source_map[source_name]
        
        # Calculate relevance scores
        relevance_scores = {node.node_id: node.score for node in nodes}
        
        # Calculate metadata priority (based on content type, recency, etc.)
        metadata_priority = {}
        for node in nodes:
            priority = 1.0
            
            # Boost tables and structured content
            content_type = node.metadata.get('content_type', 'text')
            if content_type == 'table':
                priority *= 1.2
            
            # Could add date-based boosting here if metadata has dates
            
            metadata_priority[node.node_id] = priority
        
        return RetrievalContext(
            nodes=nodes,
            source_ids=source_ids,
            relevance_scores=relevance_scores,
            metadata_priority=metadata_priority,
            total_chunks=len(nodes)
        )
    
    def format_response(self, response: StructuredResponse) -> str:
        """Format structured response for display."""
        
        output = []
        output.append("=" * 80)
        output.append("ANSWER:")
        output.append("=" * 80)
        output.append(response.answer)
        output.append("")
        
        output.append("=" * 80)
        output.append("REASONING SUMMARY:")
        output.append("=" * 80)
        output.append(response.reasoning)
        output.append("")
        
        output.append("=" * 80)
        output.append("SOURCE SUMMARY:")
        output.append("=" * 80)
        for source in response.sources:
            pages = f" (pages: {source['pages']})" if source['pages'] != 'N/A' else ""
            content_type = f" [{source['content_type']}]" if source['content_type'] != 'text' else ""
            output.append(f"{source['id']} {source['name']}{pages}{content_type}")
        
        output.append("")
        output.append(f"Confidence: {response.confidence:.2%} | Intent: {response.intent.intent_type}")
        output.append("=" * 80)
        
        return '\n'.join(output)

