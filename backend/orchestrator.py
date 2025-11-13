"""
Elite RAG Orchestrator with Hybrid Search (Dense + BM25 + Metadata)
Implements query rewriting, intent classification, and structured response generation
"""

import warnings
# Suppress annoying Pydantic warnings
warnings.filterwarnings("ignore", message=".*validate_default.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

from llama_index.core import StorageContext, load_index_from_storage, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.schema import NodeWithScore, TextNode
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
import json
import hashlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# QUERY CACHE - Only caches user-validated good answers
# ============================================================================

class QueryCache:
    """
    In-memory cache for RAG queries.
    Only caches responses that users marked as helpful (thumbs up 👍).
    """
    
    def __init__(self, max_size: int = 1000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
        logger.info(f"💾 QueryCache initialized (max_size: {max_size})")
    
    def _hash_query(self, query: str, top_k: int = 10, alpha: float = 0.5) -> str:
        """Create cache key from query parameters."""
        key = f"{query.lower().strip()}:{top_k}:{alpha}"
        return hashlib.md5(key.encode()).hexdigest()
    
    def get(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """Try to get cached response."""
        key = self._hash_query(query, top_k, alpha)
        
        if key in self.cache:
            self.hits += 1
            hit_rate = self.hits / (self.hits + self.misses) * 100
            logger.info(f"💾 ✅ CACHE HIT! Serving validated answer instantly")
            logger.info(f"   Cache stats: {self.hits} hits / {self.misses} misses ({hit_rate:.1f}% hit rate)")
            return self.cache[key]
        
        self.misses += 1
        total = self.hits + self.misses
        hit_rate = self.hits / total * 100 if total > 0 else 0
        logger.info(f"💾 ❌ Cache miss - will run full RAG")
        logger.info(f"   Cache stats: {self.hits} hits / {self.misses} misses ({hit_rate:.1f}% hit rate)")
        return None
    
    def set(self, query: str, response, top_k: int = 10, alpha: float = 0.5):
        """Cache a user-validated response."""
        key = self._hash_query(query, top_k, alpha)
        
        # LRU eviction if cache is full
        if len(self.cache) >= self.max_size:
            # Remove oldest entry (first in dict)
            oldest = next(iter(self.cache))
            del self.cache[oldest]
            logger.info(f"💾 Evicted oldest cache entry (cache was full)")
        
        self.cache[key] = response
        logger.info(f"💾 ✅ CACHED validated answer (cache size: {len(self.cache)}/{self.max_size})")
        logger.info(f"   This answer will be served instantly for future identical queries!")
    
    def remove(self, query: str, top_k: int = 10, alpha: float = 0.5):
        """Remove a cached response (e.g., if marked unhelpful later)."""
        key = self._hash_query(query, top_k, alpha)
        if key in self.cache:
            del self.cache[key]
            logger.info(f"💾 Removed query from cache")
            return True
        return False
    
    @property
    def hit_rate(self) -> float:
        """Get cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total * 100) if total > 0 else 0.0
    
    def stats(self) -> dict:
        """Get cache statistics."""
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hit_rate,
            'total_queries': self.hits + self.misses
        }


# ============================================================================
# SEMANTIC CACHE - Matches similar queries via embeddings (gated by 👍)
# ============================================================================

class SemanticCache:
    """
    Semantic cache that stores (embedding, query, response) and returns
    cached responses for semantically similar queries.
    """
    def __init__(self, embed_model, threshold: float = 0.95, max_size: int = 500):
        self.embed_model = embed_model
        self.threshold = threshold
        self.max_size = max_size
        self.entries = []  # list of dicts: {'emb': np.array, 'query': str, 'response': StructuredResponse}
        logger.info(f"💾 SemanticCache initialized (threshold={threshold}, max_size={max_size})")
    
    @staticmethod
    def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a is None or b is None:
            return 0.0
        an = a / (np.linalg.norm(a) + 1e-12)
        bn = b / (np.linalg.norm(b) + 1e-12)
        return float(np.dot(an, bn))
    
    def get(self, query: str):
        """Return cached response if similarity exceeds threshold."""
        try:
            q_emb = np.array(self.embed_model.get_text_embedding(query), dtype=np.float32)
        except Exception as e:
            logger.debug(f"SemanticCache embedding failed: {e}")
            return None
        
        best_score = 0.0
        best_resp = None
        for entry in self.entries:
            score = self._cosine_similarity(q_emb, entry['emb'])
            if score > best_score:
                best_score = score
                best_resp = entry['response']
        
        if best_score >= self.threshold and best_resp is not None:
            logger.info(f"💾 ✅ SEMANTIC CACHE HIT (similarity={best_score:.2f})")
            return best_resp
        else:
            logger.info(f"💾 ❌ Semantic cache miss (best={best_score:.2f} < {self.threshold})")
            return None
    
    def set(self, query: str, response):
        """Store a validated response with its embedding."""
        try:
            q_emb = np.array(self.embed_model.get_text_embedding(query), dtype=np.float32)
        except Exception as e:
            logger.debug(f"SemanticCache embedding failed on set: {e}")
            return
        # Evict oldest if full
        if len(self.entries) >= self.max_size:
            self.entries.pop(0)
        self.entries.append({'emb': q_emb, 'query': query, 'response': response})
        logger.info(f"💾 ✅ Added to semantic cache (size: {len(self.entries)}/{self.max_size})")
    
    def remove(self, query: str):
        """Remove entries that match the exact query text."""
        before = len(self.entries)
        self.entries = [e for e in self.entries if e['query'] != query]
        after = len(self.entries)
        if after < before:
            logger.info(f"💾 Removed {before-after} entry(ies) from semantic cache for query")


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
    matched_machine_name: Optional[str] = None  # Machine name matched in query (if >=95% similarity)
    token_input: Optional[int] = None  # Input tokens used
    token_output: Optional[int] = None  # Output tokens used
    token_total: Optional[int] = None  # Total tokens used
    cost_usd: Optional[float] = None  # Estimated cost in USD


class MachineNameMatcher:
    """
    Matches queries against canonical machine names and boosts retrieval from matched machine documents.
    Uses fuzzy string matching with >=95% similarity threshold.
    """
    
    def __init__(self, machine_names: Optional[List[str]] = None, similarity_threshold: float = 0.95):
        """
        Initialize machine name matcher.
        
        Args:
            machine_names: List of canonical machine names. If None, will auto-detect from filenames.
            similarity_threshold: Minimum similarity (0.0-1.0) to consider a match. Default: 0.95 (95%)
        """
        self.similarity_threshold = similarity_threshold
        self.machine_names = machine_names or []
        self.machine_name_patterns = {}  # Maps machine name to filename patterns
        
        # If no machine names provided, use common ones from the documentation
        if not self.machine_names:
            self.machine_names = [
                "2800 Series Mini Laser Pro",
                "2800 Series",
                "Mini Laser Pro",
                "anyCUTII",
                "anyCUTIII",
                "anyCUT",
                "ANYJET",
                "Arrow Any-002",
                "DuraFlex",
                "Dura-Printer",
                "DuraBolt",
                "DuraCore",
                "VR350",
                "Digital die cutter VR350",
                "EZCut",
                "EZCut 330"
            ]
        
        logger.info(f"🤖 MachineNameMatcher initialized with {len(self.machine_names)} machine names")
    
    def _fuzzy_match(self, query: str, machine_name: str) -> float:
        """
        Calculate fuzzy string similarity between query and machine name.
        Uses simple character-based similarity (can be enhanced with Levenshtein distance).
        
        Returns:
            Similarity score between 0.0 and 1.0
        """
        query_lower = query.lower()
        machine_lower = machine_name.lower()
        
        # Exact match
        if machine_lower in query_lower or query_lower in machine_lower:
            return 1.0
        
        # Check if all significant words from machine name appear in query
        machine_words = [w for w in machine_lower.split() if len(w) > 2]
        query_words = query_lower.split()
        
        if not machine_words:
            return 0.0
        
        # Count how many machine name words appear in query
        matching_words = sum(1 for word in machine_words if any(qw.startswith(word) or word in qw for qw in query_words))
        word_similarity = matching_words / len(machine_words)
        
        # Also check character-level similarity for partial matches
        # Simple approach: count common characters
        common_chars = set(machine_lower.replace(' ', '')) & set(query_lower.replace(' ', ''))
        char_similarity = len(common_chars) / max(len(set(machine_lower.replace(' ', ''))), 1)
        
        # Combine word and character similarity
        combined_similarity = (word_similarity * 0.7) + (char_similarity * 0.3)
        
        return min(combined_similarity, 1.0)
    
    def match_machine(self, query: str) -> Optional[Tuple[str, float]]:
        """
        Check if query matches any machine name with >= threshold similarity.
        
        Args:
            query: User query string
            
        Returns:
            Tuple of (matched_machine_name, similarity_score) if match found, else None
        """
        best_match = None
        best_score = 0.0
        
        for machine_name in self.machine_names:
            similarity = self._fuzzy_match(query, machine_name)
            
            if similarity >= self.similarity_threshold and similarity > best_score:
                best_match = machine_name
                best_score = similarity
        
        if best_match:
            logger.info(f"🤖 Machine name matched: '{best_match}' (similarity: {best_score:.2%})")
            return (best_match, best_score)
        
        return None
    
    def get_filename_patterns(self, machine_name: str) -> List[str]:
        """
        Generate filename patterns that might contain this machine's documentation.
        Used for boosting chunks from matching filenames.
        
        Args:
            machine_name: Machine name to generate patterns for
            
        Returns:
            List of filename patterns (substrings to match)
        """
        patterns = []
        
        # Add the machine name itself
        patterns.append(machine_name.lower())
        
        # Add variations
        machine_lower = machine_name.lower()
        # Remove common words
        for word in ['series', 'user', 'manual', 'guide', 'pro']:
            machine_lower = machine_lower.replace(word, '').strip()
        
        if machine_lower:
            patterns.append(machine_lower)
        
        # Add individual significant words
        words = [w for w in machine_name.lower().split() if len(w) > 2]
        patterns.extend(words)
        
        return patterns


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
    """Classify query intent for optimal retrieval (fallback pattern-matching)."""
    
    def classify(self, query: str) -> QueryIntent:
        """Classify query intent using simple pattern matching."""
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


class ClaudeIntentClassifier:
    """
    Advanced intent classifier using Claude API for 95%+ accuracy.
    
    Features:
    - Semantic understanding vs pattern matching
    - Contextual confidence scoring
    - Smart keyword extraction
    - Intent-based caching to minimize API costs
    - Automatic fallback to pattern matching
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        """Initialize Claude intent classifier."""
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.claude_client = None
        self.fallback_classifier = IntentClassifier()  # Pattern-matching fallback
        self.cache = {}  # Simple in-memory cache for queries
        self.max_cache_size = 1000
        
        # Initialize Claude client
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            # Get API key from environment and strip any Windows line endings
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')  # Remove any trailing whitespace/CRLF
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Using fallback pattern-matching for intent.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection with minimal request (with timeout)
            try:
                self.claude_client.messages.create(
                    model=self.model_name,
                    max_tokens=10,
                    messages=[{"role": "user", "content": "test"}],
                    timeout=30.0  # 30 second timeout
                )
            except Exception as test_error:
                error_msg = str(test_error)
                # Don't raise on overload errors - they're temporary
                if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in type(test_error).__name__:
                    logger.warning(f"Claude API temporarily overloaded (529) during test. Will use fallback.")
                    raise  # Still raise to trigger fallback, but with cleaner message
                logger.warning(f"Claude test request failed: {type(test_error).__name__}: {test_error}")
                raise
            
            logger.info(f"✅ Claude Intent Classifier initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Using fallback pattern-matching.")
            self.claude_client = None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle overload errors more gracefully (less verbose)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in error_type:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Using fallback pattern-matching.")
                self.claude_client = None
                return
            
            # For other errors, log more details
            import traceback
            logger.warning(f"⚠️ Claude connection failed: {error_type}: {error_msg[:200]}")
            logger.debug(f"Full traceback:\n{traceback.format_exc()}")
            logger.warning("Using fallback pattern-matching.")
            self.claude_client = None
    
    def classify(self, query: str) -> QueryIntent:
        """
        Classify query intent using Claude API with intelligent caching.
        
        Args:
            query: User query string
            
        Returns:
            QueryIntent with type, confidence, keywords, and metadata
        """
        # Check cache first
        if self.enable_caching and query in self.cache:
            logger.debug(f"📦 Intent cache hit for query: {query[:50]}...")
            return self.cache[query]
        
        # Use Claude if available, otherwise fallback
        if self.claude_client:
            try:
                intent = self._classify_with_claude(query)
                
                # Cache successful result
                if self.enable_caching:
                    self._add_to_cache(query, intent)
                
                return intent
                
            except Exception as e:
                logger.warning(f"⚠️ Claude intent classification failed: {e}. Using fallback.")
                return self.fallback_classifier.classify(query)
        else:
            # Use fallback classifier
            return self.fallback_classifier.classify(query)
    
    def _classify_with_claude(self, query: str) -> QueryIntent:
        """Classify intent using Claude API."""
        
        prompt = f"""You are an expert query intent classifier for a technical document retrieval system.

Analyze the following user query and classify its intent into ONE of these 5 categories:

1. **definition** - User wants to understand what something is or means
   Examples: "What is X?", "Define Y", "Explain Z", "What does ABC mean?"

2. **lookup** - User wants specific facts, numbers, or specifications
   Examples: "What is the temperature range?", "How much does X weigh?", "What are the specs?"

3. **troubleshooting** - User has a problem and needs help fixing it
   Examples: "Error when doing X", "How to fix Y?", "Z is not working", "Printer jam issue"

4. **reasoning** - User wants to understand a process, procedure, or how to do something
   Examples: "How to install X?", "What are the steps for Y?", "Procedure for Z", "How do I configure ABC?"

5. **comparison** - User wants to compare options, features, or alternatives
   Examples: "Compare X vs Y", "Difference between A and B", "Which is better?", "X or Y?"

USER QUERY: "{query}"

Respond in this EXACT JSON format (no markdown, no extra text):
{{
  "intent_type": "one of: definition, lookup, troubleshooting, reasoning, comparison",
  "confidence": 0.95,
  "reasoning": "Brief explanation of why this intent was chosen",
  "keywords": ["key", "terms", "from", "query"],
  "requires_subqueries": false,
  "temporal_context": null
}}

Rules:
- confidence should be 0.0-1.0 (be honest about uncertainty)
- keywords should be 3-8 most important terms
- requires_subqueries = true if query is complex or involves multiple steps/comparisons
- temporal_context can be "recent", "historical", "future" if time-related, otherwise null
"""

        response = self.claude_client.messages.create(
            model=self.model_name,
            max_tokens=500,
            temperature=0.1,  # Low temperature for consistent classification
            messages=[{"role": "user", "content": prompt}]
        )
        
        # Parse Claude's response
        response_text = response.content[0].text.strip()
        
        # Remove markdown code blocks if present
        if response_text.startswith('```'):
            response_text = response_text.split('```')[1]
            if response_text.startswith('json'):
                response_text = response_text[4:]
            response_text = response_text.strip()
        
        # Parse JSON
        import json
        result = json.loads(response_text)
        
        # Validate intent type
        valid_intents = ['definition', 'lookup', 'troubleshooting', 'reasoning', 'comparison']
        if result['intent_type'] not in valid_intents:
            logger.warning(f"Invalid intent type from Claude: {result['intent_type']}, defaulting to lookup")
            result['intent_type'] = 'lookup'
        
        # Create QueryIntent object
        intent = QueryIntent(
            intent_type=result['intent_type'],
            confidence=float(result.get('confidence', 0.85)),
            keywords=result.get('keywords', []),
            requires_subqueries=bool(result.get('requires_subqueries', False)),
            temporal_context=result.get('temporal_context')
        )
        
        logger.info(f"🎯 Claude classified intent: {intent.intent_type} (confidence: {intent.confidence:.2%}) - {result.get('reasoning', '')}")
        
        return intent
    
    def _add_to_cache(self, query: str, intent: QueryIntent):
        """Add intent to cache with size limit."""
        if len(self.cache) >= self.max_cache_size:
            # Remove oldest entry (simple FIFO)
            first_key = next(iter(self.cache))
            del self.cache[first_key]
        
        self.cache[query] = intent


class HybridRetriever:
    """Combines dense embeddings, BM25, and metadata filtering."""
    
    def __init__(self, index, embed_model, reranker=None, document_evaluator=None):
        self.index = index
        self.embed_model = embed_model
        self.reranker = reranker
        self.document_evaluator = document_evaluator
        self.bm25 = None
        self.corpus_nodes = []
        self._initialize_bm25()
    
    def _initialize_bm25(self):
        """Initialize BM25 index from document corpus."""
        try:
            logger.info("🔧 Initializing BM25 index...")
            
            # Ensure embedding model is set before creating retriever
            if self.embed_model:
                Settings.embed_model = self.embed_model
            
            # Try to get nodes directly from docstore (most reliable)
            if hasattr(self.index, 'docstore') and self.index.docstore:
                try:
                    all_doc_ids = list(self.index.docstore.docs.keys())
                    if all_doc_ids:
                        # Get first 1000 document IDs
                        for doc_id in all_doc_ids[:1000]:
                            try:
                                doc = self.index.docstore.get_document(doc_id)
                                if doc:
                                    self.corpus_nodes.append(doc)
                            except:
                                continue
                        logger.info(f"Loaded {len(self.corpus_nodes)} nodes directly from docstore")
                except Exception as e:
                    logger.warning(f"Direct docstore access failed: {e}")
            
            # Fallback: try retrieving with queries if docstore didn't work
            if not self.corpus_nodes:
                retriever = self.index.as_retriever(similarity_top_k=1000)
                dummy_queries = ["system", "installation", "configuration", "overview"]
                for query in dummy_queries:
                    try:
                        nodes = retriever.retrieve(query)
                        if nodes:
                            self.corpus_nodes.extend(nodes)
                            if len(self.corpus_nodes) >= 100:
                                break
                    except:
                        continue
            
            if self.corpus_nodes:
                self.corpus_nodes = self.corpus_nodes[:1000]  # Limit to 1000
                tokenized_corpus = [node.text.lower().split() for node in self.corpus_nodes]
                self.bm25 = BM25Okapi(tokenized_corpus)
                logger.info(f"✅ BM25 initialized with {len(self.corpus_nodes)} documents")
            else:
                logger.warning("⚠️ No documents found for BM25 initialization")
                self.bm25 = None
        
        except Exception as e:
            logger.error(f"BM25 initialization failed: {e}", exc_info=True)
            self.bm25 = None
    
    def bm25_search(self, query: str, top_k: int = 20) -> List[Tuple[NodeWithScore, float]]:
        """
        Perform BM25 keyword search with filename boosting and pluralization handling.
        Documents with matching filenames get significant score boost.
        Filters out inactive documents (is_active=False).
        """
        if not self.bm25 or not self.corpus_nodes:
            return []
        
        # Import document metadata checker
        try:
            from .utils.document_metadata import is_document_active
        except ImportError:
            # Fallback if metadata module not available
            def is_document_active(filename: str) -> bool:
                return True
        
        # Tokenize query
        tokenized_query = query.lower().split()
        query_lower = query.lower()
        
        # Expand query with plural/singular forms for better matching
        # This helps with "winders" vs "winder", "operations" vs "operation", etc.
        expanded_terms = []
        for term in tokenized_query:
            expanded_terms.append(term)
            # Add plural/singular variants (simple heuristic)
            if term.endswith('s') and len(term) > 3:
                expanded_terms.append(term[:-1])  # Remove 's' for singular
            elif not term.endswith('s') and len(term) > 3:
                expanded_terms.append(term + 's')  # Add 's' for plural
            # Also handle common plural forms
            if term.endswith('ies') and len(term) > 4:
                expanded_terms.append(term[:-3] + 'y')  # "winders" -> "winder" (if it was "windies")
        
        # Use expanded terms for BM25 (but keep original for filename matching)
        tokenized_query_expanded = expanded_terms
        
        # Get BM25 scores from text content using expanded terms
        scores = self.bm25.get_scores(tokenized_query_expanded)
        
        # Boost scores for documents with matching filenames
        # This is critical for queries like "system requirements" matching "System Requirements.pdf"
        filename_boost_multiplier = 3.0  # Strong boost for filename matches
        for idx, node_wrapper in enumerate(self.corpus_nodes):
            # Get the actual node (handle both NodeWithScore and plain nodes)
            node = node_wrapper.node if isinstance(node_wrapper, NodeWithScore) and hasattr(node_wrapper, 'node') else node_wrapper
            
            # Check filename match
            filename = ""
            if hasattr(node, 'metadata') and node.metadata:
                filename = node.metadata.get('file_name', '')
            elif hasattr(node_wrapper, 'metadata') and node_wrapper.metadata:
                filename = node_wrapper.metadata.get('file_name', '')
            
            # Filter out inactive documents
            if filename and not is_document_active(filename):
                scores[idx] = 0.0  # Set score to 0 for inactive documents
                continue
            
            if filename:
                filename_lower = filename.lower()
                # Check if query terms appear in filename (use original terms, not expanded)
                query_words_in_filename = sum(1 for word in tokenized_query if word in filename_lower)
                if query_words_in_filename > 0:
                    # Boost score based on how many query words match
                    match_ratio = query_words_in_filename / len(tokenized_query) if tokenized_query else 0
                    # Strong boost: if filename contains most/all query words, multiply score significantly
                    boost = 1.0 + (filename_boost_multiplier * match_ratio)
                    scores[idx] *= boost
        
        # Get top-k indices
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        # Return nodes with scores - ensure we return NodeWithScore objects
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only include non-zero scores
                node_wrapper = self.corpus_nodes[idx]
                # Ensure it's a NodeWithScore with the BM25 score
                if isinstance(node_wrapper, NodeWithScore):
                    # Create new NodeWithScore with BM25 score
                    result_node = NodeWithScore(
                        node=node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper,
                        score=float(scores[idx])
                    )
                    results.append((result_node, float(scores[idx])))
                else:
                    # Wrap plain node in NodeWithScore
                    results.append((NodeWithScore(node=node_wrapper, score=float(scores[idx])), float(scores[idx])))
        
        return results
    
    def dense_search(self, query: str, top_k: int = 20) -> List[NodeWithScore]:
        """Perform dense embedding search. Filters out inactive documents."""
        try:
            # Import document metadata checker
            try:
                from .utils.document_metadata import is_document_active
            except ImportError:
                # Fallback if metadata module not available
                def is_document_active(filename: str) -> bool:
                    return True
            
            # CRITICAL: Ensure embedding model is set before creating retriever
            # This must match the model used when building the index
            if not self.embed_model:
                logger.error("No embedding model available for dense search!")
                return []
            
            # Set embedding model globally BEFORE creating retriever
            Settings.embed_model = self.embed_model
            
            # Create retriever with explicit embedding model if possible
            retriever = self.index.as_retriever(similarity_top_k=top_k * 2)  # Get more to filter
            
            # Double-check: ensure retriever uses the correct embedding model
            # Some LlamaIndex versions need explicit setting
            if hasattr(retriever, 'service_context') and retriever.service_context:
                if hasattr(retriever.service_context, 'embed_model'):
                    retriever.service_context.embed_model = self.embed_model
            
            results = retriever.retrieve(query)
            
            # Filter out inactive documents
            filtered_results = []
            for node in results:
                filename = ""
                if isinstance(node, NodeWithScore) and hasattr(node, 'node'):
                    if hasattr(node.node, 'metadata') and node.node.metadata:
                        filename = node.node.metadata.get('file_name', '')
                elif hasattr(node, 'metadata') and node.metadata:
                    filename = node.metadata.get('file_name', '')
                
                # Only include active documents
                if not filename or is_document_active(filename):
                    filtered_results.append(node)
            
            results = filtered_results[:top_k]  # Trim to top_k after filtering
            
            if not results:
                logger.warning(f"Dense search returned 0 results for query: {query[:50]}")
                # Try once more with a simpler approach
                try:
                    # Direct vector store access as fallback
                    if hasattr(self.index, 'vector_store'):
                        query_embedding = self.embed_model.get_query_embedding(query)
                        if query_embedding:
                            # Try direct vector similarity search
                            logger.debug("Attempting direct vector store query...")
                except Exception as fallback_error:
                    logger.debug(f"Fallback retrieval also failed: {fallback_error}")
            
            return results
        except Exception as e:
            logger.error(f"Dense search failed: {e}", exc_info=True)
            return []
    
    def _search_by_filename(self, query: str, top_k: int = 10) -> List[NodeWithScore]:
        """
        Direct filename-based search that queries the index for documents with matching filenames.
        This is critical for queries like "system requirements" matching "System Requirements.pdf".
        """
        query_terms = [t.lower() for t in query.split() if len(t) > 2]  # Filter short words
        if len(query_terms) < 2:
            return []
        
        filename_matches = []
        seen_filenames = set()
        
        try:
            # Use retriever with very broad queries to get diverse documents
            retriever = self.index.as_retriever(similarity_top_k=500)  # Get many results
            
            # Try multiple broad queries to get diverse documents
            # Include the actual query terms to ensure we get relevant documents
            broad_queries = query_terms[:3] + ["document", "manual", "guide", "system", "installation", "configuration", "requirements", "specification"]
            all_nodes = []
            seen_node_ids = set()
            
            for broad_query in broad_queries:
                try:
                    nodes = retriever.retrieve(broad_query)
                    for node in nodes:
                        # Avoid duplicates by node_id
                        node_id = node.node_id if hasattr(node, 'node_id') else (node.node.node_id if isinstance(node, NodeWithScore) and hasattr(node, 'node') and hasattr(node.node, 'node_id') else str(id(node)))
                        if node_id not in seen_node_ids:
                            seen_node_ids.add(node_id)
                            all_nodes.append(node)
                except Exception as e:
                    logger.debug(f"Broad query '{broad_query}' failed: {e}")
                    continue
            
            # Import document metadata checker
            try:
                from .utils.document_metadata import is_document_active
            except ImportError:
                def is_document_active(filename: str) -> bool:
                    return True
            
            # Group nodes by filename first
            nodes_by_filename = {}
            for node in all_nodes:
                # Get filename from metadata
                filename = ""
                if isinstance(node, NodeWithScore) and hasattr(node, 'node'):
                    if hasattr(node.node, 'metadata') and node.node.metadata:
                        filename = node.node.metadata.get('file_name', '') or node.node.metadata.get('filename', '')
                elif hasattr(node, 'metadata') and node.metadata:
                    filename = node.metadata.get('file_name', '') or node.metadata.get('filename', '')
                
                # Skip inactive documents
                if filename and not is_document_active(filename):
                    continue
                
                if filename:
                    if filename not in nodes_by_filename:
                        nodes_by_filename[filename] = []
                    nodes_by_filename[filename].append(node)
            
            # Now check which filenames match the query
            for filename, nodes in nodes_by_filename.items():
                filename_lower = filename.lower()
                matching_terms = sum(1 for term in query_terms if term in filename_lower)
                
                if matching_terms >= 2:  # At least 2 terms match
                    logger.info(f"📄 Filename match: {filename} (matched {matching_terms}/{len(query_terms)} terms) - found {len(nodes)} chunks")
                    # Add ALL nodes from this matching document (not just one)
                    for node in nodes:
                        actual_node = node.node if isinstance(node, NodeWithScore) and hasattr(node, 'node') else node
                        scored_node = NodeWithScore(
                            node=actual_node,
                            score=0.95 + (matching_terms * 0.05)  # Very high score for filename match
                        )
                        filename_matches.append(scored_node)
            
            # Sort by score and return top_k
            filename_matches.sort(key=lambda x: x.score, reverse=True)
            return filename_matches[:top_k]
            
        except Exception as e:
            logger.warning(f"Filename search failed: {e}")
            return []
    
    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        machine_filename_patterns: Optional[List[str]] = None  # Unused but kept for API compatibility
    ) -> List[NodeWithScore]:
        """
        Perform hybrid search combining BM25 and dense embeddings (in parallel).
        Includes aggressive filename matching for queries that match document names.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for BM25). 0.5 = equal weight
            metadata_filters: Optional metadata filters
        
        Returns:
            Ranked list of nodes
        """
        # 🚀 FIRST: Try direct filename search for queries that look like they're asking for a specific document
        query_lower = query.lower()
        query_terms = [t for t in query_lower.split() if len(t) > 2]
        
        # Check if query contains terms that might match a filename (at least 2 meaningful words)
        if len(query_terms) >= 2:
            filename_results = self._search_by_filename(query, top_k=top_k)
            if filename_results:
                logger.info(f"✅ Found {len(filename_results)} documents via direct filename search - prioritizing these")
                # Return filename matches immediately - they're highly relevant
                return filename_results
        
        # ⚡ PARALLEL EXECUTION: Run BM25 and dense search simultaneously
        with ThreadPoolExecutor(max_workers=2) as executor:
            dense_future = executor.submit(self.dense_search, query, top_k * 2)
            bm25_future = executor.submit(self.bm25_search, query, top_k * 2)
            
            # Wait for both to complete
            dense_results = dense_future.result()
            bm25_results = bm25_future.result()
            
            # Log diagnostic info
            logger.debug(f"🔍 Dense search returned {len(dense_results)} results")
            logger.debug(f"🔍 BM25 search returned {len(bm25_results)} results")
            if not dense_results and not bm25_results:
                logger.warning(f"⚠️ Both dense and BM25 searches returned 0 results for query: {query[:100]}")
                logger.debug(f"Index type: {type(self.index)}, BM25 initialized: {self.bm25 is not None}, corpus_nodes: {len(self.corpus_nodes)}")
                
                # Last resort: try filename search again
                filename_results = self._search_by_filename(query, top_k=top_k)
                if filename_results:
                    logger.info(f"✅ Fallback filename search found {len(filename_results)} documents")
                    return filename_results
        
        # Fallback: If both searches returned 0 results, try direct index access
        if not dense_results and not bm25_results:
            logger.warning("⚠️ Both searches failed - attempting fallback retrieval...")
            try:
                # Try to get results with a very generic query
                fallback_retriever = self.index.as_retriever(similarity_top_k=top_k)
                fallback_results = fallback_retriever.retrieve("system")
                if fallback_results:
                    logger.info(f"✅ Fallback retrieval found {len(fallback_results)} results")
                    dense_results = fallback_results
                else:
                    # Last resort: try getting any nodes from the index
                    logger.error("⚠️ All retrieval methods failed - index may be corrupted or incompatible")
            except Exception as e:
                logger.error(f"Fallback retrieval also failed: {e}", exc_info=True)
        
        # Combine results with scoring
        combined_scores = defaultdict(lambda: {'dense': 0.0, 'bm25': 0.0, 'node': None})
        
        # Normalize dense scores
        if dense_results:
            max_dense = max(node.score for node in dense_results) if dense_results else 1.0
            for node in dense_results:
                # Extract node_id safely
                if isinstance(node, NodeWithScore):
                    node_id = node.node_id if hasattr(node, 'node_id') else (node.node.node_id if hasattr(node.node, 'node_id') else str(id(node)))
                else:
                    node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                    # Wrap in NodeWithScore if needed
                    node = NodeWithScore(node=node, score=0.0) if not isinstance(node, NodeWithScore) else node
                combined_scores[node_id]['dense'] = node.score / max_dense
                combined_scores[node_id]['node'] = node
        
        # Normalize BM25 scores
        if bm25_results:
            max_bm25 = max(score for _, score in bm25_results) if bm25_results else 1.0
            for node, score in bm25_results:
                # Extract node_id safely - node should already be NodeWithScore from bm25_search
                if isinstance(node, NodeWithScore):
                    node_id = node.node_id if hasattr(node, 'node_id') else (node.node.node_id if hasattr(node.node, 'node_id') else str(id(node)))
                else:
                    # Wrap in NodeWithScore if somehow it's not
                    node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                    node = NodeWithScore(node=node, score=score)
                combined_scores[node_id]['bm25'] = score / max_bm25
                if combined_scores[node_id]['node'] is None:
                    combined_scores[node_id]['node'] = node
        
        # Calculate hybrid scores with filename boosting
        hybrid_results = []
        query_lower = query.lower()
        tokenized_query = query_lower.split()
        
        for node_id, scores in combined_scores.items():
            if scores['node'] is not None:
                hybrid_score = alpha * scores['dense'] + (1 - alpha) * scores['bm25']
                
                # Additional filename boost in hybrid scoring (redundant but ensures we catch it)
                node = scores['node']
                underlying_node = node.node if isinstance(node, NodeWithScore) and hasattr(node, 'node') else node
                
                # Check filename match for additional boost
                filename = ""
                if hasattr(underlying_node, 'metadata') and underlying_node.metadata:
                    filename = underlying_node.metadata.get('file_name', '')
                elif hasattr(node, 'metadata') and node.metadata:
                    filename = node.metadata.get('file_name', '')
                
                if filename:
                    filename_lower = filename.lower()
                    # Strong boost if filename contains query terms
                    query_words_in_filename = sum(1 for word in tokenized_query if word in filename_lower)
                    if query_words_in_filename >= 2:  # At least 2 words match
                        # Boost hybrid score by 50% for strong filename matches
                        hybrid_score *= 1.5
                        logger.debug(f"📄 Filename boost applied: {filename} (matched {query_words_in_filename} query words)")
                
                # Apply metadata filtering and boosting
                if metadata_filters:
                    if not self._matches_filters(node, metadata_filters):
                        continue
                
                # Create new NodeWithScore with hybrid score
                # Handle both NodeWithScore and plain nodes
                node_wrapper = scores['node']
                if isinstance(node_wrapper, NodeWithScore):
                    underlying_node = node_wrapper.node if hasattr(node_wrapper, 'node') else node_wrapper
                else:
                    underlying_node = node_wrapper
                
                scored_node = NodeWithScore(
                    node=underlying_node,
                    score=hybrid_score
                )
                hybrid_results.append(scored_node)
        
        # Sort by hybrid score
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        # Apply re-ranking if available (skip on CPU for performance)
        # Re-ranker adds ~90 seconds on CPU, only use on GPU
        if self.reranker and len(hybrid_results) > 1:
            import torch
            if torch.cuda.is_available():
                # Only re-rank on GPU (fast) - skip on CPU (too slow)
                hybrid_results = self._rerank(query, hybrid_results[:top_k * 2])
            else:
                logger.debug("Skipping re-ranker on CPU for performance (would take ~90s)")
        
        # NEW: Boost section number matches before returning
        hybrid_results = self._boost_section_matches(query, hybrid_results)
        
        # NEW: Boost machine-matched documents if patterns provided
        if machine_filename_patterns:
            hybrid_results = self._boost_machine_documents(hybrid_results, machine_filename_patterns)
        
        # Re-sort after boosting
        hybrid_results.sort(key=lambda x: x.score, reverse=True)
        
        return hybrid_results[:top_k]
    
    def _boost_machine_documents(self, nodes: List[NodeWithScore], filename_patterns: List[str]) -> List[NodeWithScore]:
        """
        Boost nodes from documents matching machine name filename patterns.
        This prioritizes chunks from the matched machine's documentation.
        
        Args:
            nodes: List of nodes to boost
            filename_patterns: List of filename patterns to match (e.g., ["2800", "mini laser"])
            
        Returns:
            List of nodes with boosted scores
        """
        boosted_count = 0
        for node in nodes:
            # Get filename from node metadata
            filename = ""
            if isinstance(node, NodeWithScore) and hasattr(node, 'node'):
                underlying_node = node.node
                if hasattr(underlying_node, 'metadata') and underlying_node.metadata:
                    filename = underlying_node.metadata.get('file_name', '')
            elif hasattr(node, 'metadata') and node.metadata:
                filename = node.metadata.get('file_name', '')
            
            if not filename:
                continue
            
            filename_lower = filename.lower()
            
            # Check if filename matches any pattern
            for pattern in filename_patterns:
                pattern_lower = pattern.lower()
                if pattern_lower in filename_lower:
                    # Strong boost for machine-matched documents (3x score)
                    node.score *= 3.0
                    boosted_count += 1
                    logger.debug(f"🤖 Machine boost: '{pattern}' matched filename '{filename}' (new score: {node.score:.3f})")
                    break  # Only boost once per node
        
        if boosted_count > 0:
            logger.info(f"🤖 Boosted {boosted_count} nodes from machine-matched documents")
        
        return nodes
    
    def _boost_section_matches(self, query: str, nodes: List[NodeWithScore]) -> List[NodeWithScore]:
        """
        Boost nodes that match section numbers mentioned in query.
        E.g., "5.2" or "section 5.2" should boost chunks from page_label "5.2"
        This helps with queries like "section 5.2 how to operate winders"
        """
        import re
        
        # Extract section numbers from query (e.g., "5.2", "section 5.2", "chapter 3")
        section_patterns = [
            r'\b(\d+\.\d+)\b',  # "5.2", "3.1.2"
            r'\b(\d+\.\d+\.\d+)\b',  # "5.2.1"
            r'section\s+(\d+\.?\d*)',  # "section 5.2" or "section 5"
            r'chapter\s+(\d+)',  # "chapter 5"
            r'page\s+(\d+)',  # "page 5"
            r'sec\s+(\d+\.?\d*)',  # "sec 5.2"
        ]
        
        section_numbers = []
        for pattern in section_patterns:
            matches = re.findall(pattern, query.lower())
            section_numbers.extend(matches)
        
        if not section_numbers:
            return nodes
        
        logger.debug(f"📑 Detected section numbers in query: {section_numbers}")
        
        # Boost nodes with matching page_label
        boosted_count = 0
        for node in nodes:
            # Get page_label from node metadata
            page_label = ""
            if isinstance(node, NodeWithScore) and hasattr(node, 'node'):
                underlying_node = node.node
                if hasattr(underlying_node, 'metadata') and underlying_node.metadata:
                    page_label = underlying_node.metadata.get('page_label', '')
            elif hasattr(node, 'metadata') and node.metadata:
                page_label = node.metadata.get('page_label', '')
            
            if not page_label:
                continue
            
            # Check if page_label matches any section number
            page_label_str = str(page_label).lower()
            for section_num in section_numbers:
                section_num_str = str(section_num).lower()
                # Match if section number appears in page_label or vice versa
                if section_num_str in page_label_str or page_label_str in section_num_str:
                    # Significant boost for section number match (2x score)
                    node.score *= 2.0
                    boosted_count += 1
                    logger.debug(f"📑 Section number boost: '{section_num}' matched page_label '{page_label}' (new score: {node.score:.3f})")
                    break
        
        if boosted_count > 0:
            logger.info(f"📑 Boosted {boosted_count} nodes for section number matches")
        
        return nodes
    
    def hybrid_search_with_llm_evaluation(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        enable_llm_evaluation: bool = True,
        machine_filename_patterns: Optional[List[str]] = None
    ) -> List[NodeWithScore]:
        """
        Perform hybrid search with optional LLM-based document evaluation.
        
        Args:
            query: Search query
            top_k: Number of results to return
            alpha: Weight for dense search (1-alpha for BM25)
            metadata_filters: Optional metadata filters
            enable_llm_evaluation: Whether to use LLM evaluation
        
        Returns:
            Ranked list of nodes with LLM evaluation applied
        """
        # First, perform standard hybrid search
        hybrid_results = self.hybrid_search(
            query=query,
            top_k=top_k * 2,  # Get more results for LLM evaluation
            alpha=alpha,
            metadata_filters=metadata_filters,
            machine_filename_patterns=machine_filename_patterns
        )
        
        # Apply LLM evaluation if enabled and evaluator is available
        # PERFORMANCE: Skip LLM evaluation on CPU or for simple queries (saves 30-60s)
        if (enable_llm_evaluation and 
            self.document_evaluator and 
            self.document_evaluator.claude_client):
            
            import torch
            # Only use LLM evaluation on GPU or when explicitly needed (it's slow!)
            # For CPU, rely on hybrid search + BM25 scoring which is already good
            if torch.cuda.is_available() or len(hybrid_results) > 20:
                logger.info(f"🤖 Applying LLM document evaluation to {len(hybrid_results)} documents")
                try:
                    # Evaluate documents with LLM (limit to top 15 for better coverage)
                    evaluated_results = self.document_evaluator.evaluate_retrieved_documents(
                        query=query,
                        nodes=hybrid_results,
                        max_documents=min(15, len(hybrid_results))  # Increased from 3 to 15 for better coverage
                    )
                    
                    # Sort by new scores and return top_k
                    evaluated_results.sort(key=lambda x: x.score, reverse=True)
                    return evaluated_results[:top_k]
                    
                except Exception as e:
                    logger.warning(f"LLM evaluation failed, falling back to standard ranking: {e}")
                    return hybrid_results[:top_k]
            else:
                logger.debug("Skipping LLM evaluation on CPU for performance")
                return hybrid_results[:top_k]
        else:
            # No LLM evaluation, return standard results
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
        intent: QueryIntent,
        answer_generator=None,
        chat_history: Optional[List[Dict[str, str]]] = None,
        matched_machine_name: Optional[str] = None
    ) -> StructuredResponse:
        """Generate structured response with answer, reasoning, and sources."""
        
        # Reset source counter
        self.source_counter = 1
        
        # Build answer from context (with LLM if available, including chat history)
        answer = self._build_answer(query, context, intent, answer_generator, chat_history or [])
        
        # Capture token usage from answer generator if available
        token_input = None
        token_output = None
        token_total = None
        cost_usd = None
        if answer_generator and hasattr(answer_generator, '_last_token_usage') and answer_generator._last_token_usage:
            token_usage = answer_generator._last_token_usage
            token_input = token_usage.get('token_input')
            token_output = token_usage.get('token_output')
            token_total = token_usage.get('token_total')
            cost_usd = token_usage.get('cost_usd')
        
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
            intent=intent,
            matched_machine_name=matched_machine_name,
            token_input=token_input,
            token_output=token_output,
            token_total=token_total,
            cost_usd=cost_usd
        )
    
    def _build_answer(
        self,
        query: str,
        context: RetrievalContext,
        intent: QueryIntent,
        answer_generator=None,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """Build answer from retrieved context using LLM or fallback to chunk-based."""
        
        if not context.nodes:
            return "The provided context does not include information to answer this query."
        
        # Try LLM answer generation first if available
        if answer_generator and answer_generator.claude_client:
            try:
                logger.info("🤖 Generating LLM answer...")
                if chat_history:
                    logger.info(f"📝 Including {len(chat_history)} previous messages in context")
                llm_answer = answer_generator.generate_answer(
                    query=query,
                    documents=context.nodes,
                    intent=intent,
                    chat_history=chat_history or []
                )
                return llm_answer
            except Exception as e:
                logger.warning(f"LLM answer generation failed: {e}, falling back to chunk-based answer")
        
        # Fallback to chunk-based answer (original method)
        return self._build_chunk_based_answer(query, context, intent)
    
    def _build_chunk_based_answer(
        self,
        query: str,
        context: RetrievalContext,
        intent: QueryIntent
    ) -> str:
        """Build answer from document chunks (original method)."""
        
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
        """Compile source summary with snippets."""
        
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
                    'content_type': node.metadata.get('content_type', 'text'),
                    'snippets': []  # Store snippets from chunks
                }
            
            if page_num != 'N/A':
                source_docs[source_name]['pages'].add(str(page_num))
            
            # Collect snippet from this chunk (first 200 chars)
            snippet = node.text[:200].strip() if hasattr(node, 'text') and node.text else ""
            if snippet and snippet not in source_docs[source_name]['snippets']:
                source_docs[source_name]['snippets'].append(snippet)
        
        # Convert to list
        for source_info in source_docs.values():
            pages = sorted(list(source_info['pages']), key=lambda x: int(x) if x.isdigit() else 0)
            # Use first snippet (most relevant) or combine first two if available
            snippets = source_info['snippets']
            snippet = snippets[0] if snippets else ""
            if len(snippets) > 1:
                # Combine first two snippets for better context
                snippet = f"{snippets[0]}... {snippets[1][:100]}"
            
            sources.append({
                'id': source_info['id'],
                'name': source_info['name'],
                'pages': ', '.join(pages) if pages else 'N/A',
                'content_type': source_info['content_type'],
                'snippet': snippet[:200] if snippet else ""  # Ensure max 200 chars
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


    
class DocumentEvaluator:
    """
    Document evaluator using Claude for LLM-based document evaluation.
    Replaces Ollama-based evaluation with Claude API.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.evaluation_cache = {}
        self.claude_client = None
        
        # Initialize Claude by default
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            # Get API key from environment and strip any Windows line endings
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')  # Remove any trailing whitespace/CRLF
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Document evaluation will be disabled.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection with a simple request
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Document Evaluator initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Document evaluation will be disabled.")
            self.claude_client = None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle overload errors more gracefully (less verbose)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in error_type:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Document evaluation will be disabled.")
                self.claude_client = None
                return
            
            logger.warning(f"⚠️ Claude connection failed: {error_type}: {error_msg[:200]}. Document evaluation will be disabled.")
            logger.debug(f"Full Claude error: {e}", exc_info=True)
            self.claude_client = None
    
    def evaluate_retrieved_documents(
        self, 
        query: str, 
        nodes: List[NodeWithScore],
        max_documents: int = 15,  # Increased from 3 to 15 for better coverage
        machine_filename_patterns: Optional[List[str]] = None  # For compatibility with hybrid_search calls
    ) -> List[NodeWithScore]:
        """
        Evaluate and re-rank retrieved documents using Claude.
        
        Args:
            query: User query
            nodes: Retrieved document nodes
            max_documents: Maximum number of documents to evaluate (default: 15)
            
        Returns:
            Re-ranked nodes based on Claude evaluation
        """
        if not self.claude_client or not nodes:
            return nodes
        
        # Evaluate up to max_documents to provide better coverage
        nodes_to_evaluate = nodes[:max_documents]
        
        logger.info(f"🔍 Evaluating only {len(nodes_to_evaluate)} documents to limit API costs")
        
        evaluations = []
        for i, node in enumerate(nodes_to_evaluate):
            try:
                # Add delay between API calls to prevent rate limiting
                if i > 0:
                    import time
                    time.sleep(0.5)  # 500ms delay between calls
                
                evaluation = self._evaluate_single_document(query, node)
                
                # Only use high-confidence evaluations
                if evaluation['confidence'] > 0.7:  # Increased threshold
                    # Adjust node score based on LLM evaluation
                    original_score = node.score
                    llm_score = evaluation['relevance_score']
                    # Weighted combination: 80% original, 20% LLM (reduced LLM weight)
                    node.score = 0.8 * original_score + 0.2 * llm_score
                    
                    evaluations.append((node, evaluation))
                    logger.info(f"Document {i+1} evaluated: score={node.score:.3f}, confidence={evaluation['confidence']:.3f}")
                else:
                    logger.debug(f"Low confidence evaluation ({evaluation['confidence']:.3f}), using original score")
                    evaluations.append((node, None))
                    
            except Exception as e:
                logger.warning(f"LLM evaluation failed for document {i+1}: {e}")
                evaluations.append((node, None))
                # Stop on first error to prevent API spam
                break
        
        # Sort by new scores
        evaluations.sort(key=lambda x: x[0].score, reverse=True)
        
        return [node for node, _ in evaluations]
    
    def _evaluate_single_document(self, query: str, node: NodeWithScore) -> Dict[str, Any]:
        """Evaluate a single document with anti-hallucination measures."""
        
        # Create cache key
        cache_key = self._create_cache_key(query, node)
        
        # Check cache first
        if self.enable_caching and cache_key in self.evaluation_cache:
            logger.debug("Using cached evaluation")
            return self.evaluation_cache[cache_key]
        
        # Limit document content to prevent token overflow
        doc_content = node.text[:1500]  # Limit to 1500 characters
        
        # Build constrained prompt
        prompt = self._build_constrained_prompt(query, doc_content)
        
        try:
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=200,  # Reduced from 500 to limit costs
                temperature=0.1,
                timeout=10.0,  # 10 second timeout
                messages=[{"role": "user", "content": prompt}]
            )
            
            evaluation = self._parse_evaluation_response(response.content[0].text)
            
            # Validate facts against original document
            evaluation = self._validate_evaluation_facts(evaluation, node.text)
            
            # Cache the result
            if self.enable_caching:
                self.evaluation_cache[cache_key] = evaluation
            
            return evaluation
            
        except Exception as e:
            logger.error(f"LLM evaluation failed: {e}")
            return {
                'relevance_score': 0.5,
                'confidence': 0.0,
                'reasoning': 'Evaluation failed',
                'key_facts': [],
                'limitations': 'LLM evaluation unavailable'
            }
    
    def _create_cache_key(self, query: str, node: NodeWithScore) -> str:
        """Create cache key for evaluation."""
        content_hash = hashlib.md5(node.text[:500].encode()).hexdigest()
        query_hash = hashlib.md5(query.encode()).hexdigest()
        return f"{query_hash}_{content_hash}"
    
    def _build_constrained_prompt(self, query: str, document: str) -> str:
        """Build constrained prompt to minimize hallucinations."""
        
        return f"""TASK: Evaluate document relevance to query with ZERO hallucinations.

CONSTRAINTS:
- Only use information explicitly present in the document
- Do not add external knowledge or assumptions
- Score must be between 0.0 and 1.0
- Be conservative with scoring
- If uncertain, use lower scores

QUERY: {query}

DOCUMENT: {document}

EVALUATION CRITERIA:
1. Direct relevance to query (0.0-0.4)
2. Completeness of information (0.0-0.3)
3. Clarity and specificity (0.0-0.3)

RESPOND WITH JSON ONLY (no other text):
{{
    "relevance_score": 0.85,
    "reasoning": "Document directly addresses the query about...",
    "key_facts": ["Fact 1", "Fact 2"],
    "confidence": 0.9,
    "limitations": "Document doesn't cover..."
}}"""
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response and extract evaluation data."""
        try:
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                evaluation = json.loads(json_match.group())
            else:
                # Fallback parsing
                evaluation = self._fallback_parse(response)
            
            # Validate required fields
            required_fields = ['relevance_score', 'reasoning', 'confidence']
            for field in required_fields:
                if field not in evaluation:
                    evaluation[field] = 0.5 if field == 'relevance_score' else 'Unknown'
            
            # Ensure score is in valid range
            evaluation['relevance_score'] = max(0.0, min(1.0, float(evaluation['relevance_score'])))
            evaluation['confidence'] = max(0.0, min(1.0, float(evaluation['confidence'])))
            
            return evaluation
            
        except Exception as e:
            logger.warning(f"Failed to parse LLM response: {e}")
            return {
                'relevance_score': 0.5,
                'reasoning': 'Parse error',
                'confidence': 0.0,
                'key_facts': [],
                'limitations': 'Response parsing failed'
            }
    
    def _fallback_parse(self, response: str) -> Dict[str, Any]:
        """Fallback parsing when JSON extraction fails."""
        # Extract score from response
        score_match = re.search(r'score[:\s]*([0-9.]+)', response, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5
        
        return {
            'relevance_score': score,
            'reasoning': 'Fallback parsing used',
            'confidence': 0.3,
            'key_facts': [],
            'limitations': 'JSON parsing failed'
        }
    
    def _validate_evaluation_facts(self, evaluation: Dict, original_document: str) -> Dict:
        """Validate that evaluation facts are actually in the document."""
        claimed_facts = evaluation.get('key_facts', [])
        validated_facts = []
        
        for fact in claimed_facts:
            # Check if fact is actually present in document (case-insensitive)
            if fact.lower() in original_document.lower():
                validated_facts.append(fact)
            else:
                logger.debug(f"Fact not found in document: {fact}")
        
        evaluation['validated_facts'] = validated_facts
        evaluation['fact_validation_score'] = (
            len(validated_facts) / len(claimed_facts) 
            if claimed_facts else 1.0
        )
        
        # Adjust confidence based on fact validation
        if evaluation['fact_validation_score'] < 0.5:
            evaluation['confidence'] *= 0.7  # Reduce confidence for poor fact validation
        
        return evaluation
    
    def clear_cache(self):
        """Clear evaluation cache."""
        self.evaluation_cache.clear()
        logger.info("Document evaluation cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_evaluations': len(self.evaluation_cache),
            'cache_enabled': self.enable_caching
        }


class ClaudeQueryRewriter:
    """
    Claude-powered query rewriting and expansion for improved retrieval.
    Generates semantically-rich query variations optimized for vector search.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.cache = {}
        self.claude_client = None
        
        # Initialize Claude
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Query rewriting will use fallback.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Query Rewriter initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Query rewriting will use fallback.")
            self.claude_client = None
        except Exception as e:
            error_msg = str(e)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in type(e).__name__:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Query rewriting will use fallback.")
            else:
                logger.warning(f"⚠️ Claude Query Rewriter initialization failed: {error_msg[:200]}")
            self.claude_client = None
    
    def expand_query(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Generate 3-5 query variations optimized for retrieval.
        
        Args:
            query: Original query
            intent: Query intent classification
            
        Returns:
            List of query variations (includes original)
        """
        if not self.claude_client:
            return [query]  # Fallback: return original query
        
        # Create cache key
        cache_key = hashlib.md5(f"{query}_{intent.intent_type}".encode()).hexdigest()
        
        if self.enable_caching and cache_key in self.cache:
            logger.debug("Using cached query expansion")
            return self.cache[cache_key]
        
        try:
            prompt = f"""Generate 3-5 query variations optimized for technical document retrieval.

Original query: "{query}"
Intent: {intent.intent_type}
Confidence: {intent.confidence:.2%}

Generate variations that:
1. Use technical synonyms and related terms
2. Include domain-specific terminology
3. Maintain the core information need
4. Optimize for vector similarity search
5. Include alternative phrasings that might appear in technical docs

Return ONLY a JSON array of query strings, no explanation.
Example: ["query variation 1", "query variation 2", "query variation 3"]

Query variations:"""
            
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            variations = json.loads(response_text)
            
            # Ensure original query is included
            if query not in variations:
                variations.insert(0, query)
            
            # Limit to 5 variations
            variations = variations[:5]
            
            # Cache the result
            if self.enable_caching:
                self.cache[cache_key] = variations
            
            logger.info(f"🔍 Generated {len(variations)} query variations")
            return variations
            
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}, using original query")
            return [query]
    
    def clear_cache(self):
        """Clear query expansion cache."""
        self.cache.clear()


class ClaudeQueryDecomposer:
    """
    Claude-powered query decomposition for complex queries.
    Breaks multi-part queries into focused sub-queries for better retrieval.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.cache = {}
        self.claude_client = None
        
        # Initialize Claude
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Query decomposition will be disabled.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Query Decomposer initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Query decomposition will be disabled.")
            self.claude_client = None
        except Exception as e:
            error_msg = str(e)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in type(e).__name__:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Query decomposition will be disabled.")
            else:
                logger.warning(f"⚠️ Claude Query Decomposer initialization failed: {error_msg[:200]}")
            self.claude_client = None
    
    def decompose(self, query: str, intent: QueryIntent) -> List[str]:
        """
        Break complex queries into optimized sub-queries.
        
        Args:
            query: Original query
            intent: Query intent classification
            
        Returns:
            List of sub-queries (or single query if not complex)
        """
        # Skip decomposition for simple queries
        if not intent.requires_subqueries or not self.claude_client:
            return [query]
        
        # Create cache key
        cache_key = hashlib.md5(f"{query}_{intent.intent_type}".encode()).hexdigest()
        
        if self.enable_caching and cache_key in self.cache:
            logger.debug("Using cached query decomposition")
            return self.cache[cache_key]
        
        try:
            prompt = f"""Decompose this technical query into 2-4 focused sub-queries for document retrieval.

Query: "{query}"
Intent: {intent.intent_type}
Keywords: {', '.join(intent.keywords[:5])}

Each sub-query should:
- Be independently answerable from documents
- Focus on a specific aspect of the original query
- Use clear, technical language
- Optimize for vector similarity search
- Avoid redundancy

For comparison queries, create separate queries for each item being compared.
For procedural queries, break into logical steps or components.

Return ONLY a JSON array of sub-query strings, no explanation.
Example: ["sub-query 1", "sub-query 2", "sub-query 3"]

Sub-queries:"""
            
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=500,
                temperature=0.2,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            sub_queries = json.loads(response_text)
            
            # Ensure we have at least 2 sub-queries (otherwise decomposition wasn't helpful)
            if len(sub_queries) < 2:
                logger.debug("Decomposition produced <2 queries, using original")
                return [query]
            
            # Limit to 4 sub-queries
            sub_queries = sub_queries[:4]
            
            # Cache the result
            if self.enable_caching:
                self.cache[cache_key] = sub_queries
            
            logger.info(f"🔀 Decomposed query into {len(sub_queries)} sub-queries")
            return sub_queries
            
        except Exception as e:
            logger.warning(f"Query decomposition failed: {e}, using original query")
            return [query]
    
    def clear_cache(self):
        """Clear query decomposition cache."""
        self.cache.clear()


class ClaudeMetadataFilterGenerator:
    """
    Claude-powered metadata filter generation.
    Extracts metadata filters from queries to improve retrieval precision.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.cache = {}
        self.claude_client = None
        
        # Initialize Claude
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Metadata filter generation will be disabled.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Metadata Filter Generator initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Metadata filter generation will be disabled.")
            self.claude_client = None
        except Exception as e:
            error_msg = str(e)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in type(e).__name__:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Metadata filter generation will be disabled.")
            else:
                logger.warning(f"⚠️ Claude Metadata Filter Generator initialization failed: {error_msg[:200]}")
            self.claude_client = None
    
    def generate_filters(self, query: str, available_metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Extract metadata filters from query.
        
        Args:
            query: User query
            available_metadata: Optional dict of available metadata keys/values
            
        Returns:
            Dict of metadata filters (empty if none found)
        """
        if not self.claude_client:
            return {}
        
        # Create cache key
        cache_key = hashlib.md5(query.encode()).hexdigest()
        
        if self.enable_caching and cache_key in self.cache:
            logger.debug("Using cached metadata filters")
            return self.cache[cache_key]
        
        try:
            # Build available metadata description
            metadata_desc = ""
            if available_metadata:
                metadata_desc = f"\n\nAvailable metadata keys: {', '.join(available_metadata.keys())}"
            
            prompt = f"""Extract metadata filters from this technical query.

Query: "{query}"{metadata_desc}

Extract metadata filters such as:
- file_name patterns or specific document names mentioned
- content_type preferences (table, image, text, figure_caption)
- page_number ranges if mentioned
- Any other metadata filters that would narrow results

Return ONLY a JSON object with filter keys and values, or empty object {{}} if no filters found.
Example: {{"content_type": "table", "file_name": "manual.pdf"}}
Example: {{"content_type": ["table", "text"]}}
Example: {{}}

Metadata filters:"""
            
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.1,
                messages=[{"role": "user", "content": prompt}]
            )
            
            response_text = response.content[0].text.strip()
            
            # Remove markdown code blocks if present
            if response_text.startswith('```'):
                response_text = response_text.split('```')[1]
                if response_text.startswith('json'):
                    response_text = response_text[4:]
                response_text = response_text.strip()
            
            # Parse JSON
            filters = json.loads(response_text)
            
            # CRITICAL FIX: Remove strict file_name filters - they're too restrictive
            # Instead, rely on filename boosting in hybrid_search which is more flexible
            if filters and 'file_name' in filters:
                file_name_filter = filters['file_name']
                logger.info(f"⚠️ Removing strict file_name filter '{file_name_filter}' - using filename boosting instead (more flexible)")
                filters.pop('file_name')
            
            # Validate filters against available metadata
            if available_metadata and filters:
                validated_filters = {}
                for key, value in filters.items():
                    if key in available_metadata:
                        validated_filters[key] = value
                    elif key in ['content_type', 'page_number']:  # Removed 'file_name' from allowed keys
                        # Common metadata keys we can use
                        validated_filters[key] = value
                filters = validated_filters
            
            # Cache the result
            if self.enable_caching:
                self.cache[cache_key] = filters
            
            if filters:
                logger.info(f"🎯 Generated metadata filters: {filters}")
            
            return filters
            
        except Exception as e:
            logger.warning(f"Metadata filter generation failed: {e}")
            return {}
    
    def clear_cache(self):
        """Clear metadata filter cache."""
        self.cache.clear()


class ClaudeIterativeRetriever:
    """
    Claude-powered iterative retrieval with feedback.
    Uses initial results to refine queries and retrieve complementary information.
    """
    
    def __init__(self, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.cache = {}
        self.claude_client = None
        
        # Initialize Claude
        self._initialize_claude()
    
    def _initialize_claude(self):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Iterative retrieval will be disabled.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Iterative Retriever initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Iterative retrieval will be disabled.")
            self.claude_client = None
        except Exception as e:
            error_msg = str(e)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in type(e).__name__:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Iterative retrieval will be disabled.")
            else:
                logger.warning(f"⚠️ Claude Iterative Retriever initialization failed: {error_msg[:200]}")
            self.claude_client = None
    
    def refine_query(self, query: str, initial_results: List[NodeWithScore], intent: QueryIntent) -> Optional[str]:
        """
        Generate refined query based on initial retrieval results.
        
        Args:
            query: Original query
            initial_results: Initial retrieval results
            intent: Query intent classification
            
        Returns:
            Refined query string, or None if refinement not needed
        """
        if not self.claude_client or len(initial_results) == 0:
            return None
        
        # Only refine if we have enough results to analyze
        if len(initial_results) < 3:
            return None
        
        # Create cache key
        result_summary = "".join([n.text[:100] for n in initial_results[:5]])
        cache_key = hashlib.md5(f"{query}_{result_summary}".encode()).hexdigest()
        
        if self.enable_caching and cache_key in self.cache:
            logger.debug("Using cached query refinement")
            return self.cache[cache_key]
        
        try:
            # Prepare summaries of initial results
            result_summaries = []
            for i, node in enumerate(initial_results[:5], 1):
                source_name = node.metadata.get('file_name', 'Unknown')
                content_type = node.metadata.get('content_type', 'text')
                text_preview = node.text[:200].replace('\n', ' ')
                result_summaries.append(f"[{i}] {source_name} ({content_type}): {text_preview}...")
            
            prompt = f"""Original query: "{query}"
Intent: {intent.intent_type}

Initial retrieval results:
{chr(10).join(result_summaries)}

Analyze these results and generate a refined query that:
1. Targets information gaps in the initial results
2. Uses different terminology to find complementary documents
3. Maintains the original intent
4. Focuses on missing aspects that would complete the answer

Return ONLY the refined query string, or "NONE" if no refinement is needed.
Do not include explanations or quotes around the query.

Refined query:"""
            
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=200,
                temperature=0.3,
                messages=[{"role": "user", "content": prompt}]
            )
            
            refined_query = response.content[0].text.strip()
            
            # Remove quotes if present
            if refined_query.startswith('"') and refined_query.endswith('"'):
                refined_query = refined_query[1:-1]
            elif refined_query.startswith("'") and refined_query.endswith("'"):
                refined_query = refined_query[1:-1]
            
            # Check if refinement was recommended
            if refined_query.upper() == "NONE" or refined_query.lower() == query.lower():
                return None
            
            # Cache the result
            if self.enable_caching:
                self.cache[cache_key] = refined_query
            
            logger.info(f"🔄 Generated refined query: {refined_query}")
            return refined_query
            
        except Exception as e:
            logger.warning(f"Query refinement failed: {e}")
            return None
    
    def should_iterate(self, query: str, initial_results: List[NodeWithScore], intent: QueryIntent) -> bool:
        """
        Determine if iterative retrieval should be performed.
        
        Args:
            query: Original query
            initial_results: Initial retrieval results
            intent: Query intent classification
            
        Returns:
            True if iterative retrieval is recommended
        """
        # Only iterate for complex queries
        if not intent.requires_subqueries:
            return False
        
        # Only iterate if we have some results (but might need more)
        if len(initial_results) < 3:
            return False
        
        # Check average relevance scores
        if initial_results:
            avg_score = np.mean([node.score for node in initial_results[:5]])
            # If scores are low, iteration might help
            if avg_score < 0.5:
                return True
        
        return False
    
    def clear_cache(self):
        """Clear query refinement cache."""
        self.cache.clear()


class ClaudeAnswerGenerator:
    """
    Claude-based answer generator for ChatGPT-style responses.
    Generates clean, technical answers from retrieved documents.
    """
    
    # Token budget: conservative limit for input context (leaves room for response)
    MAX_INPUT_TOKENS = 100000  # ~100k tokens for Claude Sonnet 4
    
    def __init__(self, api_key: str = None, model_name: str = "claude-sonnet-4-20250514", enable_caching: bool = True):
        self.model_name = model_name
        self.enable_caching = enable_caching
        self.answer_cache = {}
        self.claude_client = None
        self._last_token_usage = None  # Store last token usage for access
        
        # Initialize Claude by default
        self._initialize_claude(api_key)
    
    def _initialize_claude(self, api_key: str = None):
        """Initialize Claude client with error handling."""
        try:
            import anthropic
            
            # Get API key from environment or parameter
            if not api_key:
                api_key = os.getenv('ANTHROPIC_API_KEY')
            
            # Strip any Windows line endings from API key
            if api_key:
                api_key = api_key.strip().rstrip('\r\n')  # Remove any trailing whitespace/CRLF
            
            if not api_key:
                logger.warning("⚠️ ANTHROPIC_API_KEY not found. Claude answer generation will be disabled.")
                self.claude_client = None
                return
            
            self.claude_client = anthropic.Anthropic(api_key=api_key)
            
            # Test connection with a simple request
            self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "test"}]
            )
            
            logger.info(f"✅ Claude Answer Generator initialized with model: {self.model_name}")
            
        except ImportError:
            logger.warning("⚠️ Anthropic package not installed. Claude answer generation will be disabled.")
            self.claude_client = None
        except Exception as e:
            error_type = type(e).__name__
            error_msg = str(e)
            
            # Handle overload errors more gracefully (less verbose)
            if "529" in error_msg or "overload" in error_msg.lower() or "OverloadedError" in error_type:
                logger.warning(f"⚠️ Claude API temporarily overloaded (529). Claude answer generation will be disabled.")
                self.claude_client = None
                return
            
            logger.warning(f"⚠️ Claude connection failed: {error_type}: {error_msg[:200]}. Claude answer generation will be disabled.")
            logger.debug(f"Full Claude error: {e}", exc_info=True)
            self.claude_client = None
    
    def generate_answer(
        self, 
        query: str, 
        documents: List[NodeWithScore],
        intent: QueryIntent,
        chat_history: Optional[List[Dict[str, str]]] = None
    ) -> str:
        """
        Generate a clean, ChatGPT-style answer from retrieved documents.
        
        Args:
            query: User query
            documents: Retrieved document nodes
            intent: Query intent classification
            
        Returns:
            Clean, technical answer with citations
        """
        if not self.claude_client or not documents:
            return self._fallback_answer(query, documents)
        
        # Create cache key
        cache_key = self._create_answer_cache_key(query, documents)
        
        # Check cache first
        if self.enable_caching and cache_key in self.answer_cache:
            logger.debug("Using cached answer")
            return self.answer_cache[cache_key]
        
        try:
            # Prepare context from documents
            context = self._prepare_document_context(documents)
            
            # Build base prompt template (without history) to estimate fixed token usage
            base_prompt = self._build_answer_prompt(query, context, intent, chat_history=None)
            
            # Trim chat history to fit within token budget (removes oldest first)
            trimmed_history = self._trim_chat_history(chat_history or [], query, context, base_prompt)
            
            # Build final prompt with trimmed history
            prompt = self._build_answer_prompt(query, context, intent, trimmed_history)
            
            # Final safety check: verify prompt doesn't exceed budget
            total_tokens = self._estimate_tokens(prompt)
            if total_tokens > self.MAX_INPUT_TOKENS:
                logger.warning(f"Prompt exceeds budget after trimming ({total_tokens} tokens), using minimal context")
                # Fallback: use only current query with documents, no history
                prompt = base_prompt
                trimmed_history = []
            
            # Build messages list with trimmed chat history + current query
            messages = []
            
            # Add trimmed chat history (excluding current query)
            if trimmed_history:
                for msg in trimmed_history:
                    if msg.get("content") != query:  # Don't duplicate current query
                        messages.append({
                            "role": msg.get("role", "user"),
                            "content": msg.get("content", "")
                        })
            
            # Add current query with RAG context
            messages.append({"role": "user", "content": prompt})
            
            # Generate answer with Claude
            response = self.claude_client.messages.create(
                model=self.model_name,
                max_tokens=2000,  # Increased for more detailed technical answers
                temperature=0.1,
                messages=messages
            )
            
            answer = response.content[0].text
            
            # Extract token usage from response
            token_input = None
            token_output = None
            token_total = None
            cost_usd = None
            
            if hasattr(response, 'usage') and response.usage:
                token_input = getattr(response.usage, 'input_tokens', None)
                token_output = getattr(response.usage, 'output_tokens', None)
                if token_input is not None and token_output is not None:
                    token_total = token_input + token_output
                    # Estimate cost: Claude Sonnet 4 pricing (as of 2025)
                    # Input: ~$3 per 1M tokens, Output: ~$15 per 1M tokens
                    cost_usd = (token_input / 1_000_000 * 3.0) + (token_output / 1_000_000 * 15.0)
            
            # Validate answer against source documents
            answer = self._validate_answer_facts(answer, documents)
            
            # Cache the result (without token info for cache key)
            if self.enable_caching:
                self.answer_cache[cache_key] = answer
            
            # Return answer with token usage info
            # Store token usage in a way that can be accessed later
            # We'll attach it as metadata to the answer string (hacky but works)
            # Actually, better to return a tuple or modify the return type
            # For now, we'll store it in a class attribute that can be accessed
            self._last_token_usage = {
                'token_input': token_input,
                'token_output': token_output,
                'token_total': token_total,
                'cost_usd': cost_usd
            }
            
            return answer
            
        except Exception as e:
            logger.error(f"Claude answer generation failed: {e}")
            return self._fallback_answer(query, documents)
    
    def _estimate_tokens(self, text: str) -> int:
        """Simple token estimation: ~4 chars per token (conservative)."""
        return len(text) // 4
    
    def _trim_chat_history(self, chat_history: List[Dict[str, str]], query: str, context: str, base_prompt: str) -> List[Dict[str, str]]:
        """
        Trim chat history to fit within token budget.
        Removes oldest messages first until budget is satisfied.
        """
        if not chat_history:
            return []
        
        # Estimate tokens for fixed parts (base prompt already includes query + context)
        fixed_tokens = self._estimate_tokens(base_prompt)
        # Reserve 20% buffer for prompt overhead (summary formatting, etc.)
        available_tokens = int((self.MAX_INPUT_TOKENS - fixed_tokens) * 0.8)
        
        if available_tokens <= 0:
            logger.warning(f"Fixed context exceeds token budget, using no chat history")
            return []
        
        # Calculate tokens for each message in history
        # Process in reverse (newest first) to keep most recent context
        trimmed_history = []
        total_tokens = 0
        
        for msg in reversed(chat_history):
            msg_text = msg.get("content", "")
            msg_tokens = self._estimate_tokens(msg_text)
            
            if total_tokens + msg_tokens <= available_tokens:
                trimmed_history.insert(0, msg)  # Insert at beginning to maintain order
                total_tokens += msg_tokens
            else:
                break  # Stop when we'd exceed budget
        
        if len(trimmed_history) < len(chat_history):
            logger.info(f"Trimmed chat history: {len(chat_history)} -> {len(trimmed_history)} messages ({total_tokens}/{available_tokens} tokens)")
        
        return trimmed_history
    
    def _prepare_document_context(self, documents: List[NodeWithScore]) -> str:
        """Prepare document context for LLM."""
        context_parts = []
        
        # Include all documents - let the LLM filter irrelevant chunks intelligently
        for i, node in enumerate(documents, 1):  # Use all retrieved documents
            source_name = node.metadata.get('file_name', f'Document {i}')
            page_num = node.metadata.get('page_label', 'N/A')
            
            context_parts.append(f"[{i}] {source_name} (Page {page_num}):")
            context_parts.append(node.text[:1500])  # Increased from 800 to 1500 for more context
            context_parts.append("")  # Empty line between documents
        
        return "\n".join(context_parts)
    
    def _build_answer_prompt(self, query: str, context: str, intent: QueryIntent, chat_history: Optional[List[Dict[str, str]]] = None) -> str:
        """Build prompt for technical answer generation."""
        
        intent_guidance = {
            'troubleshooting': "Focus on step-by-step troubleshooting procedures and solutions.",
            'definition': "Provide clear, technical definitions with examples.",
            'reasoning': "Explain the process or procedure in logical steps.",
            'comparison': "Compare features, benefits, and differences clearly.",
            'lookup': "Provide specific technical details and specifications."
        }
        
        guidance = intent_guidance.get(intent.intent_type, "Provide a comprehensive technical answer.")
        
        # Add chat history context if available
        history_context = ""
        if chat_history and len(chat_history) > 0:
            history_context = "\n\nPREVIOUS CONVERSATION:\n"
            for msg in chat_history[-5:]:  # Include last 5 messages for context
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "user":
                    history_context += f"User: {content}\n"
                elif role == "assistant":
                    history_context += f"Assistant: {content}\n"
            history_context += "\nUse the conversation history to understand context, corrections, or follow-up questions.\n"
        
        return f"""TASK: Generate a clean, technical answer to the user's query using ONLY the provided documents.{history_context}

CONSTRAINTS:
- Use ONLY information from the provided documents
- Do NOT add external knowledge or assumptions
- Maintain technical accuracy and precision
- Include proper citations [1], [2], etc.
- Write in a professional, technical style
- Be comprehensive but concise
- If this is a follow-up question, reference previous conversation context appropriately

QUERY: {query}

INTENT: {intent.intent_type.title()} - {guidance}

DOCUMENTS:
{context}

RESPONSE REQUIREMENTS:
1. Start with a direct answer to the query
2. Provide technical details and explanations
3. Include step-by-step procedures if applicable
4. Use citations [1], [2], etc. for all claims
5. End with a summary or conclusion
6. Keep the tone professional and technical
7. If this is a follow-up or correction, acknowledge the previous conversation

Generate a comprehensive technical answer:"""
    
    def _parse_answer_response(self, response: str) -> str:
        """Parse and clean the LLM response."""
        # Remove any extra formatting or prompts
        answer = response.strip()
        
        # Ensure it starts with actual content
        if answer.startswith("Answer:"):
            answer = answer[7:].strip()
        elif answer.startswith("Response:"):
            answer = answer[9:].strip()
        
        return answer
    
    def _validate_answer_facts(self, answer: str, documents: List[NodeWithScore]) -> str:
        """Validate that answer facts are supported by source documents."""
        # Extract citations from answer
        citations = re.findall(r'\[(\d+)\]', answer)
        
        # Check if citations are valid
        valid_citations = []
        for citation in citations:
            doc_index = int(citation) - 1
            if 0 <= doc_index < len(documents):
                valid_citations.append(citation)
        
        # If no valid citations, add a general source note
        if not valid_citations:
            answer += "\n\n*Based on retrieved technical documentation.*"
        
        return answer
    
    def _create_answer_cache_key(self, query: str, documents: List[NodeWithScore]) -> str:
        """Create cache key for answer generation."""
        query_hash = hashlib.md5(query.encode()).hexdigest()
        doc_hashes = [hashlib.md5(node.text[:200].encode()).hexdigest() for node in documents[:3]]
        docs_hash = hashlib.md5("".join(doc_hashes).encode()).hexdigest()
        return f"answer_{query_hash}_{docs_hash}"
    
    def _fallback_answer(self, query: str, documents: List[NodeWithScore]) -> str:
        """Fallback answer when LLM is not available."""
        if not documents:
            return "I couldn't find relevant information to answer your query."
        
        # Simple fallback: combine document chunks with citations
        answer_parts = []
        for i, node in enumerate(documents[:3], 1):
            source_name = node.metadata.get('file_name', f'Document {i}')
            answer_parts.append(f"According to {source_name} [{i}]:\n{node.text[:500]}...")
        
        return "\n\n".join(answer_parts)
    
    def clear_cache(self):
        """Clear answer cache."""
        self.answer_cache.clear()
        logger.info("LLM answer cache cleared")
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'cached_answers': len(self.answer_cache),
            'cache_enabled': self.enable_caching
        }


class RAGOrchestrator:
    """
    Elite RAG orchestrator implementing hybrid search, query rewriting,
    and structured response generation.
    """
    
    def __init__(self, cache_dir="/root/.cache/huggingface/hub", enable_llm_evaluation: bool = False, enable_llm_answers: bool = True, config_path: str = "config.yaml", db_manager=None):
        self.cache_dir = cache_dir
        self.embed_model = None
        self.reranker = None
        self.index = None
        self.retriever = None
        self.enable_llm_evaluation = enable_llm_evaluation
        self.enable_llm_answers = enable_llm_answers
        self.config = self._load_config(config_path)
        self.glossary_index = None
        self.db_manager = db_manager  # 🗄️ PostgreSQL manager for validated Q&A fast-path
        
        # Components
        self.query_rewriter = QueryRewriter()  # Rule-based fallback
        self.intent_classifier = ClaudeIntentClassifier()  # 🎯 Claude-powered intent classification
        self.response_generator = ResponseGenerator()
        self.document_evaluator = DocumentEvaluator() if enable_llm_evaluation else None
        self.answer_generator = ClaudeAnswerGenerator() if enable_llm_answers else None
        
        # 🚀 NEW: Claude-powered retrieval enhancements
        self.claude_query_rewriter = ClaudeQueryRewriter()  # Semantic query expansion
        self.claude_query_decomposer = ClaudeQueryDecomposer()  # Query decomposition
        self.claude_metadata_filter_generator = ClaudeMetadataFilterGenerator()  # Metadata filtering
        self.claude_iterative_retriever = ClaudeIterativeRetriever()  # Iterative retrieval
        
        # 🤖 Machine name matcher for query boosting
        self.machine_matcher = MachineNameMatcher()

        # User-validated cache (only stores answers marked helpful)
        self.cache = QueryCache(max_size=1000)
        self.semantic_cache = None  # Initialized after models are ready

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        try:
            import yaml
            if os.path.exists(config_path):
                with open(config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        return {}
    
    def _preprocess_long_query(self, query: str, max_length: int = 500) -> str:
        """
        Preprocess long queries by extracting key information.
        For error messages, extracts error codes and key error text.
        For other long queries, intelligently truncates while keeping important parts.
        """
        # If query is short enough, return as-is
        if len(query) <= max_length:
            return query
        
        # Check if this looks like an error message
        error_indicators = ['error', 'Error', 'ERROR', 'failed', 'Failed', 'FAILED', 
                          'RESULT_', '0x', 'exception', 'Exception', 'EXCEPTION']
        is_error_message = any(indicator in query for indicator in error_indicators)
        
        if is_error_message:
            # Extract key parts from error messages
            key_parts = []
            
            # Extract error codes (hex codes, RESULT_ codes, etc.)
            import re
            error_codes = re.findall(r'(RESULT_\w+|0x[0-9a-fA-F]+|\w+_ERR)', query)
            if error_codes:
                key_parts.extend(error_codes)
            
            # Extract error messages (text after "error", "failed", etc.)
            error_patterns = [
                r'error[:\s]+([^.\n]+)',
                r'failed[:\s]+([^.\n]+)',
                r'Error[:\s]+([^.\n]+)',
                r'Failed[:\s]+([^.\n]+)',
            ]
            for pattern in error_patterns:
                matches = re.findall(pattern, query, re.IGNORECASE)
                key_parts.extend(matches)
            
            # Extract key technical terms (uppercase words, technical terms)
            technical_terms = re.findall(r'\b[A-Z][A-Z0-9_]+\b', query)
            key_parts.extend(technical_terms[:5])  # Limit to 5 most important
            
            # Extract first and last sentences (often contain context)
            sentences = re.split(r'[.!?]\s+', query)
            if sentences:
                key_parts.append(sentences[0])  # First sentence
                if len(sentences) > 1:
                    key_parts.append(sentences[-1])  # Last sentence
            
            # Combine key parts
            if key_parts:
                processed = ' '.join(set(key_parts))  # Remove duplicates
                # If still too long, truncate intelligently
                if len(processed) > max_length:
                    # Keep error codes and first part
                    processed = ' '.join(key_parts[:3])[:max_length]
                return processed
        
        # For non-error long queries, keep first part and key terms
        # Extract first sentence and important keywords
        sentences = query.split('.')
        first_part = sentences[0] if sentences else query[:200]
        
        # Extract important keywords (longer words, technical terms)
        words = query.split()
        important_words = [w for w in words if len(w) > 6 or w[0].isupper()][:10]
        
        # Combine
        processed = f"{first_part} {' '.join(important_words)}"
        if len(processed) > max_length:
            processed = processed[:max_length]
        
        return processed.strip()

    def _load_glossary_index(self):
        try:
            glossary_cfg = (self.config or {}).get('glossary', {})
            if not glossary_cfg or not glossary_cfg.get('enabled', False):
                return
            path = glossary_cfg.get('path') or ''
            if not path:
                return
            if not os.path.isabs(path):
                # Resolve relative to project root
                path = os.path.join(os.getcwd(), path)
            if not os.path.exists(path):
                logger.warning(f"Glossary file not found at {path}")
                return
            from .glossary_loader import load_glossary_any
            nodes = load_glossary_any(path)
            if not nodes:
                logger.warning("No glossary entries loaded")
                return
            from llama_index.core import VectorStoreIndex
            self.glossary_index = VectorStoreIndex.from_documents(nodes, show_progress=False)
            logger.info(f"✅ Loaded glossary index with {len(nodes)} entries from {path}")
            # Optionally enrich acronym map from aliases
            try:
                # Fetch all nodes by a dummy query
                retr = self.glossary_index.as_retriever(similarity_top_k=min(500, len(nodes)))
                hits = retr.retrieve("glossary")
                for h in hits:
                    md = getattr(h, 'metadata', {}) or {}
                    term = (md.get('term') or '').strip()
                    for a in md.get('aliases', []) or []:
                        a_low = a.lower()
                        if 1 < len(a_low) <= 10 and a_low.isalnum():
                            self.query_rewriter.acronym_map[a_low] = term
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Glossary index init failed: {e}")
    
    def initialize_models(self):
        """Initialize embedding and re-ranking models."""
        logger.info("🚀 Initializing models for RAG orchestrator...")
        
        # Disable hf_transfer if not installed (RunPod issue)
        import os
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
                
                self.embed_model = HuggingFaceEmbedding(
                    model_name=model_name,
                    cache_folder=self.cache_dir,
                    trust_remote_code=True,
                    device=device
                )
                logger.info(f"✅ Embedding model loaded: {display_name} on {device}")
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
                            trust_remote_code=True,
                            device=device
                        )
                        logger.info(f"✅ Embedding model loaded: {display_name} (with prefix) on {device}")
                        break
                    except:
                        continue
        
        if not self.embed_model:
            # Emergency fallback
            try:
                self.embed_model = HuggingFaceEmbedding(
                    model_name="all-MiniLM-L6-v2",
                    cache_folder=self.cache_dir
                )
                logger.info("✅ Loaded with emergency fallback")
            except:
                raise RuntimeError("Could not load embedding model. Check internet connection.")
        
        # Re-ranker model
        try:
            logger.info("Loading re-ranker model...")
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-large",
                cache_folder=self.cache_dir,
                device=device
            )
            logger.info(f"✅ Re-ranker loaded on {device}")
        except Exception as e:
            logger.warning(f"Re-ranker not available: {e}")
            self.reranker = None
        
        # Initialize semantic cache after embed model is ready
        try:
            cache_cfg = (self.config or {}).get('cache', {})
            sem_cfg = cache_cfg.get('semantic', {})
            if sem_cfg.get('enabled', True):
                threshold = float(sem_cfg.get('threshold', 0.95))
                max_size = int(sem_cfg.get('max_size', 500))
                self.semantic_cache = SemanticCache(self.embed_model, threshold=threshold, max_size=max_size)
        except Exception as e:
            logger.warning(f"Semantic cache init failed: {e}")

        # Set global settings
        Settings.embed_model = self.embed_model
        logger.info("✅ Models initialized successfully")
    
    def load_index(self, storage_dir="latest_model"):
        """Load existing index from latest_model/ directory."""
        if not os.path.exists(storage_dir):
            raise FileNotFoundError(
                f"Index not found at {storage_dir}. "
                f"Run 'python -m backend.ingest' to build the index first, "
                f"or pull from git if using pre-built index."
            )
        
        logger.info("🔄 Loading index...")
        
        # CRITICAL: Set embedding model in Settings BEFORE loading index
        # This ensures the retriever uses the correct embedding model
        if self.embed_model:
            Settings.embed_model = self.embed_model
            logger.info(f"✅ Set global embedding model: {type(self.embed_model).__name__}")
        else:
            logger.warning("⚠️ No embedding model set - retrieval may fail!")
        
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir)
        self.index = load_index_from_storage(storage_context)
        
        # Initialize hybrid retriever
        self.retriever = HybridRetriever(
            index=self.index,
            embed_model=self.embed_model,
            reranker=self.reranker,
            document_evaluator=self.document_evaluator
        )
        
        logger.info("✅ Index and retriever initialized")
        # Initialize glossary if configured
        self._load_glossary_index()
    
    def orchestrate_query(
        self,
        query: str,
        top_k: int = 10,
        alpha: float = 0.5,
        metadata_filters: Optional[Dict[str, Any]] = None,
        dynamic_windowing: bool = True,
        chat_history: Optional[List[Dict[str, str]]] = None
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
        
        start_time = time.time()
        
        # Preprocess long queries - extract key information and truncate if needed
        original_query = query
        query = self._preprocess_long_query(query)
        if query != original_query:
            logger.info(f"📝 Preprocessed long query ({len(original_query)} -> {len(query)} chars)")
        
        logger.info(f"🎯 Orchestrating query: {query[:200]}{'...' if len(query) > 200 else ''}")

        # ------------------------------------------------------------------
        # 🤖 Machine Name Matching: Check if query matches a machine name
        # ------------------------------------------------------------------
        matched_machine_name = None
        machine_filename_patterns = []
        
        machine_match_result = self.machine_matcher.match_machine(query)
        if machine_match_result:
            matched_machine_name, similarity = machine_match_result
            machine_filename_patterns = self.machine_matcher.get_filename_patterns(matched_machine_name)
            logger.info(f"🤖 Query matched machine: '{matched_machine_name}' (similarity: {similarity:.2%})")
            logger.info(f"🤖 Will boost chunks from files matching: {machine_filename_patterns}")

        # ------------------------------------------------------------------
        # ⚡ User-validated cache: serve instantly if previously marked helpful
        #    1) Exact-match cache (query + params)
        #    2) Semantic cache (embedding similarity >= threshold)
        # ------------------------------------------------------------------
        try:
            # 1) Exact match
            cached = self.cache.get(query, top_k, alpha)
            if cached is not None:
                logger.info("✅ Served from user-validated cache (exact match)")
                return cached
            # 2) Semantic match
            if self.semantic_cache is not None:
                scached = self.semantic_cache.get(query)
                if scached is not None:
                    logger.info("✅ Served from user-validated cache (semantic match)")
                    return scached
        except Exception as e:
            logger.warning(f"Cache lookup failed (continuing without cache): {e}")
        
        # ------------------------------------------------------------------
        # 🗄️ PostgreSQL Validated Q&A Fast-Path
        #    Check database for user-validated answers before expensive RAG
        # ------------------------------------------------------------------
        if self.db_manager:
            try:
                validated = self.db_manager.get_validated_answer(query)
                if validated and validated.get('helpful_count', 0) >= 2:
                    # At least 2 users marked this as helpful - serve it!
                    logger.info(f"⚡ Served from validated Q&A database! (helpful_count: {validated['helpful_count']})")
                    
                    # Classify intent for metadata (fast)
                    intent = self.intent_classifier.classify(query)
                    
                    # Build response from validated Q&A
                    sources = []
                    for i, source_name in enumerate(validated.get('sources', []), 1):
                        sources.append({
                            'id': f'[{i}]',
                            'name': source_name,
                            'pages': 'N/A',
                            'content_type': 'text'
                        })
                    
                    return StructuredResponse(
                        query=query,
                        answer=validated['answer_text'],
                        reasoning="✅ Served from validated Q&A database (user-approved answer)",
                        sources=sources,
                        confidence=0.95,  # High confidence - users validated this
                        intent=intent,
                        matched_machine_name=matched_machine_name
                    )
            except Exception as e:
                logger.debug(f"Validated Q&A lookup skipped: {e}")
        
        # Step 1: Classify intent
        intent = self.intent_classifier.classify(query)
        logger.info(f"📋 Intent: {intent.intent_type} (confidence: {intent.confidence:.2%})")
        
        # 🚀 NEW: Step 1.5 - Query Decomposition (for complex queries)
        sub_queries = self.claude_query_decomposer.decompose(query, intent)
        logger.info(f"🔀 Query decomposition: {len(sub_queries)} sub-query(s)")
        
        # 🚀 NEW: Step 1.6 - Generate metadata filters
        claude_metadata_filters = self.claude_metadata_filter_generator.generate_filters(query)
        # Merge with user-provided metadata filters
        if metadata_filters:
            metadata_filters = {**claude_metadata_filters, **metadata_filters}
        else:
            metadata_filters = claude_metadata_filters
        
        # Optional: glossary augmentation
        augmented_query = query
        glossary_defs: List[str] = []
        if self.glossary_index:
            try:
                retr = self.glossary_index.as_retriever(similarity_top_k=5)
                gloss_hits = retr.retrieve(query)
                # Build alias expansion and capture up to one definition
                aliases: List[str] = []
                for h in gloss_hits[:3]:
                    md = getattr(h, 'metadata', {}) or {}
                    aliases.extend(md.get('aliases', []) or [])
                    term = (md.get('term') or '').strip()
                    if term and len(glossary_defs) < 1:
                        # Extract definition from "term: definition"
                        parts = (h.text or '').split(':', 1)
                        if len(parts) == 2:
                            glossary_defs.append(f"{term}: {parts[1].strip()}")
                aliases = [a for a in dict.fromkeys([a for a in aliases if a])]  # dedupe
                if aliases:
                    augmented_query = f"{query} ({' | '.join(aliases)})"
            except Exception as e:
                logger.debug(f"Glossary augmentation skipped: {e}")

        # 🚀 NEW: Step 2 - Query Expansion (for each sub-query)
        all_search_queries = []
        for sub_query in sub_queries:
            # Expand each sub-query with Claude
            try:
                expanded_queries = self.claude_query_rewriter.expand_query(sub_query, intent)
                all_search_queries.extend(expanded_queries)
            except Exception as e:
                logger.warning(f"Query expansion failed for '{sub_query}': {e}. Using original sub-query.")
                all_search_queries.append(sub_query)  # Fallback to original sub-query
        
        # Remove duplicates while preserving order
        seen = set()
        unique_search_queries = []
        for q in all_search_queries:
            if q.lower() not in seen:
                seen.add(q.lower())
                unique_search_queries.append(q)
        
        # Limit to top 5 query variations to avoid excessive API calls
        search_queries = unique_search_queries[:5]
        # Ensure we have at least one query - always fallback to original if all else fails
        if not search_queries:
            logger.warning("⚠️ No search queries generated - using original query as fallback")
            search_queries = [augmented_query if augmented_query != query else query]
        
        logger.info(f"🔍 Using {len(search_queries)} query variation(s) for retrieval")
        
        # Step 3: Multi-query retrieval (if we have multiple queries)
        unique_nodes = []
        if len(search_queries) > 1:
            # Retrieve results for each query variation and combine
            logger.info(f"🔄 Running multi-query retrieval across {len(search_queries)} variations...")
            all_nodes = []
            node_scores = defaultdict(float)
            
            for search_query in search_queries:
                try:
                    nodes = self.retriever.hybrid_search_with_llm_evaluation(
                        query=search_query,
                        top_k=top_k,  # Get top_k per query
                        alpha=alpha,
                        metadata_filters=metadata_filters,
                        enable_llm_evaluation=self.enable_llm_evaluation,
                        machine_filename_patterns=machine_filename_patterns
                    )
                    
                    # Combine scores (nodes may appear multiple times)
                    for node in nodes:
                        node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                        node_scores[node_id] = max(node_scores[node_id], node.score)
                        # Only add if not already in all_nodes
                        if not any(n.node_id == node_id if hasattr(n, 'node_id') else str(id(n)) == node_id for n in all_nodes):
                            all_nodes.append(node)
                except Exception as e:
                    logger.warning(f"Retrieval failed for query variation '{search_query}': {e}")
                    continue
            
            # Re-score nodes based on maximum score across all queries
            for node in all_nodes:
                node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                node.score = node_scores[node_id]
            
            # Sort by score and take top_k
            all_nodes.sort(key=lambda n: n.score, reverse=True)
            unique_nodes = all_nodes[:top_k]
        else:
            # Single query retrieval (original behavior)
            search_query = search_queries[0] if search_queries else augmented_query
            logger.info(f"🔍 Retrieving top {top_k} chunks for query: {search_query}")
            
            unique_nodes = self.retriever.hybrid_search_with_llm_evaluation(
                query=search_query,
                top_k=top_k,
                alpha=alpha,
                metadata_filters=metadata_filters,
                enable_llm_evaluation=self.enable_llm_evaluation,
                machine_filename_patterns=machine_filename_patterns
            )
        
        # 🚀 NEW: Step 4 - Iterative Retrieval (if needed)
        if self.claude_iterative_retriever.should_iterate(query, unique_nodes, intent):
            logger.info("🔄 Performing iterative retrieval...")
            refined_query = self.claude_iterative_retriever.refine_query(query, unique_nodes, intent)
            
            if refined_query:
                # Retrieve additional results with refined query
                refined_nodes = self.retriever.hybrid_search_with_llm_evaluation(
                    query=refined_query,
                    top_k=top_k // 2,  # Get fewer results for refinement
                    alpha=alpha,
                    metadata_filters=metadata_filters,
                    enable_llm_evaluation=self.enable_llm_evaluation,
                    machine_filename_patterns=machine_filename_patterns
                )
                
                # Combine with original results, avoiding duplicates
                existing_node_ids = {n.node_id if hasattr(n, 'node_id') else str(id(n)) for n in unique_nodes}
                for node in refined_nodes:
                    node_id = node.node_id if hasattr(node, 'node_id') else str(id(node))
                    if node_id not in existing_node_ids:
                        unique_nodes.append(node)
                        existing_node_ids.add(node_id)
                
                # Re-sort and limit to top_k
                unique_nodes.sort(key=lambda n: n.score, reverse=True)
                unique_nodes = unique_nodes[:top_k]
                logger.info(f"✅ Iterative retrieval added {len(refined_nodes)} new results")
        
        # Ensure we have exactly top_k results (in case hybrid search returned fewer)
        unique_nodes = unique_nodes[:top_k]
        
        # 🚀 CRITICAL FIX: Filename-based fallback if retrieval returns few/no results
        # This ensures documents with matching filenames are found even if text doesn't match
        if len(unique_nodes) < 3:  # If we got very few results, try filename matching
            logger.info(f"⚠️ Low retrieval results ({len(unique_nodes)}), attempting filename-based fallback...")
            try:
                # Extract key terms from query for filename matching
                query_terms = query.lower().split()
                query_terms = [t for t in query_terms if len(t) > 2]  # Filter out short words
                
                # Search all nodes in corpus for filename matches
                filename_matches = []
                if hasattr(self.retriever, 'corpus_nodes') and self.retriever.corpus_nodes:
                    for node_wrapper in self.retriever.corpus_nodes:
                        # Get the actual node
                        node = node_wrapper.node if isinstance(node_wrapper, NodeWithScore) and hasattr(node_wrapper, 'node') else node_wrapper
                        
                        # Check filename
                        filename = ""
                        if hasattr(node, 'metadata') and node.metadata:
                            filename = node.metadata.get('file_name', '')
                        
                        if filename:
                            filename_lower = filename.lower()
                            # Count how many query terms match the filename
                            matching_terms = sum(1 for term in query_terms if term in filename_lower)
                            if matching_terms >= 2:  # At least 2 terms match
                                # Create NodeWithScore with high score for filename match
                                scored_node = NodeWithScore(
                                    node=node,
                                    score=0.8 + (matching_terms * 0.1)  # High score: 0.8-1.0
                                )
                                filename_matches.append(scored_node)
                                logger.info(f"📄 Filename match found: {filename} (matched {matching_terms} terms)")
                
                # Add filename matches to results (avoid duplicates)
                if filename_matches:
                    existing_node_ids = {n.node_id if hasattr(n, 'node_id') else str(id(n)) for n in unique_nodes}
                    for match_node in filename_matches:
                        node_id = match_node.node_id if hasattr(match_node, 'node_id') else str(id(match_node))
                        if node_id not in existing_node_ids:
                            unique_nodes.append(match_node)
                            existing_node_ids.add(node_id)
                    
                    # Re-sort by score
                    unique_nodes.sort(key=lambda n: n.score, reverse=True)
                    unique_nodes = unique_nodes[:top_k]
                    logger.info(f"✅ Filename fallback added {len(filename_matches)} matching documents")
            except Exception as e:
                logger.warning(f"Filename fallback failed: {e}")
        
        retrieval_time = time.time() - start_time
        if len(search_queries) > 1:
            logger.info(f"⚡ Retrieval completed in {retrieval_time:.2f}s (multi-query retrieval with {len(search_queries)} variations)")
        else:
            logger.info(f"⚡ Retrieval completed in {retrieval_time:.2f}s")
        
        # Skip dynamic windowing - just use the top_k chunks directly
        # (Simple approach: just get the requested number of best chunks)
        
        logger.info(f"📚 Retrieved {len(unique_nodes)} unique chunks")
        
        # 🚨 CRITICAL: If retrieval returned 0 results, try fallback strategies
        if not unique_nodes:
            logger.warning("⚠️ Initial retrieval returned 0 results - attempting fallback strategies...")
            
            # Fallback 1: Try original query without Claude rewriting
            logger.info("🔄 Fallback 1: Trying original query without query expansion...")
            try:
                fallback_nodes = self.retriever.hybrid_search(
                    query=original_query,  # Use the original query before preprocessing
                    top_k=top_k * 2,  # Get more results
                    alpha=alpha,
                    machine_filename_patterns=machine_filename_patterns
                )
                if fallback_nodes:
                    logger.info(f"✅ Fallback 1 succeeded: found {len(fallback_nodes)} results")
                    unique_nodes = fallback_nodes[:top_k]
                else:
                    logger.warning("⚠️ Fallback 1 failed: 0 results")
            except Exception as e:
                logger.warning(f"⚠️ Fallback 1 error: {e}")
            
            # Fallback 2: Try simplified query (remove "DuraFlex" prefix, just search for core terms)
            if not unique_nodes:
                logger.info("🔄 Fallback 2: Trying simplified query...")
                simplified_query = query.lower()
                # Remove common prefixes
                for prefix in ['duraflex', 'duracore', 'durabolt', 'what are', 'what is', 'tell me about']:
                    if simplified_query.startswith(prefix):
                        simplified_query = simplified_query[len(prefix):].strip()
                        break
                
                if simplified_query and simplified_query != query.lower():
                    try:
                        fallback_nodes = self.retriever.hybrid_search(
                            query=simplified_query,
                            top_k=top_k * 2,
                            alpha=alpha,
                            machine_filename_patterns=machine_filename_patterns
                        )
                        if fallback_nodes:
                            logger.info(f"✅ Fallback 2 succeeded: found {len(fallback_nodes)} results")
                            unique_nodes = fallback_nodes[:top_k]
                    except Exception as e:
                        logger.warning(f"⚠️ Fallback 2 error: {e}")
            
            # Fallback 3: Try keyword-only search (extract key terms)
            if not unique_nodes:
                logger.info("🔄 Fallback 3: Trying keyword-only search...")
                # Extract key terms (words longer than 3 chars, excluding common words)
                common_words = {'the', 'are', 'for', 'and', 'with', 'from', 'that', 'this', 'what', 'how', 'when', 'where', 'which'}
                keywords = [w.lower() for w in query.split() if len(w) > 3 and w.lower() not in common_words]
                if keywords:
                    keyword_query = ' '.join(keywords[:5])  # Use top 5 keywords
                    try:
                        fallback_nodes = self.retriever.hybrid_search(
                            query=keyword_query,
                            top_k=top_k * 2,
                            alpha=alpha,
                            machine_filename_patterns=machine_filename_patterns
                        )
                        if fallback_nodes:
                            logger.info(f"✅ Fallback 3 succeeded: found {len(fallback_nodes)} results")
                            unique_nodes = fallback_nodes[:top_k]
                    except Exception as e:
                        logger.warning(f"⚠️ Fallback 3 error: {e}")
            
            # Fallback 4: Last resort - try very generic terms from the query
            if not unique_nodes:
                logger.info("🔄 Fallback 4: Trying generic term search...")
                # Extract any technical terms (capitalized words or common technical words)
                technical_terms = ['network', 'requirements', 'configuration', 'setup', 'installation', 'system']
                found_terms = [term for term in technical_terms if term in query.lower()]
                if found_terms:
                    generic_query = ' '.join(found_terms[:3])
                    try:
                        fallback_nodes = self.retriever.hybrid_search(
                            query=generic_query,
                            top_k=top_k * 2,
                            alpha=alpha,
                            machine_filename_patterns=machine_filename_patterns
                        )
                        if fallback_nodes:
                            logger.info(f"✅ Fallback 4 succeeded: found {len(fallback_nodes)} results")
                            unique_nodes = fallback_nodes[:top_k]
                    except Exception as e:
                        logger.warning(f"⚠️ Fallback 4 error: {e}")
        
        # Step 3: Build retrieval context
        context = self._build_retrieval_context(unique_nodes)
        
        # Step 4: Generate structured response (with chat history if provided)
        response = self.response_generator.generate_structured_response(
            query=query,
            context=context,
            intent=intent,
            answer_generator=self.answer_generator,
            chat_history=chat_history or [],
            matched_machine_name=matched_machine_name
        )

        # Optionally preface with a short glossary definition if we found one
        if glossary_defs and response and isinstance(response.answer, str):
            preface = f"Definition: {glossary_defs[0]}\n\n"
            response.answer = preface + response.answer
        
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

