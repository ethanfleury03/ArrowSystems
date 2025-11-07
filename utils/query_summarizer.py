"""
Query Summarization Utility for Long User Inputs

Automatically summarizes long queries (emails, error logs, etc.) before sending to RAG backend.
This improves UX by allowing users to paste long content without manual editing.
"""

import os
import logging
import hashlib
import json
from typing import Optional, Dict, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)

# Try to import Anthropic (optional dependency)
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("Anthropic not available - query summarization will be disabled")


class QuerySummarizer:
    """
    Summarizes long user queries using Claude API.
    Designed for emails, error logs, and verbose questions.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-4-20250514",
        enabled: bool = True,
        min_length: int = 500,  # Only summarize queries longer than this
        cache_dir: str = ".query_summary_cache"
    ):
        """
        Initialize query summarizer.
        
        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
            model: Claude model to use
            enabled: Whether summarization is enabled
            min_length: Minimum query length to trigger summarization (chars)
            cache_dir: Directory for caching summaries
        """
        self.enabled = enabled and ANTHROPIC_AVAILABLE
        self.model = model
        self.min_length = min_length
        self.cache_dir = Path(cache_dir)
        self.client = None
        
        # Create cache directory if it doesn't exist
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
            api_key = api_key or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                logger.warning("⚠️ Query summarization enabled but ANTHROPIC_API_KEY not found. Disabling.")
                self.enabled = False
            else:
                try:
                    # Strip any Windows line endings
                    api_key = api_key.strip().rstrip('\r\n')
                    self.client = anthropic.Anthropic(api_key=api_key)
                    logger.info(f"✅ QuerySummarizer initialized (model: {model}, min_length: {min_length})")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to initialize Claude client: {e}. Disabling summarization.")
                    self.enabled = False
    
    def _get_cache_path(self, query_hash: str) -> Path:
        """Get cache file path for a query hash."""
        return self.cache_dir / f"{query_hash}.json"
    
    def _load_from_cache(self, query_hash: str) -> Optional[str]:
        """Load summary from cache if exists."""
        cache_path = self._get_cache_path(query_hash)
        if cache_path.exists():
            try:
                with open(cache_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data.get('summary')
            except Exception as e:
                logger.debug(f"Failed to load cache: {e}")
        return None
    
    def _save_to_cache(self, query_hash: str, original: str, summary: str):
        """Save summary to cache."""
        cache_path = self._get_cache_path(query_hash)
        try:
            with open(cache_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'original': original,
                    'summary': summary,
                    'hash': query_hash
                }, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save cache: {e}")
    
    def _detect_content_type(self, query: str) -> str:
        """
        Detect if query is an email, error log, or regular question.
        Returns: 'email', 'error', or 'question'
        """
        query_lower = query.lower()
        
        # Email indicators
        email_indicators = [
            'from:', 'to:', 'subject:', 'sent:', 'date:',
            '@', 'mailto:', 'reply-to:', 'cc:', 'bcc:'
        ]
        if any(indicator in query_lower for indicator in email_indicators):
            return 'email'
        
        # Error log indicators
        error_indicators = [
            'error:', 'exception:', 'traceback:', 'stack trace',
            'failed', 'failure', 'fatal', 'warning:', 'debug:',
            'log:', 'timestamp:', 'at ', 'line ', 'file:'
        ]
        if sum(1 for indicator in error_indicators if indicator in query_lower) >= 2:
            return 'error'
        
        return 'question'
    
    def summarize(self, query: str) -> Tuple[str, bool, Optional[str]]:
        """
        Summarize a query if it's long enough.
        
        Args:
            query: Original user query
            
        Returns:
            Tuple of (processed_query, was_summarized, content_type)
            - processed_query: Summarized query if applicable, else original
            - was_summarized: Whether summarization occurred
            - content_type: Detected content type ('email', 'error', 'question')
        """
        if not self.enabled or not self.client:
            return query, False, None
        
        # Check length threshold
        if len(query) < self.min_length:
            return query, False, None
        
        # Detect content type
        content_type = self._detect_content_type(query)
        
        # Check cache
        query_hash = hashlib.md5(query.encode('utf-8')).hexdigest()
        cached_summary = self._load_from_cache(query_hash)
        if cached_summary:
            logger.debug(f"✅ Using cached summary for query (hash: {query_hash[:8]}...)")
            return cached_summary, True, content_type
        
        # Generate summary using Claude
        try:
            logger.info(f"📝 Summarizing long query ({len(query)} chars, type: {content_type})")
            
            # Build prompt based on content type
            if content_type == 'email':
                prompt = f"""You are helping a user query a technical documentation system. They pasted an email. Extract the key technical question or problem they need help with.

Email content:
{query}

Extract and summarize the core technical question or problem in 1-3 sentences. Focus on:
- What technical issue or question they're asking about
- Any specific product names, error codes, or technical terms mentioned
- The main problem they need solved

Return ONLY the summarized question, nothing else."""
            
            elif content_type == 'error':
                prompt = f"""You are helping a user query a technical documentation system. They pasted an error log or stack trace. Extract the key technical problem they need help with.

Error content:
{query}

Extract and summarize the core technical problem in 1-3 sentences. Focus on:
- What error or issue occurred
- Any error codes, component names, or technical terms
- What they're trying to accomplish that failed

Return ONLY the summarized problem statement, nothing else."""
            
            else:  # question
                prompt = f"""You are helping a user query a technical documentation system. They asked a very long question. Summarize it to extract the core technical question.

Original question:
{query}

Summarize this into a concise 1-3 sentence technical question. Preserve:
- All technical terms, product names, and specifications
- The core question or problem they're asking about
- Any specific details needed to answer accurately

Return ONLY the summarized question, nothing else."""
            
            # Call Claude API
            response = self.client.messages.create(
                model=self.model,
                max_tokens=200,
                temperature=0.1,  # Low temperature for consistency
                messages=[{
                    "role": "user",
                    "content": prompt
                }]
            )
            
            summary = response.content[0].text.strip()
            
            # Validate summary (should be shorter than original)
            if len(summary) >= len(query):
                logger.warning(f"Summary not shorter than original, using original query")
                return query, False, content_type
            
            # Cache the summary
            self._save_to_cache(query_hash, query, summary)
            
            logger.info(f"✅ Summarized query: {len(query)} → {len(summary)} chars ({len(summary)/len(query)*100:.1f}% reduction)")
            
            return summary, True, content_type
            
        except Exception as e:
            logger.error(f"Failed to summarize query: {e}", exc_info=True)
            # Fallback to original query on error
            return query, False, content_type
    
    def get_stats(self) -> Dict[str, any]:
        """Get cache statistics."""
        if not self.cache_dir.exists():
            return {'cached_summaries': 0, 'cache_enabled': False}
        
        cache_files = list(self.cache_dir.glob("*.json"))
        return {
            'cached_summaries': len(cache_files),
            'cache_enabled': True,
            'cache_dir': str(self.cache_dir)
        }

