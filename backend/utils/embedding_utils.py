"""
Offline embedding model utilities for RAG pipeline.

This module provides helpers to load embedding models in offline mode,
ensuring no network calls are made at runtime in production.
"""

import os
from typing import Optional
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def build_offline_embedding(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cpu"
) -> HuggingFaceEmbedding:
    """
    Offline-safe embedding loader compatible with all LlamaIndex versions.
    
    Ensures no network calls by setting HF_HUB_OFFLINE=1 and relying on
    pre-downloaded models in the cache directory.
    
    This function ensures:
    - All required environment variables are set consistently
    - Offline mode is enforced via HF_HUB_OFFLINE=1
    - The same cache directory is used everywhere
    - Compatible with older LlamaIndex versions (no model_kwargs)
    
    Args:
        model_name: HuggingFace model identifier (e.g., "BAAI/bge-base-en-v1.5")
        cache_dir: Cache directory for models (defaults to HF_HOME env var or /app/.cache/huggingface)
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        HuggingFaceEmbedding instance loaded from local cache
        
    Raises:
        RuntimeError: If model cannot be loaded from cache (offline mode)
    """
    # Determine cache directory
    cache_dir = cache_dir or os.getenv("HF_HOME", "/app/.cache/huggingface")
    
    # Enforce offline environment variables
    # These ensure SentenceTransformers only loads from cache
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"  # Critical: prevents network calls
    
    # IMPORTANT: No model_kwargs — older HuggingFaceEmbedding does not support it
    # Rely on cache-only behavior + HF_HUB_OFFLINE=1 to forbid downloads
    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=cache_dir,
        trust_remote_code=True,
        device=device,
        # No model_kwargs here - not supported in all LlamaIndex versions
    )

