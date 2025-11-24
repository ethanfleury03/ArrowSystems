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
    Build a HuggingFaceEmbedding model in offline mode.
    
    This function ensures:
    - All required environment variables are set consistently
    - Offline mode is enforced (HF_HUB_OFFLINE=1, local_files_only=True)
    - The same cache directory is used everywhere
    
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
    if cache_dir is None:
        cache_dir = os.getenv("HF_HOME", "/app/.cache/huggingface")
    
    # Ensure cache directory structure (HuggingFace expects 'hub' subdirectory)
    if not cache_dir.endswith("hub"):
        # HuggingFace models are typically in a 'hub' subdirectory
        # But we store them directly in the cache_dir, so use it as-is
        pass
    
    # Enforce consistent environment variables
    os.environ.setdefault("HF_HOME", cache_dir)
    os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
    os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
    os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)
    
    # Enforce offline mode for HuggingFace Hub
    # This prevents any network calls at runtime
    os.environ["HF_HUB_OFFLINE"] = "1"
    
    # Build embedding model with offline constraints
    return HuggingFaceEmbedding(
        model_name=model_name,
        cache_folder=cache_dir,
        trust_remote_code=True,
        device=device,
        model_kwargs={
            "local_files_only": True,  # Critical: only load from local cache
            "trust_remote_code": True,
        },
    )

