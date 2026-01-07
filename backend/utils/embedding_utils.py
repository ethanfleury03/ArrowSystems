"""
Offline embedding model utilities for RAG pipeline.

This module provides helpers to load embedding models in offline mode,
ensuring no network calls are made at runtime in production.

CRITICAL: Heavy imports (llama_index, sentence_transformers, torch) are lazy-loaded
to prevent Gunicorn worker boot timeouts during module import.

Usage:
    from backend.utils.embedding_utils import build_offline_embedding
    embed_model = build_offline_embedding("BAAI/bge-large-en-v1.5", device="cpu")

Sanity check command:
    python -c "import backend.utils.embedding_utils"  # Should succeed without heavy imports
"""

from __future__ import annotations  # CRITICAL: Postponed annotation evaluation

import os
from typing import Optional, Any, TYPE_CHECKING

# TYPE_CHECKING: Import types only for type hints, not at runtime
if TYPE_CHECKING:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def get_embedding_cache_dir() -> str:
    """
    Determine the cache directory for embedding models.
    
    Checks env vars in order:
    1. SENTENCE_TRANSFORMERS_HOME
    2. TRANSFORMERS_CACHE
    3. HF_HOME
    4. Default: /app/.cache/huggingface
    
    Returns:
        Path to the cache directory
    """
    return (
        os.getenv("SENTENCE_TRANSFORMERS_HOME") or
        os.getenv("TRANSFORMERS_CACHE") or
        os.getenv("HF_HOME") or
        "/app/.cache/huggingface"
    )


def check_embedding_model_cache(model_name: str, cache_dir: Optional[str] = None) -> dict:
    """
    Check if an embedding model exists in the cache.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "BAAI/bge-large-en-v1.5")
        cache_dir: Cache directory to check (defaults to get_embedding_cache_dir())
        
    Returns:
        Dict with:
        - exists: bool - whether model appears to be cached
        - cache_dir: str - the cache directory checked
        - model_name: str - the model name
        - notes: str - additional information
        - env_vars: dict - relevant environment variables
    """
    cache_dir = cache_dir or get_embedding_cache_dir()
    
    env_vars = {
        "SENTENCE_TRANSFORMERS_HOME": os.getenv("SENTENCE_TRANSFORMERS_HOME", ""),
        "TRANSFORMERS_CACHE": os.getenv("TRANSFORMERS_CACHE", ""),
        "HF_HOME": os.getenv("HF_HOME", ""),
        "HF_HUB_OFFLINE": os.getenv("HF_HUB_OFFLINE", ""),
    }
    
    # Check if cache directory exists
    if not os.path.isdir(cache_dir):
        return {
            "exists": False,
            "cache_dir": cache_dir,
            "model_name": model_name,
            "notes": f"Cache directory does not exist: {cache_dir}",
            "env_vars": env_vars,
        }
    
    # Check for model-specific directories
    # Models are typically stored in: {cache_dir}/hub/models--{org}--{model}/
    # or {cache_dir}/sentence_transformers/{org}_{model}/
    model_slug = model_name.replace("/", "--")
    model_slug_st = model_name.replace("/", "_")
    
    possible_paths = [
        os.path.join(cache_dir, "hub", f"models--{model_slug}"),
        os.path.join(cache_dir, f"models--{model_slug}"),
        os.path.join(cache_dir, "sentence_transformers", model_slug_st),
        os.path.join(cache_dir, model_slug_st),
    ]
    
    found_paths = [p for p in possible_paths if os.path.isdir(p)]
    
    if found_paths:
        return {
            "exists": True,
            "cache_dir": cache_dir,
            "model_name": model_name,
            "notes": f"Found model in: {found_paths[0]}",
            "found_paths": found_paths,
            "env_vars": env_vars,
        }
    
    # List what's in the cache directory for debugging
    try:
        cache_contents = os.listdir(cache_dir)[:20]  # First 20 items
    except Exception as e:
        cache_contents = [f"Error listing: {e}"]
    
    return {
        "exists": False,
        "cache_dir": cache_dir,
        "model_name": model_name,
        "notes": f"Model not found in cache. Checked: {possible_paths}",
        "cache_contents_sample": cache_contents,
        "env_vars": env_vars,
    }


def build_offline_embedding(
    model_name: str,
    cache_dir: Optional[str] = None,
    device: str = "cpu"
) -> "HuggingFaceEmbedding":
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
        model_name: HuggingFace model identifier (e.g., "BAAI/bge-large-en-v1.5")
        cache_dir: Cache directory for models (defaults to get_embedding_cache_dir())
        device: Device to use ("cpu" or "cuda")
        
    Returns:
        HuggingFaceEmbedding instance loaded from local cache
        
    Raises:
        RuntimeError: If model cannot be loaded from cache (offline mode)
    """
    # LAZY IMPORT: Import only when function is called (not at module import time)
    # This prevents torch/sentence_transformers from blocking Gunicorn worker boot
    print(f"[RAG] embedding_import_begin model={model_name}", flush=True)
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        print(f"[RAG] embedding_import_done", flush=True)
    except Exception as e:
        import traceback
        error_msg = f"Failed to import HuggingFaceEmbedding: {type(e).__name__}: {str(e)}"
        print(f"[RAG] embedding_import_failed: {error_msg}\n{traceback.format_exc()}", flush=True)
        raise RuntimeError(error_msg) from e
    
    # Determine cache directory (check env vars in order)
    cache_dir = cache_dir or get_embedding_cache_dir()
    
    # Enforce offline environment variables
    # These ensure SentenceTransformers only loads from cache
    os.environ["HF_HOME"] = cache_dir
    os.environ["TRANSFORMERS_CACHE"] = cache_dir
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    os.environ["SENTENCE_TRANSFORMERS_HOME"] = cache_dir
    os.environ["HF_HUB_OFFLINE"] = "1"  # Critical: prevents network calls
    
    print(f"[RAG] embedding_load_begin model={model_name} cache_dir={cache_dir} device={device}", flush=True)
    
    try:
        # IMPORTANT: No model_kwargs — older HuggingFaceEmbedding does not support it
        # Rely on cache-only behavior + HF_HUB_OFFLINE=1 to forbid downloads
        embed_model = HuggingFaceEmbedding(
            model_name=model_name,
            cache_folder=cache_dir,
            trust_remote_code=True,
            device=device,
            # No model_kwargs here - not supported in all LlamaIndex versions
        )
        
        # Validate embedding dimension (bge-large-en-v1.5 should be 1024)
        # CRITICAL: Lightweight check - use embed_dim attribute if available, NOT actual embedding
        # This avoids blocking startup with a full text embedding operation
        expected_dim = 1024 if "bge-large" in model_name.lower() else None
        if expected_dim and hasattr(embed_model, 'embed_dim'):
            actual_dim = embed_model.embed_dim
            if actual_dim != expected_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                    f"Model: {model_name}, Cache: {cache_dir}. "
                    f"This suggests the wrong model is loaded - check cache contents."
                )
            print(f"[RAG] embedding_dim_validated model={model_name} dim={actual_dim}", flush=True)
        elif expected_dim:
            # Fallback: try to get dimension from the underlying model if embed_dim not available
            # But don't do a full embedding operation - just check model config
            try:
                if hasattr(embed_model, '_model') and hasattr(embed_model._model, 'get_sentence_embedding_dimension'):
                    actual_dim = embed_model._model.get_sentence_embedding_dimension()
                    if actual_dim != expected_dim:
                        raise RuntimeError(
                            f"Embedding dimension mismatch: expected {expected_dim}, got {actual_dim}. "
                            f"Model: {model_name}, Cache: {cache_dir}"
                        )
                    print(f"[RAG] embedding_dim_validated model={model_name} dim={actual_dim}", flush=True)
            except Exception as e:
                # Non-fatal: dimension check failed but model loaded
                print(f"[RAG] embedding_dim_check_warning: {e}", flush=True)
        
        print(f"[RAG] embedding_load_done model={model_name}", flush=True)
        print(f"[RAG] VERIFICATION_MARKER: embedding_model_ready model={model_name}", flush=True)
        return embed_model
        
    except Exception as e:
        import traceback
        # Check cache status for better error message
        cache_status = check_embedding_model_cache(model_name, cache_dir)
        
        error_msg = (
            f"Could not load embedding model {model_name} from cache (offline mode). "
            f"Cache dir: {cache_dir}. "
            f"Cache exists: {cache_status.get('exists', False)}. "
            f"Notes: {cache_status.get('notes', 'N/A')}. "
            f"Env vars: SENTENCE_TRANSFORMERS_HOME={os.getenv('SENTENCE_TRANSFORMERS_HOME', '')}, "
            f"HF_HOME={os.getenv('HF_HOME', '')}, "
            f"HF_HUB_OFFLINE={os.getenv('HF_HUB_OFFLINE', '')}. "
            f"Original error: {type(e).__name__}: {str(e)}"
        )
        print(f"[RAG] embedding_load_failed: {error_msg}\n{traceback.format_exc()}", flush=True)
        raise RuntimeError(error_msg) from e


# Backwards-compatible alias (in case any code uses a different name)
build_embedding_offline = build_offline_embedding

