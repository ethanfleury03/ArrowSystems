#!/usr/bin/env python3
"""
Smoke test for offline embedding model loading.

This script validates that the embedding model can be loaded from local cache
without any network calls. It should fail if the model is not pre-downloaded.

Usage:
    python scripts/smoke_embedding_offline.py

Exit codes:
    0: Success - model loaded from cache
    1: Failure - model not found or load error
"""

import os
import sys
import time

# Ensure we're in offline mode
os.environ["HF_HUB_OFFLINE"] = "1"

def main():
    print("=" * 60)
    print("Embedding Model Offline Smoke Test")
    print("=" * 60)
    
    # Step 1: Check environment
    print("\n[1/4] Checking environment variables...")
    cache_dir = (
        os.getenv("SENTENCE_TRANSFORMERS_HOME") or
        os.getenv("TRANSFORMERS_CACHE") or
        os.getenv("HF_HOME") or
        "/app/.cache/huggingface"
    )
    print(f"  SENTENCE_TRANSFORMERS_HOME: {os.getenv('SENTENCE_TRANSFORMERS_HOME', '(not set)')}")
    print(f"  TRANSFORMERS_CACHE: {os.getenv('TRANSFORMERS_CACHE', '(not set)')}")
    print(f"  HF_HOME: {os.getenv('HF_HOME', '(not set)')}")
    print(f"  HF_HUB_OFFLINE: {os.getenv('HF_HUB_OFFLINE', '(not set)')}")
    print(f"  Effective cache_dir: {cache_dir}")
    
    # Step 2: Check cache directory exists
    print("\n[2/4] Checking cache directory...")
    if os.path.isdir(cache_dir):
        print(f"  ✅ Cache directory exists: {cache_dir}")
        try:
            contents = os.listdir(cache_dir)[:10]
            print(f"  Contents (first 10): {contents}")
        except Exception as e:
            print(f"  ⚠️ Could not list contents: {e}")
    else:
        print(f"  ❌ Cache directory does not exist: {cache_dir}")
        print("  Run the Docker build or manually download models first.")
        return 1
    
    # Step 3: Import and check cache status
    print("\n[3/4] Checking model cache status...")
    try:
        # Add parent directory to path if running from scripts/
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        
        from backend.utils.embedding_utils import check_embedding_model_cache, get_embedding_cache_dir
        
        embedding_status = check_embedding_model_cache("BAAI/bge-large-en-v1.5")
        print(f"  Embedding model cached: {embedding_status.get('exists', False)}")
        print(f"  Notes: {embedding_status.get('notes', 'N/A')}")
        
        reranker_status = check_embedding_model_cache("BAAI/bge-reranker-large")
        print(f"  Reranker model cached: {reranker_status.get('exists', False)}")
        print(f"  Notes: {reranker_status.get('notes', 'N/A')}")
        
        if not embedding_status.get('exists', False):
            print("\n  ❌ Embedding model not found in cache!")
            print("  To fix: Run Docker build with model pre-download step")
            return 1
            
    except Exception as e:
        print(f"  ❌ Failed to check cache: {type(e).__name__}: {e}")
        return 1
    
    # Step 4: Actually load the model
    print("\n[4/4] Loading embedding model from cache (offline)...")
    start_time = time.time()
    try:
        from backend.utils.embedding_utils import build_offline_embedding
        
        embed_model = build_offline_embedding(
            model_name="BAAI/bge-large-en-v1.5",
            cache_dir=cache_dir,
            device="cpu"
        )
        
        load_time = time.time() - start_time
        print(f"  ✅ Model loaded successfully in {load_time:.2f}s")
        
        # Validate dimension
        if hasattr(embed_model, 'embed_dim'):
            dim = embed_model.embed_dim
            print(f"  Embedding dimension: {dim}")
            if dim != 1024:
                print(f"  ⚠️ Warning: Expected 1024 dimensions for bge-large-en-v1.5, got {dim}")
        
        # Quick embedding test
        print("\n  Testing embedding generation...")
        test_text = "This is a test sentence for embedding."
        try:
            # LlamaIndex HuggingFaceEmbedding uses get_text_embedding
            embedding = embed_model.get_text_embedding(test_text)
            print(f"  ✅ Generated embedding with {len(embedding)} dimensions")
        except Exception as e:
            print(f"  ⚠️ Could not generate test embedding: {e}")
        
    except Exception as e:
        load_time = time.time() - start_time
        print(f"  ❌ Failed to load model after {load_time:.2f}s")
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "=" * 60)
    print("✅ SMOKE TEST PASSED - Embedding model loads from cache")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

