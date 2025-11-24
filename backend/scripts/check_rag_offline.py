"""
CI check script to verify embedding model can be loaded offline.

This script ensures that the primary embedding model (BAAI/bge-base-en-v1.5)
can be loaded from the local cache without network access, matching production behavior.

This should be run in CI before deployment to catch model loading issues early.
"""

import os
import sys

# Set offline mode BEFORE any imports that might trigger network calls
os.environ["HF_HUB_OFFLINE"] = "1"

# Get cache directory (should match Dockerfile: /app/.cache/huggingface)
cache_dir = os.getenv("HF_HOME", "/app/.cache/huggingface")

# Ensure consistent environment variables
os.environ.setdefault("HF_HOME", cache_dir)
os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
os.environ.setdefault("HF_DATASETS_CACHE", cache_dir)
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_dir)

print("=" * 70)
print("🔍 CI Check: Offline Embedding Model Loading")
print("=" * 70)
print(f"Cache directory: {cache_dir}")
print(f"HF_HUB_OFFLINE: {os.environ.get('HF_HUB_OFFLINE')}")
print()

try:
    # Import after setting environment variables
    from backend.utils.embedding_utils import build_offline_embedding
    
    # Primary production model
    model_name = "BAAI/bge-base-en-v1.5"
    print(f"📦 Loading model: {model_name}")
    print("   (This should load from cache, no network calls)")
    
    # Build embedding model in offline mode
    embed_model = build_offline_embedding(
        model_name=model_name,
        cache_dir=cache_dir,
        device="cpu"  # Use CPU for CI check
    )
    
    print("   ✅ Model loaded successfully")
    
    # Test that model actually works by computing an embedding
    print("   🧪 Testing model with sample query...")
    test_query = "test query for CI verification"
    embedding = embed_model.get_query_embedding(test_query)
    
    if embedding is None or len(embedding) == 0:
        print("   ❌ ERROR: Model returned empty embedding")
        sys.exit(1)
    
    print(f"   ✅ Model functional (embedding dim: {len(embedding)})")
    print()
    print("=" * 70)
    print("✅ CI Check PASSED: Model can be loaded offline")
    print("=" * 70)
    sys.exit(0)
    
except ImportError as e:
    print(f"❌ ERROR: Failed to import required modules: {e}")
    print("   Ensure backend code is available in CI environment")
    sys.exit(1)
    
except Exception as e:
    print(f"❌ ERROR: Failed to load model offline: {e}")
    print()
    print("Possible causes:")
    print("  1. Model not pre-downloaded in Dockerfile")
    print("  2. Cache directory path mismatch")
    print("  3. Model files corrupted or incomplete")
    print("  4. Missing dependencies")
    print()
    print(f"Cache directory exists: {os.path.exists(cache_dir)}")
    if os.path.exists(cache_dir):
        print(f"Cache directory contents: {os.listdir(cache_dir)[:5]}...")
    sys.exit(1)

