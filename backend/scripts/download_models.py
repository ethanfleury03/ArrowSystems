"""
Pre-download HuggingFace embedding models for offline use.

This script downloads all required embedding models into the HF_HOME cache directory
so they are available offline in production (Cloud Run).

Models are downloaded during Docker build and baked into the image.
"""

import os
import sys
from sentence_transformers import SentenceTransformer

# Models to pre-download (in priority order)
MODELS = [
    "BAAI/bge-base-en-v1.5",                     # Primary prod model
    "BAAI/bge-large-en-v1.5",                    # Optional fallback
    "sentence-transformers/all-MiniLM-L6-v2",    # Existing fallback
    "sentence-transformers/all-mpnet-base-v2",   # Existing fallback
]


def main():
    """Download all required models to cache directory."""
    # Get cache directory from environment (set in Dockerfile)
    cache_dir = os.getenv("HF_HOME", "/app/.cache/huggingface")
    
    # Ensure cache directory exists
    os.makedirs(cache_dir, exist_ok=True)
    
    print(f"📥 Pre-downloading embedding models to: {cache_dir}")
    print("=" * 70)
    
    success_count = 0
    failed_models = []
    
    for model_name in MODELS:
        try:
            print(f"\n📦 Downloading: {model_name}")
            # This will download the model and cache it in HF_HOME
            model = SentenceTransformer(model_name, cache_folder=cache_dir)
            
            # Test that model is actually loaded by computing a test embedding
            test_embedding = model.encode("test query", convert_to_numpy=True)
            print(f"   ✅ Successfully downloaded and verified (dim: {len(test_embedding)})")
            success_count += 1
            
        except Exception as e:
            print(f"   ❌ Failed to download {model_name}: {e}")
            failed_models.append((model_name, str(e)))
    
    print("\n" + "=" * 70)
    print(f"📊 Summary: {success_count}/{len(MODELS)} models downloaded successfully")
    
    if failed_models:
        print("\n⚠️  Failed models:")
        for model_name, error in failed_models:
            print(f"   - {model_name}: {error}")
        # Don't fail the build if some fallback models fail, but primary should succeed
        if success_count == 0:
            print("\n❌ ERROR: No models downloaded successfully. Build will fail.")
            sys.exit(1)
        elif "BAAI/bge-base-en-v1.5" in [m[0] for m in failed_models]:
            print("\n❌ ERROR: Primary model (BAAI/bge-base-en-v1.5) failed to download. Build will fail.")
            sys.exit(1)
        else:
            print("\n⚠️  WARNING: Some fallback models failed, but primary model succeeded.")
    
    print("\n✅ Model download complete!")


if __name__ == "__main__":
    main()

