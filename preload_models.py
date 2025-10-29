#!/usr/bin/env python3
"""
Preload HuggingFace models during Docker build.
This script downloads and caches all required models so they're ready at runtime.
"""

import os
import sys
import torch
import glob

def main():
    os.environ.setdefault('HF_HUB_ENABLE_HF_TRANSFER', '0')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'🖥️ Using device: {device}')
    sys.stdout.flush()

    # Determine cache directory
    cache_dir = os.getenv('HF_HOME', '/app/.cache/huggingface')
    if not cache_dir.endswith('hub'):
        cache_dir = os.path.join(cache_dir, 'hub')
    
    os.makedirs(cache_dir, exist_ok=True)
    print(f'📂 Cache directory: {cache_dir}')
    sys.stdout.flush()

    # Preload embedding model
    print('📥 Preloading embedding model: BAAI/bge-large-en-v1.5...')
    sys.stdout.flush()
    try:
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        embed_model = HuggingFaceEmbedding(
            model_name='BAAI/bge-large-en-v1.5',
            cache_folder=cache_dir,
            trust_remote_code=True,
            device=device
        )
        # Warm up the model
        test_embedding = embed_model.get_text_embedding('test query warmup')
        print(f'✅ Embedding model preloaded (embedding dimension: {len(test_embedding)})')
        sys.stdout.flush()
    except Exception as e:
        print(f'❌ ERROR loading embedding model: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Preload reranker model
    print('📥 Preloading reranker model: BAAI/bge-reranker-large...')
    sys.stdout.flush()
    try:
        from sentence_transformers import CrossEncoder
        reranker = CrossEncoder(
            'BAAI/bge-reranker-large',
            cache_folder=cache_dir,
            device=device
        )
        # Warm up the model
        test_score = reranker.predict([('test query', 'test document')])
        print(f'✅ Reranker model preloaded (test score: {float(test_score):.4f})')
        sys.stdout.flush()
    except Exception as e:
        print(f'❌ ERROR loading reranker model: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

    # Verify cache files exist
    print('🔍 Verifying cached model files...')
    sys.stdout.flush()
    cache_files = glob.glob(os.path.join(cache_dir, '**', '*.bin'), recursive=True) + \
                  glob.glob(os.path.join(cache_dir, '**', '*.safetensors'), recursive=True)
    
    if cache_files:
        total_size = sum(os.path.getsize(f) for f in cache_files if os.path.exists(f)) / (1024*1024*1024)
        print(f'✅ Verified {len(cache_files)} model files cached')
        print(f'   Total cache size: {total_size:.2f} GB')
        sys.stdout.flush()
    else:
        print('❌ ERROR: No cache files found after preloading!', file=sys.stderr)
        print('   Models were loaded but not cached properly.', file=sys.stderr)
        return 1

    print('✅ All models preloaded and verified successfully!')
    return 0

if __name__ == '__main__':
    sys.exit(main())

