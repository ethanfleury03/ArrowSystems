# Dockerfile Optimization Notes

## Changes Made

The Dockerfile has been optimized to reduce image size from ~3-4GB to under 1GB by:

1. **Removed model preloading during build** - Models now download at runtime on first use
2. **Removed RAG index preloading** - Index loads at runtime
3. **Removed unnecessary system libraries** - Removed GUI libraries (libgl1-mesa-dri, libsm6, libxrender-dev, libxext6) that aren't needed for text-only RAG
4. **Moved build-essential to dependencies stage only** - Build tools are not included in final image
5. **Used python:3.11-slim-bookworm** - Lighter base image
6. **Multi-stage optimization** - Only runtime dependencies copied to final stage

## Code Changes Needed

### ✅ No Code Changes Required!

Your existing code already handles runtime model downloads correctly:

1. **Cache directory is properly configured**:
   - `HF_HOME` environment variable is set in Dockerfile
   - Code uses `os.getenv('HF_HOME', '/app/.cache/huggingface/hub')` with fallback
   - Cache directory is created and writable by `appuser`

2. **Model loading handles downloads automatically**:
   - `HuggingFaceEmbedding` and `CrossEncoder` automatically download models if not cached
   - Error handling is in place (see `orchestrator.py` lines 2404-2430)
   - Fallback models are available if primary download fails

3. **Cache directory permissions**:
   - Directory is created with `chmod -R 755 /app/.cache` 
   - Owned by `appuser` so models can be written

## Runtime Considerations

### First Container Start

On the **first container start**, models will download automatically:
- **BAAI/bge-large-en-v1.5**: ~1.3GB (embedding model)
- **BAAI/bge-reranker-large**: ~1.3GB (reranker model)
- **Total**: ~2.6GB download on first use

**Expected behavior:**
- First API call may take 2-5 minutes (download + load)
- Subsequent calls are instant (models cached)
- Health check start period increased to 120s to account for first-time downloads

### Network Requirements

Ensure your container has:
- **Internet access** for HuggingFace Hub downloads
- **Sufficient disk space** (~3GB) for model cache
- **Write permissions** to `/app/.cache/huggingface`

### Persistent Cache (Recommended)

To avoid re-downloading models on every container restart, mount a volume:

```yaml
# docker-compose.yml
volumes:
  - hf_cache:/app/.cache/huggingface
```

This way models persist across container restarts.

### Health Check

The health check start period is set to **120 seconds** to allow time for:
- Model downloads (if needed)
- Model loading
- Index loading
- API startup

If models are already cached, startup is much faster (~10-30 seconds).

## Image Size Comparison

- **Before**: ~3-4GB (with preloaded models)
- **After**: ~500-800MB (base + dependencies only)
- **Runtime**: +2.6GB when models download (cached in volume)

## Verification

To verify runtime downloads work:

1. Build the image: `docker build -t rag-app .`
2. Run without cache volume: `docker run -p 8000:8000 rag-app`
3. Check logs for: `📥 Models will download automatically on first use if not cached...`
4. Make first API call - should see model download progress in logs
5. Subsequent calls should be instant

## Troubleshooting

If models fail to download:

1. **Check network**: Container needs internet access
2. **Check disk space**: `df -h` in container
3. **Check permissions**: `ls -la /app/.cache/huggingface`
4. **Check logs**: Look for HuggingFace download errors
5. **Manual download**: Can pre-download models to a volume before starting container

