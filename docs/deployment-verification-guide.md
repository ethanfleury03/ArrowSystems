# Deployment Verification Guide

This guide provides step-by-step verification that the new revision is running and models are correctly cached.

## 4 Critical Verification Steps (In Order)

### 1) Are you actually running the new revision/image?

**Check Cloud Run logs for unique log markers:**

Look for ONE of these new log strings (they prove new code is running):

#### Build-time markers (in build logs):
```
[BUILD] VERIFICATION_MARKER: model_pre_download_start
[BUILD] VERIFICATION_MARKER: embedding_download_start
[BUILD] VERIFICATION_MARKER: embedding_download_done
[BUILD] VERIFICATION_MARKER: reranker_download_start
[BUILD] VERIFICATION_MARKER: reranker_download_done
[BUILD] VERIFICATION_MARKER: model_pre_download_done
```

#### Runtime markers (in runtime logs):
```
[VERIFICATION_MARKER] model_cache_status_accessed
[RAG] load_mode=eager eager=... background=...
[RAG] embedding_import_begin model=BAAI/bge-large-en-v1.5
[RAG] embedding_load_begin model=BAAI/bge-large-en-v1.5
[RAG] VERIFICATION_MARKER: embedding_model_ready
```

**Command to check:**
```bash
# Check for any verification markers in last 2 hours
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"VERIFICATION_MARKER|load_mode="' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 50 \
  --format 'value(textPayload)'
```

**If none of these markers appear:** You're still running an old image. Check:
- CI/CD pipeline built and deployed the new image
- Cloud Run revision actually updated
- Correct Dockerfile was used (`deployment/Dockerfile.api`)

---

### 2) Did the build really bake the models into the final image?

**Critical Dockerfile checks:**

1. **Is it multi-stage?** 
   - Our Dockerfile is **single-stage** (not multi-stage), so models downloaded during build are already in the final image ✅
   - If it were multi-stage, you'd need: `COPY --from=builder /app/.cache/huggingface /app/.cache/huggingface`

2. **Are permissions correct?**
   - The Dockerfile runs: `chown -R app:app /app` after copying code
   - This includes `/app/.cache/huggingface` which was downloaded earlier
   - Verify with: Check build logs for `chown` command and cache directory listing

3. **Check build logs for model download:**
```bash
# Look for build-time verification markers
gcloud builds list --project arrow-rag-support-prod --limit 1
# Then check logs for that build
gcloud builds log <BUILD_ID> --project arrow-rag-support-prod | grep VERIFICATION_MARKER
```

**What to look for in build logs:**
```
[BUILD] Downloading BAAI/bge-large-en-v1.5 (embedding model)...
[BUILD] Embedding model downloaded. Dimension: 1024
[BUILD] Downloading BAAI/bge-reranker-large (reranker model)...
[BUILD] Reranker model downloaded.
[BUILD] VERIFICATION_MARKER: model_pre_download_done
```

**If models aren't in the image:**
- Check Dockerfile actually includes the download step
- Check CI/CD is building from `deployment/Dockerfile.api` (not `Dockerfile`)
- Check build logs for errors during model download

---

### 3) Confirm cache status without triggering RAG

**Call the new endpoint BEFORE touching /query:**

```bash
# Set your backend URL
BACKEND_URL="https://arrow-rag-backend-xxxxx-uc.a.run.app"

# Check model cache status (doesn't load models, just checks files)
curl -sS "$BACKEND_URL/api/model_cache_status" | jq
```

**Expected successful output:**
```json
{
  "embedding_model": {
    "name": "BAAI/bge-large-en-v1.5",
    "expected_dim": 1024,
    "exists": true,
    "cache_dir": "/app/.cache/huggingface",
    "notes": "Found model in: /app/.cache/huggingface/hub/models--BAAI--bge-large-en-v1.5",
    "env_vars": {
      "SENTENCE_TRANSFORMERS_HOME": "/app/.cache/huggingface",
      "TRANSFORMERS_CACHE": "/app/.cache/huggingface",
      "HF_HOME": "/app/.cache/huggingface",
      "HF_HUB_OFFLINE": "1"
    }
  },
  "reranker_model": {
    "name": "BAAI/bge-reranker-large",
    "exists": true,
    "cache_dir": "/app/.cache/huggingface",
    "notes": "Found model in: /app/.cache/huggingface/..."
  },
  "cache_dir": "/app/.cache/huggingface",
  "all_models_cached": true,
  "verification_note": "If all_models_cached is false, check Dockerfile model download step and permissions"
}
```

**Critical checks:**
- ✅ `all_models_cached: true`
- ✅ `cache_dir` matches env vars (`/app/.cache/huggingface`)
- ✅ `exists: true` for both embedding and reranker models
- ✅ No "checked paths mismatch" in notes

**If `all_models_cached: false`:**
1. **Dockerfile step didn't run:**
   - Check build logs for `VERIFICATION_MARKER: model_pre_download_done`
   - Verify `deployment/Dockerfile.api` was used (not a different Dockerfile)

2. **Permissions wrong:**
   - Check if `app` user can read `/app/.cache/huggingface`
   - Should see `chown -R app:app /app` in Dockerfile
   - Check Cloud Run logs for permission errors

3. **Runtime path mismatch:**
   - Check env vars: `SENTENCE_TRANSFORMERS_HOME`, `HF_HOME`, `TRANSFORMERS_CACHE`
   - Should all point to `/app/.cache/huggingface`
   - If they differ, models were downloaded to wrong location

4. **Models in different location:**
   - Check `cache_contents_sample` in the response
   - Verify models are actually in `/app/.cache/huggingface`

**Also verify the endpoint logs access:**
```bash
# Check logs for endpoint access
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"VERIFICATION_MARKER.*model_cache_status"' \
  --project arrow-rag-support-prod \
  --freshness 1h \
  --limit 10
```

Should see: `[VERIFICATION_MARKER] model_cache_status_accessed`

---

### 4) Confirm RAG loads after cache is confirmed

**Only after step 3 passes (`all_models_cached: true`):**

```bash
# Check index status
curl -sS "$BACKEND_URL/api/index_status" | jq

# Check readiness
curl -i "$BACKEND_URL/api/readyz"
```

**Expected behavior:**

1. **`/api/index_status` should show:**
   ```json
   {
     "phase": "loading" or "ready",
     "ready": true or false,
     "files": { ... }
   }
   ```
   - Phase should progress: `idle -> downloading -> downloaded -> loading -> ready`
   - If stuck at `downloading` or `loading`, check logs for errors

2. **`/api/readyz` should:**
   - Return `503` while loading (with `Retry-After` header)
   - Return `200` once ready (with `{"ready": true}`)

**What to check in logs during load:**

```bash
# Watch for embedding load checkpoints
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"embedding_import_begin|embedding_load_begin|embedding_model_ready"' \
  --project arrow-rag-support-prod \
  --freshness 1h \
  --limit 20 \
  --format 'value(timestamp,textPayload)'
```

**Expected log sequence:**
```
[RAG] embedding_import_begin model=BAAI/bge-large-en-v1.5
[RAG] embedding_import_done
[RAG] embedding_load_begin model=BAAI/bge-large-en-v1.5 cache_dir=/app/.cache/huggingface device=cpu
[RAG] embedding_dim_validated model=BAAI/bge-large-en-v1.5 dim=1024
[RAG] embedding_load_done model=BAAI/bge-large-en-v1.5
[RAG] VERIFICATION_MARKER: embedding_model_ready model=BAAI/bge-large-en-v1.5
```

**If RAG doesn't become ready:**

1. **Embedding load fails:**
   - Check for `embedding_load_failed` in logs
   - Error message will include cache dir checked and env vars
   - Verify model files exist at the expected path

2. **Dimension mismatch:**
   - Should see error: "Embedding dimension mismatch: expected 1024, got X"
   - Means wrong model is loaded - check model name in cache

3. **Permission errors:**
   - Check if `app` user can read model files
   - Should see `chown` in Dockerfile and build logs

4. **Model mismatch:**
   - Index was built with `bge-large-en-v1.5` but runtime loads different model
   - Check `orchestrator.py` model name matches index build time

---

## Common Failure Modes

### Failure: "NameError: build_offline_embedding is not defined"

**Cause:** Import error or function not found

**Fix:** Verify `backend/utils/embedding_utils.py` exports `build_offline_embedding`

**Check:**
```python
python -c "from backend.utils.embedding_utils import build_offline_embedding; print('OK')"
```

---

### Failure: "Could not load embedding model from cache"

**Cause:** Model not in image or wrong path

**Steps:**
1. Check step 2 above (models baked into image)
2. Check step 3 above (cache status endpoint)
3. Verify `HF_HOME` env var matches cache location
4. Check permissions on `/app/.cache/huggingface`

---

### Failure: "Embedding dimension mismatch: expected 1024, got X"

**Cause:** Wrong model loaded or model files corrupted

**Fix:**
- Verify Dockerfile downloads `BAAI/bge-large-en-v1.5` (not `bge-base` or other variant)
- Check build logs confirm dimension is 1024
- Verify index was built with same model

---

### Failure: CI uses wrong Dockerfile

**Cause:** CI builds `Dockerfile` but you edited `deployment/Dockerfile.api`

**Fix:** Check CI build command uses correct Dockerfile:
```bash
# Should be:
docker build -f deployment/Dockerfile.api -t ...

# NOT:
docker build -t ...
```

---

## Quick Verification Checklist

After deployment, run these commands:

```bash
# 1. Check for new revision markers
gcloud logging read '... VERIFICATION_MARKER ...' --freshness 1h

# 2. Check model cache status
curl -sS "$BACKEND_URL/api/model_cache_status" | jq '.all_models_cached'

# 3. Check health
curl -sS "$BACKEND_URL/api/healthz" | jq

# 4. Check readiness
curl -sS "$BACKEND_URL/api/readyz" | jq

# 5. Check RAG mode
curl -H "Authorization: Bearer $TOKEN" "$BACKEND_URL/api/rag_mode" | jq '.mode'
```

All should pass before serving production traffic.

