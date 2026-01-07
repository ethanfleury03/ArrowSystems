# RAG Readiness Diagnosis - Root Cause Analysis

## Root Cause Statement

**The RAG backend never becomes ready because the vector store JSON parsing has a hard 120-second timeout (`backend/orchestrator.py` line 4177), and parsing the 183MB `default__vector_store.json` file exceeds this timeout on Cloud Run, causing a `RuntimeError` that prevents `IndexLoadState._status` from ever being set to `"ready"` (line 601).**

**Secondary Issues:**
1. **Dockerfile mismatch:** CI builds `backend/Dockerfile.backend` but deployment may use a different image
2. **State synchronization:** Multiple state systems (`IndexLoadState._status`, `RAGPipeline._initialized`, `index_state.phase`) must all align for readiness

---

## Task 1: Dockerfile Verification

### Current State

**CI Builds:**
- `.github/workflows/ci.yml` lines 232, 249, 264, 279: Uses `file: ./backend/Dockerfile.backend`
- Image tag: `us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/arrow-rag-backend/backend:${{ github.sha }}`

**Deployment Configuration:**
- `deployment/cloud-run-service.yaml` line 13: Hardcoded image `gcr.io/ragapp-476414/rag-app:latest` (DIFFERENT REGISTRY/TAG)
- `deployment/deploy-backend.sh`: Runs `gcloud run services replace deployment/cloud-run-service.yaml` (line 28), which may not update the image

**Problem:** The YAML has a hardcoded old image that doesn't match what CI builds.

**Fix Required:**
1. Update `deployment/cloud-run-service.yaml` to use the image built by CI
2. Or ensure `deploy-backend.sh` updates the image after replacing the YAML
3. Or verify which image Cloud Run is actually running

---

## Task 2: Stage Where Initialization Stalls

### Critical Timeout Location

**File:** `backend/orchestrator.py`  
**Lines:** 4165-4191

```python
load_thread = threading.Thread(target=_load_with_timeout, daemon=True)
load_thread.start()
load_thread.join(timeout=120)  # 2 minute timeout for index loading  <-- CRITICAL

if load_thread.is_alive():
    # Thread is still running - timeout occurred
    error_msg = (
        f"Index loading timed out after 120 seconds. "
        f"This usually indicates corrupted index files or extremely large files. "
        ...
    )
    raise RuntimeError(error_msg)
```

**This 120s timeout wraps `load_index_from_storage()` which parses the 183MB `default__vector_store.json`.**

### Expected Checkpoint Sequence

1. ✅ `[RAG] build_offline_embedding_START` → `build_offline_embedding_DONE` (embedding loads)
2. ✅ `[RAG] reranker_load_START` → `reranker_load_DONE` (reranker loads)
3. ✅ `[RAG] vector_store_parse_START` (vector store parse begins)
4. ❌ **MISSING:** `[RAG] vector_store_parse_DONE` (never appears because timeout triggers)
5. ❌ **MISSING:** `rag_index_load_done` (never reached because exception raised)
6. ❌ **MISSING:** `IndexLoadState._status = "ready"` (line 601 never reached)

### Evidence from Code Flow

**Flow:**
```
backend/api.py:startup_event() (line 1091)
  → await load_state.ensure_loaded()
    → backend/rag/index_manager.py:_do_load() (line 268)
      → await asyncio.to_thread(pipeline.ensure_initialized()) (line 489)
        → backend/rag_pipeline.py:ensure_initialized() (line 271)
          → _load_index() (line 86)
            → self.orchestrator.initialize_models() (line 110) ✅ COMPLETES
            → self.orchestrator.load_index() (line 118)
              → backend/orchestrator.py:load_index() (line 3862)
                → load_index_from_storage() in thread (line 4162) ⏱️ TIMEOUT AT 120s
                  → Raises RuntimeError (line 4191)
                → Exception propagates up, status never set to "ready"
```

---

## Task 3: Readiness Gating Logic Consistency

### `/query` Endpoint Readiness Check

**File:** `backend/api.py`  
**Lines:** 2940-2993

```python
load_state = get_index_load_state()
state = load_state.get_state()

if state["status"] != "ready":  # Checks IndexLoadState._status
    # Returns 503 with "RAG index is loading. Try again shortly."
```

### `/api/readyz` Endpoint Readiness Check

**File:** `backend/api.py`  
**Lines:** 1859-1897

```python
load_state = get_index_load_state()
state = load_state.get_state()
rag_status = state.get("status", "unknown")  # Checks IndexLoadState._status

if rag_status == "ready":
    return {"ready": True, ...}
else:
    return JSONResponse(status_code=503, ...)
```

**Also checks:**
- `pipeline.is_initialized()` (line 1870) - from `RAGPipeline._initialized`
- `index_state.get("phase")` (line 1878) - from separate `index_state` module

### State Systems (3 Total)

1. **`IndexLoadState._status`** (`backend/rag/index_manager.py` line 601)
   - Set to `"ready"` only after `pipeline.ensure_initialized()` completes successfully
   - **This is the primary check used by both `/query` and `/readyz`**

2. **`RAGPipeline._initialized`** (`backend/rag_pipeline.py` line 274)
   - Set to `True` after `_load_index()` completes
   - Checked by `/readyz` for additional verification

3. **`index_state.phase`** (`backend/rag/index_state.py`)
   - Set to `"ready"` via `set_phase("ready")` (called in multiple places)
   - Checked by `/readyz` for observability

**Issue:** If `load_index_from_storage()` times out, none of these get set correctly:
- `IndexLoadState._status` stays `"loading"` (never reaches line 601)
- `RAGPipeline._initialized` stays `False` (exception in `_load_index()`)
- `index_state.phase` might be `"loading"` or `"error"` depending on exception handling

**Conclusion:** All three state systems correctly depend on successful index load, but the 120s timeout prevents that from completing.

---

## Task 4: Multi-Process / Multi-Instance Desync

### Worker Configuration

- `GUNICORN_WORKERS=1` (set in `deployment/deploy-backend.sh` line 74)
- `min-instances=1` (set in `deployment/deploy-backend.sh` line 70)
- **Single worker per instance, minimum 1 instance**

### File Locking

**File:** `backend/rag/startup_downloader.py`  
**Mechanism:** Uses `fcntl.flock()` for per-process locking (Linux-only)

**Issue:** Lock files are per-instance, not global. If multiple instances start simultaneously:
- Each instance downloads to `/tmp/latest_model` (separate filesystems)
- Locks don't coordinate across instances
- Multiple redundant downloads can occur

**However:** Since `min-instances=1`, this is less likely unless Cloud Run scales up during startup.

### In-Memory State

- `IndexLoadState` uses module-level singleton (`_instance` pattern, line 22)
- `RAGPipeline` uses module-level singleton (`_pipeline_instance`, line 430)
- **Each worker process has its own singleton instance**
- With `GUNICORN_WORKERS=1`, only one process, so no multi-process desync

---

## Task 5: Concrete Patch Plan

### Patch 1: Increase Vector Store Parse Timeout (CRITICAL)

**File:** `backend/orchestrator.py`  
**Lines:** 4165-4191

**Change:**
```python
# BEFORE (line 4177):
load_thread.join(timeout=120)  # 2 minute timeout for index loading

# AFTER:
# Increase timeout to match RAG_MAX_LOAD_TIME_SEC (default 600s)
# Parsing 183MB JSON can legitimately take 3-5 minutes on slow CPU
vector_store_parse_timeout = int(os.getenv("RAG_VECTOR_STORE_PARSE_TIMEOUT_SEC", os.getenv("RAG_MAX_LOAD_TIME_SEC", "600")))
load_thread.join(timeout=vector_store_parse_timeout)

# Update error message to reflect actual timeout
if load_thread.is_alive():
    error_msg = (
        f"Index loading timed out after {vector_store_parse_timeout} seconds. "
        f"Vector store file (default__vector_store.json) is 183MB and may take several minutes to parse on Cloud Run. "
        f"Storage directory: {storage_dir}. "
        f"Try re-downloading the index from GCS or check file sizes."
    )
    logger.error("orchestrator_index_load_timeout",
               storage_dir=storage_dir,
               timeout_seconds=vector_store_parse_timeout,  # Update this line too
               message=error_msg)
```

**Rationale:** 183MB JSON parsing can take 3-5 minutes on Cloud Run CPU. 120s is too short.

### Patch 2: Add Timeout Logging with Duration

**File:** `backend/orchestrator.py`  
**Lines:** 4151-4201

**Add before vector store parse:**
```python
# Already exists, but ensure we log the timeout value
vector_store_parse_timeout = int(os.getenv("RAG_VECTOR_STORE_PARSE_TIMEOUT_SEC", os.getenv("RAG_MAX_LOAD_TIME_SEC", "600")))
logger.info(
    "rag_vector_store_parse_timeout_config",
    timeout_seconds=vector_store_parse_timeout,
    vector_store_size_bytes=vector_store_size,
    message=f"Vector store parse will timeout after {vector_store_parse_timeout}s for {vector_store_size:,} byte file"
)
print(f"[RAG] vector_store_parse_timeout={vector_store_parse_timeout}s size={vector_store_size:,} bytes", flush=True)
```

**Rationale:** Makes timeout visible in logs so we can confirm it's being applied.

### Patch 3: Ensure State Transition on Timeout

**File:** `backend/orchestrator.py`  
**Lines:** 4179-4191

**Change:**
```python
if load_thread.is_alive():
    # Thread is still running - timeout occurred
    # CRITICAL: Update index_state to "error" before raising
    try:
        from backend.rag.index_state import set_phase
        set_phase("error", error=f"Vector store parse timed out after {vector_store_parse_timeout}s")
    except Exception:
        pass  # Non-fatal
    
    error_msg = (...)
    raise RuntimeError(error_msg)
```

**Rationale:** Ensures `index_state.phase` reflects timeout so `/readyz` reports correct state.

### Patch 4: Fix Deployment Image Reference (If Needed)

**File:** `deployment/cloud-run-service.yaml`  
**Line:** 13

**Option A (Recommended):** Remove hardcoded image, let deploy script set it:
```yaml
# Remove: image: gcr.io/ragapp-476414/rag-app:latest
# Add comment: Image is set by deploy script
```

**Option B:** Update image to match CI build:
```yaml
image: us-central1-docker.pkg.dev/arrow-rag-support-prod/arrow-rag-backend/backend:latest
```

**Then update `deployment/deploy-backend.sh`** to set image explicitly:
```bash
# After line 31, add:
IMAGE_TAG="us-central1-docker.pkg.dev/${PROJECT}/arrow-rag-backend/backend:${GITHUB_SHA:-latest}"
gcloud run services update $SERVICE \
  --image="$IMAGE_TAG" \
  ...
```

---

## Verification Checklist

### Expected Log Sequence (Success Path)

After patch, you should see:

```
[FAULTHANDLER] enabled with timeout=600s
[GUNICORN_START] pid=1 ... timeout=600 port=8080
[APP_START] pid=X ... GUNICORN_TIMEOUT=600
[RAG] load_mode=eager eager=1 background=0
[RAG] eager load begin (timeout=600s)
[RAG] build_offline_embedding_START pid=X hostname=... revision=... model=BAAI/bge-large-en-v1.5 cache_dir=/app/.cache/huggingface
[RAG] embedding_load_done model=BAAI/bge-large-en-v1.5 duration=45.23s
[RAG] build_offline_embedding_DONE pid=X ... duration=45.23s
[RAG] reranker_load_START pid=X hostname=... revision=...
[RAG] reranker_load_DONE pid=X ... duration=12.45s
[RAG] vector_store_parse_START pid=X hostname=... revision=... storage_dir=/tmp/latest_model vector_store_size_bytes=183142647
[RAG] vector_store_parse_timeout=600s size=183,142,647 bytes
[RAG] vector_store_parse_DONE pid=X ... duration=125.67s  <-- Should appear before timeout
[RAG] rag_index_load_done status=ready total_elapsed_s=183.35s
[RAG] eager load completed; READY (duration=183.35s)
```

**If timeout still occurs, you'll see:**
```
[RAG] vector_store_parse_START ...
[RAG] vector_store_parse_timeout=600s size=183,142,647 bytes
orchestrator_index_load_timeout timeout_seconds=600  <-- If 600s timeout triggers
```

### Curl Commands

```bash
# 1. Health check (should always return 200)
curl -i https://arrow-rag-backend-xxx.run.app/api/healthz

# 2. Readiness check (should return 200 once ready)
curl -i https://arrow-rag-backend-xxx.run.app/api/readyz | jq '{ready, rag_state, phase, pipeline_initialized}'

# Expected when loading:
# {"ready": false, "rag_state": "loading", "phase": "loading", "pipeline_initialized": false}

# Expected when ready:
# {"ready": true, "rag_state": "ready", "phase": "ready", "pipeline_initialized": true}

# 3. Query endpoint (should return 503 until ready, then 200)
curl -X POST https://arrow-rag-backend-xxx.run.app/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test"}' \
  -i

# Expected when loading:
# HTTP/1.1 503 Service Unavailable
# Retry-After: 5
# {"detail": {"code": "RAG_WARMING", "status": "loading", "message": "RAG index is loading. Try again shortly."}}

# Expected when ready:
# HTTP/1.1 200 OK
# {normal query response}
```

### GCloud Verification Commands

```bash
# 1. Confirm env vars are set
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="yaml(spec.template.spec.containers[0].env)" | grep -E "GUNICORN_TIMEOUT|RAG_EAGER_LOAD|RAG_MAX_LOAD"

# Expected:
# - name: GUNICORN_TIMEOUT
#   value: "600"
# - name: RAG_EAGER_LOAD_ON_STARTUP
#   value: "1"
# - name: RAG_MAX_LOAD_TIME_SEC
#   value: "600" (if set)

# 2. Confirm image digest matches CI build
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="yaml(spec.template.spec.containers[0].image)"

# 3. Check logs for timeout occurrence
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"vector_store_parse" OR textPayload=~"index_load_timeout")' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 50 \
  --format 'table(timestamp,textPayload)'

# 4. Check for readiness transitions
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"rag_index_load_done" OR textPayload=~"eager load completed")' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 20 \
  --format 'table(timestamp,textPayload)'
```

---

## Summary

**Immediate Fix:** Increase vector store parse timeout from 120s to 600s (matching `RAG_MAX_LOAD_TIME_SEC`).

**Root Cause:** 183MB JSON file parsing exceeds 120s timeout, causing `RuntimeError` that prevents state from becoming "ready".

**Expected Outcome:** Vector store parse completes within 600s, `IndexLoadState._status` is set to `"ready"`, and `/query`/`/readyz` return 200.

