# RAG Load Analysis and Patches

## Verified Facts (Code + Deployment)

### 1. Source of "Timeout (0:01:00)!" Messages
**Location:** `backend/api.py` lines 26-33

```python
faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True, file=sys.stderr)
```

**Confirmed:** This is Python's faulthandler, NOT Gunicorn timeout. It dumps stack traces every 60 seconds when the process is stuck, which matches the "Timeout (0:01:00)!" pattern.

**Evidence:** The stack traces show it's triggered during model loading in `build_offline_embedding()` → `HuggingFaceEmbedding()` → `SentenceTransformer()`.

### 2. Exact Control Flow for RAG Init

```
backend/api.py:startup_event() (line 785)
  → _get_rag_load_mode() (determines eager/background/lazy)
  → load_state.ensure_loaded() (line 1091, if eager mode)
    → backend/rag/index_manager.py:_do_load() (line 268)
      → asyncio.to_thread(pipeline.ensure_initialized()) (line 489)
        → backend/rag_pipeline.py:ensure_initialized() (line 271)
          → _load_index() (line 86)
            → self.orchestrator.initialize_models() (line 110)
              → backend/orchestrator.py:initialize_models() (line 3717)
                → build_offline_embedding() (line 3771)
                  → HuggingFaceEmbedding() constructor (SLOW - loads SentenceTransformer)
                → CrossEncoder() constructor (reranker)
            → self.orchestrator.load_index() (line 118)
              → StorageContext.from_defaults() + load_index_from_storage() (line 4156)
                → Parses default__vector_store.json (183MB) - SLOW
```

**Confirmed:** The hang is most likely in one of these slow operations:
1. `build_offline_embedding()` loading `HuggingFaceEmbedding` (SentenceTransformer load from cache)
2. `CrossEncoder()` loading reranker model
3. `load_index_from_storage()` parsing 183MB `default__vector_store.json`

### 3. Eager Mode Logic
**Location:** `backend/api.py` lines 1054-1091

**Confirmed:** 
- `_get_rag_load_mode()` parses env vars: `RAG_EAGER_LOAD_ON_STARTUP` and `RAG_BACKGROUND_LOAD_ON_STARTUP`
- If eager mode: uses `await asyncio.wait_for(load_state.ensure_loaded(), timeout=eager_timeout)` - **TRUE blocking await**
- Logs mode early: `[RAG] load_mode=eager eager=... background=...` (line 1074-1075)

**Verification:** Check Cloud Run logs for `[RAG] load_mode=` to confirm which mode is active.

### 4. Deployment Configuration Discrepancy
**CI Uses:** `backend/Dockerfile.backend` (`.github/workflows/ci.yml` line 249)
**User Mentioned:** `deployment/Dockerfile.api`

**IMPORTANT:** Verify which Dockerfile is actually used in Cloud Run deployment. If deployment script uses `deployment/Dockerfile.api` but CI builds with `backend/Dockerfile.backend`, the changes to `start-gunicorn.sh` won't be in the deployed image.

### 5. Model Pre-download Status
**Location:** `deployment/Dockerfile.api` lines 53-95

**Confirmed:** Dockerfile includes pre-download steps for:
- `BAAI/bge-large-en-v1.5` (embedding)
- `BAAI/bge-reranker-large` (reranker)

**Verification:** Look for `[RAG] VERIFICATION_MARKER: embedding_model_ready` in build logs (should appear during image build, not runtime).

## Most Likely Root Causes (Ranked)

### #1: SentenceTransformer Model Loading from Cache Takes >60s (HIGHEST PROBABILITY)
**Evidence:**
- Faulthandler dumps show stack in `SentenceTransformer.__init__()`
- 183MB vector store downloads successfully
- Pre-download verification markers exist, but that's during build - runtime cache access might be slow

**Why It Happens:**
- Even with models pre-downloaded, `SentenceTransformer` must:
  1. Read model files from `/app/.cache/huggingface` (I/O bound)
  2. Load weights into memory (CPU bound, ~1.3GB for bge-large-en-v1.5)
  3. Initialize tokenizer, normalization layers, etc.
- On Cloud Run with 4Gi memory, this can take 60-120 seconds
- If CPU is throttled or memory is fragmented, even longer

**Fix Applied:** Increased faulthandler timeout to match Gunicorn timeout (600s), added detailed instrumentation around `build_offline_embedding()`.

### #2: Vector Store JSON Parsing Takes >120s (MODERATE PROBABILITY)
**Evidence:**
- `default__vector_store.json` is 183MB
- Current timeout is 120s (`threading.Thread.join(timeout=120)`, line 4168)
- If parsing exceeds 120s, `orchestrator.load_index()` raises `RuntimeError`

**Why It Happens:**
- JSON parsing of 183MB file requires:
  1. Reading entire file into memory
  2. Parsing JSON structure (CPU bound)
  3. Deserializing embeddings (numpy arrays, ~183MB of float32s)
- On slow CPU or if memory is constrained, can exceed 120s

**Fix Applied:** Added instrumentation around vector store parsing with duration tracking.

### #3: Multiple Workers Restarting Each Other (LOW PROBABILITY)
**Evidence:**
- `GUNICORN_WORKERS=1` is set in deploy script
- But if multiple instances are starting simultaneously, each triggers its own load

**Why It Would Happen:**
- If `min-instances=1` but Cloud Run starts a new instance while old one is still loading
- Or if worker restarts mid-load due to OOM

**Fix Applied:** Added pid/revision/hostname to all logs to track multi-instance behavior.

### #4: Dockerfile Not Actually Used / Wrong Revision Deployed (VERIFICATION NEEDED)
**Evidence:**
- CI uses `backend/Dockerfile.backend`
- User mentioned `deployment/Dockerfile.api`
- If wrong Dockerfile is deployed, `start-gunicorn.sh` changes won't be present

**Fix:** Verify which Dockerfile is actually deployed and ensure changes are in that file.

## Minimal Patch Plan

### Patch 1: Fix Faulthandler Timeout (CRITICAL)
**File:** `backend/api.py` lines 26-33

**Change:** Increase faulthandler timeout from 60s to 600s (configurable via env var)

```python
faulthandler_timeout = int(os.getenv("FAULTHANDLER_TIMEOUT_SEC", os.getenv("GUNICORN_TIMEOUT", "600")))
faulthandler.dump_traceback_later(faulthandler_timeout, repeat=True, file=sys.stderr)
print(f"[FAULTHANDLER] enabled with timeout={faulthandler_timeout}s (will dump traceback if stuck)", flush=True)
```

**Why:** Eliminates repeated "Timeout (0:01:00)!" dumps during legitimate slow loads.

### Patch 2: Add Instrumentation to build_offline_embedding()
**File:** `backend/utils/embedding_utils.py` lines 125-220

**Changes:**
1. Add START/DONE checkpoints with pid/revision/hostname/cache_dir/duration
2. Log at function entry and exit with timing

**Why:** Pinpoints if hang is in embedding model load.

### Patch 3: Add Instrumentation to Reranker Load
**File:** `backend/orchestrator.py` lines 3810-3829

**Changes:**
1. Add START/DONE checkpoints with pid/revision/hostname/cache_dir/duration
2. Wrap in try/except to log failures separately

**Why:** Pinpoints if hang is in reranker load.

### Patch 4: Enhance Vector Store Parse Checkpoints
**File:** `backend/orchestrator.py` lines 4151-4191

**Changes:**
1. Add pid/revision/hostname/file_size to START checkpoint
2. Add duration to DONE checkpoint

**Why:** Pinpoints if hang is in vector store parsing.

## Verification Commands

### 1. Confirm Faulthandler Timeout Is Applied
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"FAULTHANDLER.*enabled"' \
  --project arrow-rag-support-prod \
  --freshness 1h \
  --limit 5 \
  --format 'value(textPayload)'

# Expected: "[FAULTHANDLER] enabled with timeout=600s"
```

### 2. Check Which Load Mode Is Active
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"load_mode="' \
  --project arrow-rag-support-prod \
  --freshness 1h \
  --limit 5 \
  --format 'value(textPayload)'

# Expected: "[RAG] load_mode=eager eager=1 background=0 ..."
```

### 3. Find Where Load Hangs (Embedding Model)
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"build_offline_embedding_START" OR textPayload=~"build_offline_embedding_DONE")' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 20 \
  --format 'table(timestamp,textPayload)'

# Expected: See START followed by DONE (if DONE is missing, hang is in embedding load)
```

### 4. Find Where Load Hangs (Reranker)
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"reranker_load_START" OR textPayload=~"reranker_load_DONE")' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 20 \
  --format 'table(timestamp,textPayload)'

# Expected: See START followed by DONE (if DONE is missing, hang is in reranker load)
```

### 5. Find Where Load Hangs (Vector Store Parse)
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"vector_store_parse_START" OR textPayload=~"vector_store_parse_DONE")' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 20 \
  --format 'table(timestamp,textPayload)'

# Expected: See START followed by DONE (if DONE is missing, hang is in vector store parse)
```

### 6. Verify No More "Timeout (0:01:00)!" After Patch
```bash
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND textPayload=~"Timeout.*0:01:00"' \
  --project arrow-rag-support-prod \
  --freshness 2h \
  --limit 10

# Expected: Zero results after patch (or timeout should be "Timeout (0:10:00)!")
```

### 7. Confirm Model Cache Directory Is Writable
```bash
# This must be checked inside container at runtime
# Add a one-time log in build_offline_embedding() showing:
# - cache_dir path
# - whether directory exists
# - whether directory is writable (os.access(cache_dir, os.W_OK))
# - file count in cache_dir
```

### 8. Verify Which Dockerfile Is Deployed
```bash
# Check Cloud Run service revision metadata
gcloud run revisions describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="yaml(spec.containers[0].image)"

# Then check the image to see which CMD it uses:
gcloud artifacts docker images describe \
  us-central1-docker.pkg.dev/arrow-rag-support-prod/arrow-rag-backend/backend:LATEST \
  --format="yaml"

# Or check if start-gunicorn.sh exists in the image
docker run --rm --entrypoint ls \
  us-central1-docker.pkg.dev/arrow-rag-support-prod/arrow-rag-backend/backend:LATEST \
  /start-gunicorn.sh
```

## Expected Log Sequence (After Patch)

If load succeeds, you should see:

```
[FAULTHANDLER] enabled with timeout=600s
[APP_START] pid=X hostname=... GUNICORN_TIMEOUT=600 ...
[RAG] load_mode=eager eager=1 background=0 ...
[RAG] eager load begin (timeout=600s)
[RAG] build_offline_embedding_START pid=X hostname=... revision=... model=BAAI/bge-large-en-v1.5 cache_dir=/app/.cache/huggingface
[RAG] embedding_import_begin model=BAAI/bge-large-en-v1.5
[RAG] embedding_import_done
[RAG] embedding_load_begin model=BAAI/bge-large-en-v1.5 ...
[RAG] embedding_load_done model=BAAI/bge-large-en-v1.5 duration=45.23s
[RAG] build_offline_embedding_DONE pid=X ... duration=45.23s
[RAG] reranker_load_START pid=X hostname=... revision=... cache_dir=/app/.cache/huggingface
[RAG] reranker_model_load_done duration=12.45s
[RAG] reranker_load_DONE pid=X ... duration=12.45s
[RAG] vector_store_parse_START pid=X hostname=... revision=... storage_dir=/tmp/latest_model vector_store_size_bytes=183142647
[RAG] vector_store_parse_DONE pid=X ... duration=67.89s
[RAG] eager load completed; READY (duration=125.57s)
```

If load hangs, you'll see START but no corresponding DONE, pinpointing the exact stage.

## Next Steps After Verification

1. **If hang is in embedding model load:**
   - Verify model cache directory exists and is writable
   - Check if models are actually present (list files in cache_dir)
   - Consider using `local_files_only=True` explicitly
   - Check Cloud Run CPU/memory metrics during load

2. **If hang is in reranker load:**
   - Similar checks as embedding model
   - Consider making reranker optional if it's blocking

3. **If hang is in vector store parse:**
   - Increase timeout in `orchestrator.load_index()` (currently 120s)
   - Check file size matches expected (183MB)
   - Verify JSON is valid before parsing
   - Consider streaming/chunked parsing for large files

4. **If no hang but still never reaches READY:**
   - Check if `rag_state["ready"]` is set correctly after load
   - Verify `index_state.set_phase("ready")` is called
   - Check for exceptions being swallowed

