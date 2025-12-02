# RAG Startup Behavior - Graceful Degradation

## Overview

The backend now tolerates missing or invalid RAG index during startup. The service will start successfully even when the RAG index is not mounted or not uploaded, allowing non-RAG endpoints (like `/auth/login`, `/health`) to function normally while RAG endpoints (like `/query`) return appropriate 503 errors.

## Changes Made

### 1. `backend/api.py` - Lifespan Function

**Before:** The lifespan function raised `RuntimeError` in production when RAG initialization failed, causing the entire service to crash and return 503 for all endpoints.

**After:** 
- RAG initialization is wrapped in try/except
- On failure, errors are logged with full context but no exception is raised
- `app.state.rag_enabled` flag is set to `False` when RAG fails to initialize
- Server continues startup normally, allowing non-RAG routes to work

**Key Changes:**
- Removed all `raise RuntimeError(...)` calls related to RAG initialization failures
- Added comprehensive error logging with context (error message, storage path, error type)
- Added `app.state.rag_enabled` flag to track RAG availability
- Logs clear messages about what endpoints will work vs. return 503

### 2. `backend/utils/storage_path.py` - Storage Path Resolution

**Before:** In production, `resolve_storage_path()` raised `RuntimeError` when the RAG index was not found.

**After:**
- Returns `None` in all environments (production and dev/test) when index is not found
- Allows the caller (lifespan function) to handle missing index gracefully
- No exceptions raised - graceful degradation is the default behavior

### 3. RAG Endpoints

All RAG endpoints check for `rag_pipeline.is_initialized()` and return structured error responses:
- `/query` - Returns 503 with structured error:
  ```json
  {
    "detail": "RAG pipeline not initialized. Please contact the administrator.",
    "code": "RAG_NOT_INITIALIZED",
    "rag_enabled": false
  }
  ```
- Other RAG endpoints use the same pattern via `get_rag_disabled_response()` helper
- The `code: "RAG_NOT_INITIALIZED"` field allows the frontend to distinguish RAG-disabled errors from transient 503 errors (e.g., cold starts)
- Structured logging is added for all RAG-disabled 503 responses

### 4. Health Endpoint

The `/health` endpoint already handles RAG disabled correctly:
- Returns `rag_pipeline_initialized: false` when RAG is disabled
- Does NOT fail the health check when RAG is disabled
- Service is considered healthy as long as database is connected

## Behavior in Different Scenarios

### Scenario 1: RAG Index Present and Valid

**Startup:**
- `resolve_storage_path()` returns path to index
- RAG pipeline initializes successfully
- `app.state.rag_enabled = True`
- Log: "Server started with RAG pipeline enabled"

**Endpoints:**
- All endpoints work normally
- `/query` and other RAG endpoints function correctly
- `/health` returns `rag_pipeline_initialized: true`

### Scenario 2: RAG Index Missing (Production)

**Startup:**
- `resolve_storage_path()` returns `None`
- RAG pipeline initialization is attempted but fails
- Exception is caught, logged with full context
- `app.state.rag_enabled = False`
- Server continues startup normally
- Log: "Server started without RAG pipeline. Non-RAG endpoints are functional. RAG endpoints will return 503."

**Endpoints:**
- `/auth/login` - ✅ Works normally
- `/health` - ✅ Returns `rag_pipeline_initialized: false`, status: "ok"
- `/rag/status` - ✅ Returns `rag_enabled: false` with details
- `/query` - ❌ Returns 503 with structured error:
  ```json
  {
    "detail": "RAG pipeline not initialized. Please contact the administrator.",
    "code": "RAG_NOT_INITIALIZED",
    "rag_enabled": false
  }
  ```
- Other RAG endpoints - ❌ Return 503 with same structured format

### Scenario 3: RAG Index Invalid or Corrupted

**Startup:**
- `resolve_storage_path()` may return a path, but index loading fails
- Exception during `initialize_rag_pipeline()` is caught
- Error is logged with full context (error message, storage path, error type)
- `app.state.rag_enabled = False`
- Server continues startup normally

**Endpoints:**
- Same behavior as Scenario 2

## Log Messages

### Successful RAG Initialization
```
INFO: rag_pipeline_initialized storage_path=/app/latest_model cache_dir=/app/.cache/huggingface
INFO: server_started_with_rag message="Server started with RAG pipeline enabled"
```

### Failed RAG Initialization (Index Missing)
```
WARNING: rag_index_not_found message="RAG index not found. Continuing startup without RAG. Non-RAG endpoints (e.g., /auth/login, /health) will work normally. RAG endpoints (e.g., /query) will return 503."
ERROR: rag_pipeline_init_failed error="..." error_type="RuntimeError" storage_path=latest_model
WARNING: rag_init_failed_continuing message="RAG pipeline initialization failed. Server will start normally, but RAG endpoints (e.g., /query) will return 503. Non-RAG endpoints (e.g., /auth/login, /health) will work normally. Error: ..."
INFO: server_started_without_rag message="Server started without RAG pipeline. Non-RAG endpoints are functional. RAG endpoints will return 503."
```

## Testing

### Test 1: Backend Starts Without RAG Index
1. Deploy backend without mounting `/app/latest_model` volume
2. Verify backend starts successfully (no 503 on startup)
3. Verify `/auth/login` works
4. Verify `/health` returns status "ok" with `rag_pipeline_initialized: false`
5. Verify `/query` returns 503

### Test 2: Backend Starts With RAG Index
1. Deploy backend with `/app/latest_model` volume mounted and index present
2. Verify backend starts successfully
3. Verify `/health` returns `rag_pipeline_initialized: true`
4. Verify `/query` works normally

## Migration Notes

- **No breaking changes**: All existing endpoints maintain their behavior
- **Backward compatible**: When RAG index is present, behavior is identical to before
- **Improved resilience**: Service no longer crashes due to missing RAG index
- **Better user experience**: Login and other critical endpoints work even when RAG is unavailable

## Related Issues

This fix addresses the issue where:
- Backend crashed on startup in production when RAG index was missing
- `/auth/login` returned 503 because the entire service failed to start
- Cloud Run marked the revision as unhealthy, preventing traffic

Now:
- Backend starts successfully even without RAG index
- `/auth/login` works normally
- Only RAG-specific endpoints return 503 when RAG is unavailable
- Cloud Run marks the service as healthy (database connectivity is the primary health check)
- Frontend can distinguish RAG-disabled 503s from transient errors and handle them appropriately (no retries, clear messaging)

## Structured Error Responses

### RAG-Disabled Error Format

When RAG endpoints are called but RAG is not initialized, they return:

```json
{
  "detail": "RAG pipeline not initialized. Please contact the administrator.",
  "code": "RAG_NOT_INITIALIZED",
  "rag_enabled": false
}
```

**Key Fields:**
- `code: "RAG_NOT_INITIALIZED"` - Allows frontend to reliably detect RAG-disabled condition
- `rag_enabled: false` - Explicit flag indicating RAG is not available
- `detail` - User-facing error message

### Frontend Handling

The frontend (`frontend/lib/iam-backend.ts` and `frontend/app/api/query/route.ts`) now:

1. **Checks for `code: "RAG_NOT_INITIALIZED"`** in 503 responses
2. **Skips retries** when RAG is disabled (no exponential backoff)
3. **Shows clear user message**: "Document search is currently unavailable because the RAG index is not loaded. Please contact your administrator."
4. **Retries transient 503s** (e.g., cold starts) with exponential backoff

This prevents:
- Pointless retries when RAG is permanently disabled
- Confusing "Service temporarily unavailable" messages for permanent RAG issues
- Wasted backend requests

## New Endpoints

### GET /rag/status

Returns the current RAG pipeline status without requiring authentication.

**Response (RAG enabled):**
```json
{
  "rag_enabled": true,
  "mode": "local_index",
  "details": "RAG pipeline initialized and ready."
}
```

**Response (RAG disabled):**
```json
{
  "rag_enabled": false,
  "mode": "disabled",
  "details": "RAG index not loaded. Non-RAG endpoints are functional."
}
```

**Response Fields:**
- `rag_enabled`: Boolean indicating if RAG is available
- `mode`: One of:
  - `"local_index"`: RAG is using a local index (current implementation)
  - `"vector_db"`: RAG is using an external vector database (future)
  - `"disabled"`: RAG is not available
- `details`: Human-readable status message

**Use Cases:**
- Frontend can check RAG status after login to enable/disable query UI
- Show banner/warning when RAG is disabled
- Monitoring and health dashboards
- Pre-flight checks before attempting queries

**Frontend Integration:**
- Frontend calls `/api/rag/status` (Next.js API route)
- API route proxies to backend `/rag/status` with IAM authentication
- ChatInterface checks status on mount and shows banner if disabled
- Query input is disabled when RAG is disabled to prevent spam

## Logging

### Startup Logging

At server startup, RAG initialization is comprehensively logged:

**RAG Initialization Start:**
```
INFO: rag_init_starting message="Starting RAG pipeline initialization"
INFO: rag_storage_path_resolved storage_path=/app/latest_model
INFO: rag_index_files_present storage_path=/app/latest_model message="All required index files found"
INFO: rag_storage_directory_contents storage_path=/app/latest_model file_count=5 files=[...]
```

**RAG Enabled (Success):**
```
INFO: orchestrator_load_index_starting storage_dir=/app/latest_model
INFO: orchestrator_index_check storage_dir=/app/latest_model directory_exists=True docstore_exists=True index_exists=True
INFO: orchestrator_loading_index storage_dir=/app/latest_model message="🔄 Loading index from storage..."
INFO: orchestrator_index_loaded storage_dir=/app/latest_model index_type=VectorStoreIndex message="Index loaded successfully from storage"
INFO: index_and_retriever_initialized storage_dir=/app/latest_model message="✅ Index and retriever initialized successfully"
INFO: rag_pipeline_initialized storage_path=/app/latest_model cache_dir=/app/.cache/huggingface message="RAG pipeline successfully initialized and ready for queries"
INFO: server_started_with_rag message="Server started with RAG pipeline enabled" rag_mode="local_index" rag_enabled=True
```

**RAG Disabled - Index Not Found:**
```
INFO: rag_init_starting message="Starting RAG pipeline initialization"
WARNING: rag_storage_path_resolution_failed message="resolve_storage_path() returned None - no valid index directory found"
WARNING: rag_index_not_found message="RAG index not found. Continuing startup without RAG..."
WARNING: rag_pipeline_initialized_without_index storage_path=latest_model message="Pipeline initialized but index not loaded..."
INFO: server_started_without_rag message="Server started without RAG pipeline..." rag_mode="disabled" rag_enabled=False
```

**RAG Disabled - Directory Exists But Files Missing:**
```
INFO: rag_init_starting message="Starting RAG pipeline initialization"
INFO: rag_storage_path_resolved storage_path=/app/latest_model
WARNING: rag_index_files_missing storage_path=/app/latest_model missing_files=['docstore.json', 'default__vector_store.json'] message="Index directory exists but missing required files..."
WARNING: orchestrator_index_docstore_missing storage_dir=/app/latest_model docstore_path=/app/latest_model/docstore.json message="Index directory exists but docstore.json is missing"
INFO: orchestrator_index_directory_contents storage_dir=/app/latest_model file_count=0 files=[] message="Directory exists with 0 items"
WARNING: index_not_found_ingestion_disabled storage_dir=/app/latest_model directory_exists=True docstore_exists=False message="Index not found but ingestion is disabled (Cloud Run)..."
INFO: orchestrator_load_index_aborted storage_dir=/app/latest_model reason="Index not found and ingestion disabled"
WARNING: rag_pipeline_initialized_without_index storage_path=/app/latest_model message="Pipeline initialized but index not loaded..."
INFO: server_started_without_rag rag_mode="disabled" rag_enabled=False
```

**RAG Disabled - Exception During Load:**
```
INFO: rag_init_starting message="Starting RAG pipeline initialization"
INFO: rag_storage_path_resolved storage_path=/app/latest_model
INFO: rag_index_files_present storage_path=/app/latest_model message="All required index files found"
INFO: orchestrator_loading_index storage_dir=/app/latest_model message="🔄 Loading index from storage..."
ERROR: orchestrator_index_load_failed storage_dir=/app/latest_model error="..." error_type="..." message="Failed to load index from /app/latest_model: ..."
ERROR: rag_pipeline_init_failed error="..." error_type="..." storage_path=/app/latest_model
ERROR: rag_init_exception_debug storage_path=/app/latest_model directory_exists=True file_count=5 files=[...] message="Exception occurred during RAG init. Directory exists with files listed above."
WARNING: rag_init_failed_continuing message="RAG pipeline initialization failed with exception..."
INFO: server_started_without_rag rag_mode="disabled" rag_enabled=False
```

### RAG-Disabled Query Attempts

When `/query` is called but RAG is disabled, the backend logs:

```
WARNING: rag_query_rejected_rag_disabled path=/query reason="RAG pipeline not initialized" rag_enabled=False
```

This structured log allows:
- Easy filtering in Cloud Logging
- Monitoring of RAG-disabled query attempts
- Distinguishing from other 503 errors in logs

## Interpreting RAG Status

### When RAG is Disabled

RAG will remain disabled until:
1. **Index is built locally**: Run ingestion to create `latest_model/` directory with required files:
   - `docstore.json`
   - `default__vector_store.json`
   - `index_store.json`
   - `graph_store.json` (optional)
   - `image__vector_store.json` (optional)
2. **Index is uploaded to GCS bucket root**: 
   ```bash
   gsutil -m rsync -r latest_model/ gs://arrow-rag-support-prod-rag/
   ```
   Or use the upload script:
   ```bash
   python -m backend.scripts.upload_index_to_gcs --dir latest_model --bucket arrow-rag-support-prod-rag
   ```
3. **Cloud Run volume is mounted**: Ensure Cloud Run service has volume mount:
   - GCS bucket: `arrow-rag-support-prod-rag`
   - Mount path: `/app/latest_model` (bucket root mounts to this path)
   - Files at `gs://arrow-rag-support-prod-rag/docstore.json` will appear at `/app/latest_model/docstore.json`

### Checking RAG Status

- **Backend**: Call `GET /rag/status` (returns JSON with `rag_enabled`, `mode`, `details`)
- **Frontend**: Call `/api/rag/status` (proxies to backend)
- **Logs**: Check startup logs for detailed initialization flow:
  - Look for `rag_init_starting` to see initialization began
  - Check `rag_storage_path_resolved` to see what path was used
  - Check `rag_index_files_present` or `rag_index_files_missing` to see file status
  - Check `orchestrator_index_check` to see directory/file existence
  - Check `rag_pipeline_initialized` for success or `rag_pipeline_initialized_without_index` for failure
  - Final status: `server_started_with_rag` (enabled) or `server_started_without_rag` (disabled)
- **Health endpoint**: `/health` includes `rag_pipeline_initialized` field

### Debugging RAG Initialization Failures

When RAG fails to initialize, check logs for:

1. **Storage Path Resolution**:
   - `rag_storage_path_resolved` - Shows the path that was resolved
   - `rag_storage_path_resolution_failed` - Path resolution returned None

2. **Directory and File Checks**:
   - `rag_storage_directory_missing` - Directory doesn't exist
   - `rag_index_files_missing` - Directory exists but files are missing
   - `rag_storage_directory_contents` - Lists files in directory for debugging

3. **Orchestrator Checks**:
   - `orchestrator_index_check` - Shows directory_exists, docstore_exists, index_exists flags
   - `orchestrator_index_directory_contents` - Lists directory contents when index not found
   - `orchestrator_index_load_failed` - Exception during index loading

4. **Final Status**:
   - `rag_pipeline_initialized` - Success
   - `rag_pipeline_initialized_without_index` - Failure (with debug info)
   - `rag_init_exception_debug` - Exception occurred (with directory contents)

### Common Failure Scenarios

**Scenario 1: Index directory doesn't exist**
- Logs: `rag_storage_directory_missing`, `orchestrator_index_directory_missing`
- Fix: Upload index to GCS and ensure volume mount is configured

**Scenario 2: Directory exists but is empty**
- Logs: `rag_index_files_missing`, `orchestrator_index_directory_contents` shows empty files=[]
- Fix: Ensure index files were uploaded correctly to GCS

**Scenario 3: Directory exists but docstore.json missing**
- Logs: `orchestrator_index_docstore_missing`, `orchestrator_index_directory_contents` shows files but no docstore.json
- Fix: Re-upload index, ensuring all required files are included

**Scenario 4: Files exist but index load fails**
- Logs: `orchestrator_index_load_failed` with error details
- Fix: Check index file integrity, may need to rebuild index

