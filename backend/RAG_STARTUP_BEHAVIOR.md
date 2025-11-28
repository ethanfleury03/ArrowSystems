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

All RAG endpoints already had proper checks for `rag_pipeline.is_initialized()`:
- `/query` - Returns 503 with message "RAG pipeline not initialized. Please contact the administrator."
- `/admin/documents/search` - Returns 503
- `/admin/documents/chunks` - Returns 503
- All other RAG-related endpoints - Return 503 appropriately

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
- `/query` - ❌ Returns 503: "RAG pipeline not initialized. Please contact the administrator."
- Other RAG endpoints - ❌ Return 503

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

