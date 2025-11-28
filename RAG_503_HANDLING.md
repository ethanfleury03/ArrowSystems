# RAG 503 Error Handling - Implementation Summary

## Problem

The `/query` endpoint was returning 503 errors when RAG was disabled, but the frontend was treating all 503s as transient errors and retrying with exponential backoff. This caused:
- Pointless retries when RAG was permanently disabled
- Confusing "Service temporarily unavailable" messages
- Wasted backend requests

## Solution

### Backend Changes

#### 1. Structured Error Responses (`backend/api.py`)

**Added helper function:**
```python
def get_rag_disabled_response(path: str = "/query") -> JSONResponse:
    """Return structured 503 response for RAG-disabled endpoints."""
```

**Updated `/query` endpoint:**
- Returns structured JSON with `code: "RAG_NOT_INITIALIZED"` and `rag_enabled: false`
- Logs structured warning: `rag_query_rejected_rag_disabled`
- Allows frontend to distinguish from transient 503s

**Error response format:**
```json
{
  "detail": "RAG pipeline not initialized. Please contact the administrator.",
  "code": "RAG_NOT_INITIALIZED",
  "rag_enabled": false
}
```

#### 2. New Status Endpoint (`backend/api.py`)

**Added `GET /rag/status`:**
- Returns RAG availability status
- No authentication required
- Useful for pre-flight checks

**Response models:**
- `RAGStatusResponse` with `rag_enabled: bool` and `details: str`

### Frontend Changes

#### 1. Retry Logic (`frontend/lib/iam-backend.ts`)

**Updated `iamBackendRequest()`:**
- Checks response body for `code: "RAG_NOT_INITIALIZED"` before retrying
- Skips retries for RAG-disabled errors
- Continues retrying transient 503s (cold starts, network issues)

**Key changes:**
- Parse response data to check for `code` field
- Set `isRagDisabled` flag when detected
- Skip retry loop if RAG-disabled
- Handle in both try block (successful response) and catch block (error response)

#### 2. Query Route (`frontend/app/api/query/route.ts`)

**Updated error handling:**
- Detects `code: "RAG_NOT_INITIALIZED"` in error responses
- Returns user-friendly message: "Document search is currently unavailable because the RAG index is not loaded. Please contact your administrator."
- Preserves error code and `rag_enabled` flag for client-side handling

## Behavior Matrix

| Scenario | Backend Response | Frontend Behavior |
|----------|-----------------|-------------------|
| RAG disabled | 503 with `code: "RAG_NOT_INITIALIZED"` | No retries, show clear message |
| Cold start | 503 without `code` field | Retry with exponential backoff (2-3 attempts) |
| Network error | Connection refused/timeout | Retry with exponential backoff |
| RAG enabled | 200 with query results | Normal processing |

## Files Changed

### Backend
- `backend/api.py`
  - Added `get_rag_disabled_response()` helper
  - Updated `/query` endpoint to use structured errors
  - Added `GET /rag/status` endpoint
  - Added `RAGStatusResponse` model

### Frontend
- `frontend/lib/iam-backend.ts`
  - Updated retry logic to check for `RAG_NOT_INITIALIZED`
  - Skip retries for RAG-disabled errors
- `frontend/app/api/query/route.ts`
  - Updated error handling to detect RAG-disabled errors
  - Return user-friendly messages

## Testing

### Test 1: RAG Disabled
1. Deploy backend without RAG index
2. Call `/query` from frontend
3. **Expected:** Single 503 response (no retries), clear error message
4. **Logs:** `rag_query_rejected_rag_disabled` warning

### Test 2: RAG Enabled
1. Deploy backend with RAG index
2. Call `/query` from frontend
3. **Expected:** Normal query processing, 200 response

### Test 3: Cold Start
1. Scale backend to 0 instances
2. Call `/query` (triggers cold start)
3. **Expected:** Retries with exponential backoff, eventually succeeds or shows transient error

### Test 4: Status Endpoint
1. Call `GET /rag/status`
2. **Expected:** Returns `rag_enabled: true/false` with details

## Benefits

1. **Better UX:** Clear messages for permanent vs transient errors
2. **Reduced load:** No pointless retries when RAG is disabled
3. **Easier debugging:** Structured logs and error codes
4. **Frontend flexibility:** Can check `/rag/status` to enable/disable query UI

## Future Enhancements

- Frontend could call `/rag/status` after login to conditionally enable query UI
- Admin dashboard could show RAG status
- Alerting could monitor `rag_query_rejected_rag_disabled` logs

