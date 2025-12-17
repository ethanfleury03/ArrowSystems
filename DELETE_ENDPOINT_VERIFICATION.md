# Delete Endpoint Verification

## Current State

### ✅ Main Delete Endpoint: `DELETE /admin/documents/metadata/{metadata_id}`

**Location:** `backend/api.py` line 3930

**Implementation:**
- ✅ Uses `backend.utils.simple_delete.delete_document_metadata_simple()`
- ✅ Does NOT call `delete_runner.py` or `run_delete_and_reindex()`
- ✅ Does NOT check any ingestion flags
- ✅ Returns 204 No Content on success
- ✅ Returns 500 on error (never returns "rebuild disabled" message)

**Code Path:**
```python
@app.delete("/admin/documents/metadata/{metadata_id}")
async def delete_document_by_metadata_id(...):
    # Uses simple_delete - incremental deletion only
    from backend.utils.simple_delete import delete_document_metadata_simple
    delete_result = await run_sync(delete_document_metadata_simple, metadata_id)
    return Response(status_code=204)  # Always succeeds
```

### ✅ Simple Delete Function: `backend/utils/simple_delete.py`

**Implementation:**
- ✅ Does NOT import or call `delete_runner.py`
- ✅ Does NOT call `run_delete_and_reindex()`
- ✅ Does NOT check any ingestion flags
- ✅ Performs incremental deletion from index
- ✅ Never triggers full index rebuild
- ✅ Best-effort deletion (continues even if index cleanup fails)

**What it does:**
1. Deletes chunks from vector index (incremental, by metadata_id)
2. Deletes DocumentIngestionMetadata row
3. Deletes Document row
4. Deletes chunks JSON file
5. Deletes GCS file (best-effort)
6. Deletes local files (best-effort)

**What it does NOT do:**
- ❌ Does NOT rebuild the entire index
- ❌ Does NOT call any rebuild functions
- ❌ Does NOT check `allow_app_ingestion` or any flags
- ❌ Does NOT return "rebuild disabled" messages

### ✅ Frontend Delete Call

**Location:** `frontend/app/admin/documents/page.tsx` line 552

**Implementation:**
- ✅ Calls `/api/admin/documents/metadata/${metadataId}` (correct endpoint)
- ✅ Handles 204 No Content response
- ✅ Shows success message
- ✅ Updated confirmation dialog (removed "requires full index rebuild" message)

## Verification

### Search Results

**No "Index rebuild is disabled" message found in:**
- ✅ `backend/api.py` - delete endpoint
- ✅ `backend/utils/simple_delete.py` - delete function
- ✅ `backend/utils/delete_runner.py` - rebuild function (not called from delete endpoint)

**Frontend message updated:**
- ✅ Removed "This requires a full index rebuild" from confirmation dialog
- ✅ Updated to: "This will permanently delete the document and remove it from the search index"

## If Message Still Appears

If you're still seeing "Index rebuild is disabled in this environment. Document deletion will remove metadata only. Index must be rebuilt via external GPU pipeline.", it's likely from:

1. **Old deployment** - The backend service may be running an older version of the code
   - **Fix:** Deploy the latest code to Cloud Run
   - **Verify:** Check Cloud Run revision matches latest commit

2. **Different endpoint** - The frontend might be calling a different delete endpoint
   - **Check:** Verify frontend is calling `/api/admin/documents/metadata/{metadataId}`
   - **Verify:** Check browser network tab during delete operation

3. **Cached response** - Browser or CDN might be caching old responses
   - **Fix:** Clear browser cache or use incognito mode

## Next Steps

1. ✅ Code is correct - delete endpoint uses simple_delete only
2. ✅ Frontend message updated
3. ⚠️ **Deploy latest code** to ensure old version isn't running
4. ⚠️ **Verify Cloud Run revision** matches latest commit

## Testing

After deployment, test delete operation:
1. Delete a document from admin UI
2. Verify it returns 204 No Content
3. Verify no "rebuild disabled" message appears
4. Verify document is removed from database
5. Verify document is removed from GCS (if applicable)
6. Verify chunks are removed from index (if RAG pipeline available)

