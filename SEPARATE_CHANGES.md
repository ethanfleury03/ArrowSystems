# Separating Production vs Test Mode Changes

The stash contains BOTH production changes AND test mode changes. Here's how to separate them:

## Production Changes (Needed for App to Run)

These are the core ingestion pipeline changes:

### Backend:
- `backend/api.py` - Upload endpoint, delete endpoint, machine model saving
- `backend/utils/chunking_runner.py` - Chunking logic with failure handling
- `backend/utils/embedding_runner.py` - Embedding logic
- `backend/utils/delete_runner.py` - Delete + reindex logic
- `backend/orchestrator.py` - Empty index handling fix

### Frontend:
- `frontend/app/admin/documents/page.tsx` - Status display, polling, delete modal fix
- `frontend/app/api/admin/documents/upload/route.ts` - Machine model forwarding
- `frontend/components/admin/documents-tab.tsx` - Status display (card view)

### Database:
- Migration files (should already exist)

## Test Mode Changes (Optional - Only for Testing)

These are ONLY needed when `TEST_MODE=true`:

### Backend:
- `backend/utils/test_mode.py` - NEW FILE (test mode utilities)
- `backend/api.py` - Test mode path routing, clear test mode endpoint
- `backend/utils/chunking_runner.py` - Uses `get_chunks_dir()` from test_mode
- `backend/utils/embedding_runner.py` - Uses test mode paths
- `backend/utils/delete_runner.py` - Uses test mode paths
- `backend/orchestrator.py` - Test mode index loading

### Frontend:
- `frontend/app/api/admin/test-mode/route.ts` - NEW FILE
- `frontend/app/api/admin/test/clear-test-mode/route.ts` - NEW FILE
- `frontend/app/api/admin/test/mode-status/route.ts` - NEW FILE (if created)
- `frontend/components/admin/documents-tab.tsx` - Test mode badge display

### Config:
- `docker-compose.dev.yml` - TEST_MODE environment variable

## Strategy: Two Separate Stashes

You can create two stashes:

1. **Production stash** - Core ingestion pipeline (Phases 1-4)
2. **Test mode stash** - Test mode functionality

Or keep them together and just skip test mode files when you don't need them.

## Recommendation

**Keep them together** because:
- Test mode is controlled by `TEST_MODE=false` by default
- When `TEST_MODE=false`, test mode code doesn't run
- It's easier to manage one set of changes
- You can always disable test mode by setting `TEST_MODE=false`

The test mode code is **safe to include** - it only activates when `TEST_MODE=true`.






