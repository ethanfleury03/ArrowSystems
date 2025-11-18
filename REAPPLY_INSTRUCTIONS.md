# Instructions to Go Back a Commit and Re-apply Changes

## Step 1: Verify Current State
You should have:
- `changes.patch` file (created)
- `CHANGES_SUMMARY.md` file (this document)
- All your changes are uncommitted

## Step 2: Go Back One Commit

Run this command:
```bash
git reset --hard HEAD~1
```

**WARNING**: This will discard all uncommitted changes. But we have the patch file, so it's safe.

## Step 3: Re-apply the Patch

```bash
git apply changes.patch
```

If there are any conflicts or issues, you can apply with more verbose output:
```bash
git apply --verbose changes.patch
```

## Step 4: Create New Files

You'll need to recreate these new files:

1. **backend/utils/test_mode.py** - See CHANGES_SUMMARY.md for contents
2. **frontend/app/api/admin/test-mode/route.ts** - See CHANGES_SUMMARY.md for contents  
3. **frontend/app/api/admin/test/clear-test-mode/route.ts** - See CHANGES_SUMMARY.md for contents

## Step 5: Add Environment Variable

Add to `.env`, `.env.development`, `.env.example`:
```
TEST_MODE=false
```

## Step 6: Verify Everything Works

1. Start your dev environment
2. Test uploading a document
3. Verify status updates in UI
4. Test delete functionality

## Alternative: Stash Instead of Patch

If you prefer, you can use git stash instead:

```bash
# Stash all changes
git stash push -u -m "Ingestion pipeline + test mode changes"

# Go back one commit
git reset --hard HEAD~1

# Re-apply stash
git stash pop
```

This preserves untracked files too.

