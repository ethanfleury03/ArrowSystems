# Quick Guide: Go Back and Re-apply Changes

## ✅ Step 1: Changes are Stashed
All your changes have been saved to git stash. You can see them with:
```bash
git stash list
```

## Step 2: Go Back One Commit

Run this command to go back to the previous commit:
```bash
git reset --hard HEAD~1
```

This will reset to commit `b9644d9` (the one before "Added document ingestion admin access")

## Step 3: Re-apply Your Changes

After going back, restore all your changes:
```bash
git stash pop
```

This will restore:
- All modified files
- All new files (test_mode.py, API routes, etc.)
- Everything exactly as it was

## Step 4: Verify

Check that everything is back:
```bash
git status
```

You should see all your modified and new files again.

## Alternative: If Stash Pop Has Conflicts

If there are any conflicts when you `git stash pop`, you can:
1. Resolve conflicts manually
2. Or use the patch file: `git apply changes.patch`
3. Then manually recreate the new files from `CHANGES_SUMMARY.md`

## Files You'll Need to Recreate (if stash doesn't work)

If for some reason the stash doesn't restore new files, you'll need to recreate:
1. `backend/utils/test_mode.py`
2. `frontend/app/api/admin/test-mode/route.ts`
3. `frontend/app/api/admin/test/clear-test-mode/route.ts`

All contents are in `CHANGES_SUMMARY.md`.

## Ready to Go Back?

Run these commands:
```bash
git reset --hard HEAD~1
git stash pop
```

Then verify everything is restored!




