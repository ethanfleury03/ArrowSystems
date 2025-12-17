# CI/CD Deployment Verification

## Current CI Configuration

### GitHub Actions Workflow
**File:** `.github/workflows/ci.yml`

**Deployment Trigger:**
- Runs on push to `main` branch
- Runs on `workflow_dispatch` (manual trigger)

**Deployment Step (Line 636-661):**
```yaml
- name: Deploy backend to Cloud Run
  run: |
    gcloud run deploy arrow-rag-backend \
      --image="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/arrow-rag-backend/backend:${GITHUB_SHA}" \
      ...
```

**Key Points:**
1. ✅ Uses `${GITHUB_SHA}` - deploys exact commit SHA
2. ✅ Should create new revision on each push
3. ⚠️ **Does NOT set `--to-latest` flag** - traffic routing may not update automatically

## Potential Issues

### Issue 1: Traffic Routing Not Updated
**Problem:** Cloud Run deployment creates a new revision but doesn't automatically route traffic to it.

**Current behavior:**
- New revision is created (e.g., `arrow-rag-backend-00187-hid`)
- But traffic may still go to old revision (e.g., `arrow-rag-backend-00186-fkr`)

**Fix needed:** Add `--to-latest` flag to deployment command OR manually update traffic after deployment.

### Issue 2: CI Not Running
**Possible causes:**
- Workflow file not in `.github/workflows/` (it is ✅)
- Workflow syntax error (check GitHub Actions tab)
- Branch protection rules blocking deployment
- Secrets not configured correctly

### Issue 3: Image Not Being Built
**Possible causes:**
- Docker build step failing silently
- Image push failing
- Wrong image tag format

## Verification Steps

### 1. Check if CI is Running
```bash
# Check GitHub Actions runs
# Go to: https://github.com/YOUR_REPO/actions
# Look for recent runs on main branch
```

### 2. Check Cloud Run Revisions
```bash
gcloud run revisions list \
  --service=arrow-rag-backend \
  --region=us-central1 \
  --project=arrow-rag-support-prod \
  --limit=5
```

### 3. Check Traffic Distribution
```bash
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --project=arrow-rag-support-prod \
  --format="value(status.traffic)"
```

### 4. Check Latest Revision Image
```bash
gcloud run revisions describe arrow-rag-backend-<LATEST_REVISION> \
  --region=us-central1 \
  --project=arrow-rag-support-prod \
  --format="value(spec.containers[0].image)"
```

Compare the image SHA with your latest commit SHA.

## Recommended Fixes

### Fix 1: Add `--to-latest` Flag
Update `.github/workflows/ci.yml` line 636-661:

```yaml
- name: Deploy backend to Cloud Run
  run: |
    gcloud run deploy arrow-rag-backend \
      --image="us-central1-docker.pkg.dev/${{ secrets.GCP_PROJECT_ID }}/arrow-rag-backend/backend:${GITHUB_SHA}" \
      --to-latest \  # ADD THIS LINE
      ...
```

### Fix 2: Verify CI is Running
1. Go to GitHub Actions tab
2. Check if workflow runs on push to main
3. Check if deployment step succeeds
4. Check logs for any errors

### Fix 3: Manual Traffic Update (Immediate Fix)
```bash
gcloud run services update-traffic arrow-rag-backend \
  --region=us-central1 \
  --to-latest \
  --project=arrow-rag-support-prod
```

## Current Status

**Uncommitted Changes:**
- `frontend/components/admin/documents-tab.tsx` (updated delete message)

**Recent Commits:**
- Multiple "Fix: Ingestion and CI" commits
- Multiple "Fix: document deleting" commits

**Action Required:**
1. Commit the frontend change
2. Push to main
3. Verify CI runs
4. Check if new revision is created
5. Verify traffic routes to new revision

## Environment Variable Check

**Line 650 in CI workflow:**
```yaml
--update-env-vars="...,ARROW_ALLOW_APP_INGESTION=true,..."
```

⚠️ **ISSUE FOUND:** CI is still setting `ARROW_ALLOW_APP_INGESTION=true` in the deployment!

This is now deprecated and shouldn't be needed, but it's not causing the issue since we removed all gates.

However, we should remove it from the CI workflow to avoid confusion.

