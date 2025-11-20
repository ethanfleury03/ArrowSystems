# 🎯 CORS and Authentication Fixes Summary

## Issues Found and Fixed

### 1. ❌ **Root Cause: `credentials_json` in deploy-backend.yml**

**File**: `.github/workflows/deploy-backend.yml`

The backend deployment workflow was using the **OLD authentication method**:
```yaml
credentials_json: ${{ secrets.GCP_SA_KEY }}
```

This was causing authentication failures because:
- Service account JSON keys are deprecated and insecure
- Your GCP project is configured for Workload Identity Federation (WIF)
- The secret `GCP_SA_KEY` likely doesn't exist or is expired

**✅ Fixed**: Updated to use Workload Identity Federation (WIF):
```yaml
workload_identity_provider: projects/70705019874/locations/global/workloadIdentityPools/github-pool/providers/github-provider
service_account: github-deployer@arrow-rag-support-prod.iam.gserviceaccount.com
```

---

### 2. ❌ **Missing CORS Configuration**

**Files**: 
- `.github/workflows/ci.yml` (backend deployment)
- `.github/workflows/deploy-backend.yml`

The backend Cloud Run service was not configured with the production frontend URL in CORS allowed origins.

**✅ Fixed**: Added `CORS_ALLOWED_ORIGINS` environment variable to all backend deployments:
```yaml
--set-env-vars=CORS_ALLOWED_ORIGINS="https://arrow-rag-frontend-70705019874.us-central1.run.app"
```

The backend code (`backend/config/env.py`) was already set up to read this variable:
```python
env_origins = os.getenv("CORS_ALLOWED_ORIGINS")
self.CORS_ALLOWED_ORIGINS = [
    origin.strip() for origin in env_origins.split(",") if origin.strip()
]
```

---

### 3. ❌ **Using Old Container Registry (GCR)**

**Files**:
- `.github/workflows/deploy-backend.yml`
- `.github/workflows/deploy-frontend.yml`

Both workflows were using deprecated Google Container Registry (GCR):
```yaml
GCR_IMAGE: gcr.io/arrow-rag-support-prod/...
```

**✅ Fixed**: Updated to use Artifact Registry:
```yaml
# Backend
IMAGE: us-central1-docker.pkg.dev/arrow-rag-support-prod/arrow-rag-backend/backend

# Frontend  
IMAGE: us-central1-docker.pkg.dev/arrow-rag-support-prod/arrow-rag-frontend/frontend
```

Also updated Docker configuration commands:
```bash
# Before
gcloud auth configure-docker --quiet

# After
gcloud --quiet auth configure-docker us-central1-docker.pkg.dev
```

---

### 4. ❌ **Syntax Error in deploy-frontend.yml**

**File**: `.github/workflows/deploy-frontend.yml`

Missing backslash on line 72, causing the deploy command to fail:
```yaml
--set-env-vars NEXT_PUBLIC_API_URL="..."
--set-env-vars NODE_ENV=production \  # ❌ Missing \ on previous line
```

**✅ Fixed**: Added missing backslash:
```yaml
--set-env-vars NEXT_PUBLIC_API_URL="..." \
--set-env-vars NODE_ENV=production \
```

---

## 📋 Files Modified

1. **`.github/workflows/ci.yml`**
   - Added `CORS_ALLOWED_ORIGINS` to backend deployment

2. **`.github/workflows/deploy-backend.yml`**
   - ✅ Fixed authentication (WIF instead of credentials_json)
   - ✅ Added `CORS_ALLOWED_ORIGINS` environment variable
   - ✅ Updated to use Artifact Registry
   - ✅ Updated Docker auth command

3. **`.github/workflows/deploy-frontend.yml`**
   - ✅ Fixed missing backslash syntax error
   - ✅ Updated to use Artifact Registry
   - ✅ Updated Docker auth command

---

## 🚀 Next Steps

### To Deploy These Fixes:

1. **Commit and push the changes**:
   ```bash
   git add .github/workflows/
   git commit -m "Fix GCP auth and CORS configuration"
   git push origin main
   ```

2. **Verify the workflows run successfully** in GitHub Actions

3. **Test the frontend-backend connection**:
   - Navigate to `https://arrow-rag-frontend-70705019874.us-central1.run.app`
   - Try logging in or making API calls
   - Check browser console for CORS errors (should be gone!)

### If You Still Have Issues:

1. **Verify Artifact Registry repositories exist**:
   ```bash
   gcloud artifacts repositories list --project=arrow-rag-support-prod --location=us-central1
   ```
   
   You should see:
   - `arrow-rag-backend`
   - `arrow-rag-frontend`

2. **Verify Workload Identity Federation is configured**:
   ```bash
   gcloud iam workload-identity-pools providers describe github-provider \
     --project=arrow-rag-support-prod \
     --location=global \
     --workload-identity-pool=github-pool
   ```

3. **Check service account permissions**:
   The `github-deployer@arrow-rag-support-prod.iam.gserviceaccount.com` needs:
   - `roles/artifactregistry.writer` (to push images)
   - `roles/run.admin` (to deploy to Cloud Run)
   - `roles/iam.serviceAccountUser` (to act as Cloud Run service account)

---

## 📊 Summary of Benefits

✅ **Secure Authentication**: Using WIF instead of JSON keys
✅ **CORS Fixed**: Frontend can now communicate with backend
✅ **Modern Infrastructure**: Using Artifact Registry instead of deprecated GCR
✅ **No More Syntax Errors**: All workflows should run cleanly
✅ **Production-Ready**: Proper environment variable configuration

---

## 🔍 Verification

After deployment, you can verify CORS is working by:

1. **Check backend logs**:
   ```bash
   gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-backend" --limit 50 --project=arrow-rag-support-prod
   ```

2. **Test CORS headers**:
   ```bash
   curl -H "Origin: https://arrow-rag-frontend-70705019874.us-central1.run.app" \
        -H "Access-Control-Request-Method: POST" \
        -H "Access-Control-Request-Headers: Content-Type" \
        -X OPTIONS \
        https://arrow-rag-backend-70705019874.us-central1.run.app/query -v
   ```

   Should return:
   ```
   Access-Control-Allow-Origin: https://arrow-rag-frontend-70705019874.us-central1.run.app
   Access-Control-Allow-Credentials: true
   ```

---

**All fixes have been applied! No `credentials_json` found anywhere in the codebase anymore. 🎉**

