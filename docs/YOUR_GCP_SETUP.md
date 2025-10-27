# Google Cloud Setup for ragapp-476414
# Complete setup guide for your specific project

## ✅ What You Already Have
- **Project ID**: `ragapp-476414`
- **Buckets**: 
  - `ragapp-data` (for PDFs and documents)
  - `ragapp-models` (for ML models and indices)  
  - `ragapp-logs` (for application logs)
- **Regions**: Access to all major regions (us-central1 recommended)
- **Database**: PostgreSQL (via Google Cloud SQL or local)

## ❌ What You Need to Set Up

### 1. Enable Required APIs
Go to: https://console.developers.google.com/apis/api/run.googleapis.com/overview?project=ragapp-476414

**Enable these APIs:**
- Cloud Run Admin API ✅ (required)
- Cloud Build API ✅
- Container Registry API ✅
- Secret Manager API ✅

### 2. Create Secrets in Secret Manager
Go to: **Security** → **Secret Manager**

**Create these secrets:**
```bash
# Anthropic API Key
echo "your-anthropic-api-key" | gcloud secrets create anthropic-api-key --data-file=-
```

### 3. Grant Cloud Run Access to Secrets
```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe ragapp-476414 --format="value(projectNumber)")

# Grant access to all secrets
gcloud secrets add-iam-policy-binding anthropic-api-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## 🚀 Ready to Deploy!

### Option 1: Deploy via Google Cloud Console
1. Go to **Cloud Build** → **Triggers**
2. Create a new trigger
3. Connect your GitHub repository
4. Use `cloudbuild.yaml` as the build configuration
5. Set substitution variables:
   - `_PROJECT_ID`: `ragapp-476414`
   - `_REGION`: `us-central1`
   - `_SERVICE_NAME`: `rag-app`

### Option 2: Deploy via Command Line
```bash
# Install Google Cloud SDK first
# Then run:
gcloud builds submit --config cloudbuild.yaml .
```

## 📁 Upload Your Data
```bash
# Upload your PDFs and documents
gsutil -m cp -r ./data/* gs://ragapp-data/

# Upload your existing model (if you have one)
gsutil -m cp -r ./latest_model/* gs://ragapp-models/
```

## 🔧 Configuration Files Updated
All these files now match your project:
- ✅ `cloudbuild.yaml` - Updated with your project ID and bucket names
- ✅ `deploy-gcp.sh` - Updated with your project ID
- ✅ `deployment/cloud-run-service.yaml` - Updated with your bucket names
- ✅ `scripts/setup-gcs-buckets.sh` - Updated with your bucket names

## 🎯 Next Steps
1. **Enable Cloud Run API** (most important!)
2. **Create secret** in Secret Manager (anthropic-api-key)
3. **Upload your data** to the buckets
4. **Deploy** using Cloud Build

## 🆘 Need Help?
- **Cloud Run API**: https://console.developers.google.com/apis/api/run.googleapis.com/overview?project=ragapp-476414
- **Secret Manager**: https://console.cloud.google.com/security/secret-manager?project=ragapp-476414
- **Cloud Build**: https://console.cloud.google.com/cloud-build?project=ragapp-476414
