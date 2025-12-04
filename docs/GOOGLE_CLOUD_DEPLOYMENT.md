# Google Cloud Deployment Guide for RAG Application
# Complete setup and deployment instructions

## Prerequisites

1. **Google Cloud Account** with billing enabled
2. **Google Cloud SDK** installed (`gcloud` command)
3. **Docker** installed locally
4. **API Keys** for:
   - Anthropic (Claude)
   - OpenAI
   - Google Cloud SQL (PostgreSQL)

## Step 1: Initial Setup

### 1.1 Create a Google Cloud Project
```bash
# Create new project (replace with your project name)
gcloud projects create your-project-id --name="RAG App"

# Set as active project
gcloud config set project your-project-id

# Enable billing (required for Cloud Run)
# Go to: https://console.cloud.google.com/billing
```

### 1.2 Authenticate with Google Cloud
```bash
# Login to Google Cloud
gcloud auth login

# Set default project
gcloud config set project your-project-id

# Enable required APIs
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
gcloud services enable storage.googleapis.com
```

## Step 2: Configure Secrets

### 2.1 Create Secret Manager Secrets
```bash
# Set your API keys as secrets
echo "your-anthropic-api-key" | gcloud secrets create anthropic-api-key --data-file=-
echo "your-openai-api-key" | gcloud secrets create openai-api-key --data-file=-
echo "your-aws-access-key" | gcloud secrets create aws-access-key --data-file=-
echo "your-aws-secret-key" | gcloud secrets create aws-secret-key --data-file=-
```

### 2.2 Grant Cloud Run access to secrets
```bash
# Get your project number
PROJECT_NUMBER=$(gcloud projects describe your-project-id --format="value(projectNumber)")

# Grant access to secrets
gcloud secrets add-iam-policy-binding anthropic-api-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding openai-api-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding aws-access-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"

gcloud secrets add-iam-policy-binding aws-secret-key \
    --member="serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com" \
    --role="roles/secretmanager.secretAccessor"
```

## Step 3: Setup Storage

### 3.1 Create GCS Buckets
```bash
# Run the setup script
chmod +x scripts/setup-gcs-buckets.sh
export PROJECT_ID=your-project-id
./scripts/setup-gcs-buckets.sh
```

### 3.2 Upload Your Data
```bash
# Upload your PDFs and documents
gsutil -m cp -r ./data/* gs://your-project-id-rag-data/

# Upload your existing model (if you have one)
gsutil -m cp -r ./latest_model/* gs://your-project-id-rag-models/
```

## Step 4: Deploy Application

### 4.1 Update Configuration Files
1. Edit `cloudbuild.yaml` and replace `$PROJECT_ID` with your actual project ID
2. Edit `deployment/cloud-run-service.yaml` and replace `PROJECT_ID` with your project ID
3. Edit `deploy-gcp.sh` and set `PROJECT_ID="your-project-id"`

### 4.2 Deploy
```bash
# Make deployment script executable
chmod +x deploy-gcp.sh

# Run deployment
./deploy-gcp.sh
```

## Step 5: Verify Deployment

### 5.1 Check Service Status
```bash
# Get service URL
gcloud run services describe rag-app --region=us-central1 --format="value(status.url)"

# Check service logs
gcloud run services logs tail rag-app --region=us-central1
```

### 5.2 Test Application
1. Open the service URL in your browser
2. Test login with: `admin` / `admin123`
3. Try a sample query to verify RAG functionality

## Step 6: Production Optimizations

### 6.1 Custom Domain (Optional)
```bash
# Map custom domain
gcloud run domain-mappings create --service=rag-app --domain=your-domain.com --region=us-central1
```

### 6.2 SSL Certificate
```bash
# Create managed SSL certificate
gcloud compute ssl-certificates create rag-app-ssl \
    --domains=your-domain.com \
    --global
```

### 6.3 Load Balancer (Optional)
```bash
# Create load balancer for high availability
gcloud compute url-maps create rag-app-lb \
    --default-service=rag-app-backend
```

## Monitoring and Maintenance

### View Logs
```bash
# Real-time logs
gcloud run services logs tail rag-app --region=us-central1

# Historical logs
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=rag-app"
```

### Update Application
```bash
# Deploy new version
gcloud builds submit --config cloudbuild.yaml .

# Or use the deployment script
./deploy-gcp.sh
```

### Scale Service
```bash
# Update service configuration
# Note: Memory increased to 8Gi to prevent OOM kills when loading BGE-large embedding model + 350MB dense vector index
gcloud run services update rag-app \
    --region=us-central1 \
    --min-instances=1 \
    --max-instances=20 \
    --memory=8Gi \
    --cpu=2
```

## Troubleshooting

### Common Issues

1. **Build Timeout**: Increase timeout in `cloudbuild.yaml`
2. **Memory Issues**: Increase memory allocation in Cloud Run service
3. **API Key Errors**: Verify secrets are properly configured
4. **Storage Access**: Ensure Cloud Run has access to GCS buckets

### Debug Commands
```bash
# Check build logs
gcloud builds list --limit=5

# Check service configuration
gcloud run services describe rag-app --region=us-central1

# Check IAM permissions
gcloud projects get-iam-policy your-project-id
```

## Cost Optimization

1. **Set min-instances to 0** for development
2. **Use appropriate memory/CPU** allocation
3. **Enable request-based scaling**
4. **Monitor usage** in Google Cloud Console

## Security Best Practices

1. **Use Secret Manager** for API keys
2. **Enable IAM** for fine-grained access control
3. **Use HTTPS** for all communications
4. **Regular security updates** for base images
5. **Monitor access logs** for suspicious activity
