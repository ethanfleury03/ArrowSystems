# Finding Your Google Cloud Configuration in Console
# Step-by-step guide to locate all your existing settings

## 1. Find Your Project ID
**Location**: Top-left corner of Google Cloud Console
- Look for the project selector dropdown
- Your Project ID is shown (e.g., "my-rag-app-123456")
- Note this down - you'll need it for all config files

## 2. Check Your Region/Zone
**Location**: 
- Go to **Compute Engine** → **VM instances** (if you have any)
- Or go to **Cloud Run** → **Services** → Click your service
- Look for "Region" field (e.g., "us-central1", "us-east1")

## 3. Find Your Storage Buckets
**Location**: **Cloud Storage** → **Buckets**
- You'll see all your buckets listed
- Common names might be:
  - `your-project-id-rag-data`
  - `your-project-id-rag-models` 
  - `your-project-id-rag-logs`
  - Or custom names you chose
- Note down the exact bucket names

## 4. Check Your Cloud Run Services
**Location**: **Cloud Run** → **Services**
- Look for existing services
- Note the service name (e.g., "rag-app", "my-rag-service")
- Check the configuration:
  - Memory allocation
  - CPU allocation
  - Min/max instances
  - Port (usually 8501 for Streamlit)

## 5. Find Your Secrets
**Location**: **Security** → **Secret Manager**
- Look for secrets like:
  - `anthropic-api-key`
  - `openai-api-key`
  - `aws-access-key`
  - `aws-secret-key`
- Note the exact secret names

## 6. Check Your APIs
**Location**: **APIs & Services** → **Enabled APIs**
- Look for these enabled APIs:
  - Cloud Build API
  - Cloud Run API
  - Container Registry API
  - Cloud Storage API
  - Secret Manager API

## 7. Quick Checklist
Copy this template and fill it out as you find each item:

```
Project ID: ________________
Region: ________________
Zone: ________________
Service Name: ________________
Data Bucket: ________________
Models Bucket: ________________
Logs Bucket: ________________
Anthropic Secret: ________________
OpenAI Secret: ________________
AWS Access Secret: ________________
AWS Secret Key: ________________
```

## 8. Alternative: Use Cloud Shell
If you have Cloud Shell enabled:
1. Click the Cloud Shell icon (terminal icon) in top-right
2. Run these commands to get your info:

```bash
# Get project info
gcloud config get-value project
gcloud config get-value compute/region
gcloud config get-value compute/zone

# List buckets
gsutil ls

# List Cloud Run services
gcloud run services list

# List secrets
gcloud secrets list
```

## 9. What to Look For
- **Project ID**: Usually looks like "my-app-123456" or "rag-app-789012"
- **Region**: Common ones are "us-central1", "us-east1", "europe-west1"
- **Buckets**: Look for names containing "rag", "data", "models", "storage"
- **Secrets**: Look for names with "api", "key", "secret"
- **Services**: Look for names like "rag-app", "my-app", or similar

Once you have all this info, I can update all the configuration files to match your exact setup!
