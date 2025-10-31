# GCP Deployment Guide for RAG App

This guide explains how to deploy the RAG app to Google Cloud Platform using Cloud Run.

## Architecture

The app consists of:
- **Backend**: FastAPI on port 8000 (`deployment/Dockerfile.api`)
- **Frontend**: Next.js on port 3000 (`frontend/Dockerfile`)
- **Storage**: RAG index in `latest_model/` directory

## Docker Configuration

### Backend (`deployment/Dockerfile.api`)

**Optimizations for GCP:**
- Uses CPU-only PyTorch (300MB vs 900MB for CUDA)
- Multi-stage install for faster builds
- Proper layer caching
- Health checks for Cloud Run

**Key Features:**
- Detects GPU/CPU automatically at runtime
- Works on Cloud Run (CPU only)
- Can deploy to GKE with GPU nodes if needed

### Frontend (`frontend/Dockerfile`)

**Optimizations:**
- Next.js standalone build (minimal production image)
- Multi-stage build for small final image
- Production-ready settings

## Deployment Options

### Option 1: Cloud Run (Recommended for Backend)

**Backend Only:**
```bash
# Build and push backend image
docker build -f deployment/Dockerfile.api -t gcr.io/YOUR_PROJECT_ID/rag-backend:latest .
docker push gcr.io/YOUR_PROJECT_ID/rag-backend:latest

# Deploy to Cloud Run
gcloud run deploy rag-backend \
  --image gcr.io/YOUR_PROJECT_ID/rag-backend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8000 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 0 \
  --max-instances 10
```

**Frontend (Cloud Run or Cloud Run + Cloud CDN):**
```bash
# Build and push frontend image
docker build -f frontend/Dockerfile -t gcr.io/YOUR_PROJECT_ID/rag-frontend:latest ./frontend
docker push gcr.io/YOUR_PROJECT_ID/rag-frontend:latest

# Deploy to Cloud Run
gcloud run deploy rag-frontend \
  --image gcr.io/YOUR_PROJECT_ID/rag-frontend:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --port 3000 \
  --memory 512Mi \
  --cpu 1 \
  --set-env-vars BACKEND_URL=https://rag-backend-XXXXX.run.app
```

**Note:** Frontend needs `BACKEND_URL` environment variable pointing to your deployed backend.

### Option 2: Cloud Build + Cloud Run (Fully Automated)

Create `cloudbuild.yaml`:
```yaml
steps:
  # Build backend
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'deployment/Dockerfile.api', '-t', 'gcr.io/$PROJECT_ID/rag-backend:$SHORT_SHA', '.']
  
  # Build frontend  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['build', '-f', 'frontend/Dockerfile', '-t', 'gcr.io/$PROJECT_ID/rag-frontend:$SHORT_SHA', './frontend']
  
  # Push both images
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/rag-backend:$SHORT_SHA']
  
  - name: 'gcr.io/cloud-builders/docker'
    args: ['push', 'gcr.io/$PROJECT_ID/rag-frontend:$SHORT_SHA']
  
  # Deploy backend
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
    - 'run'
    - 'deploy'
    - 'rag-backend'
    - '--image=gcr.io/$PROJECT_ID/rag-backend:$SHORT_SHA'
    - '--region=us-central1'
    - '--platform=managed'
  
  # Deploy frontend
  - name: 'gcr.io/cloud-builders/gcloud'
    args:
    - 'run'
    - 'deploy'
    - 'rag-frontend'
    - '--image=gcr.io/$PROJECT_ID/rag-frontend:$SHORT_SHA'
    - '--region=us-central1'
    - '--platform=managed'

images:
  - 'gcr.io/$PROJECT_ID/rag-backend:$SHORT_SHA'
  - 'gcr.io/$PROJECT_ID/rag-frontend:$SHORT_SHA'
```

Then trigger builds:
```bash
gcloud builds submit --config cloudbuild.yaml
```

### Option 3: Artifact Registry + Docker Compose (Alternative)

If you want to keep using Docker Compose on GCP:
1. Push images to Artifact Registry
2. Use GCE with Docker Compose or GKE

## Storage for RAG Index

The `latest_model/` directory contains the vector index. You have options:

### Option A: Build into Image
```dockerfile
COPY latest_model /app/latest_model
```
**Pros:** Simple, fast startup  
**Cons:** Large image, need to rebuild on index updates

### Option B: Cloud Storage Mount
Use Cloud Run volume mounts (Gen 2 execution environment) to mount a GCS bucket.

### Option C: Startup Download
Download index from GCS on container startup.

**Recommended:** For Cloud Run, Option A (build into image) is simplest unless index > 10GB.

## Environment Variables

Set these in Cloud Run:

**Backend:**
- `ANTHROPIC_API_KEY`: Your Claude API key
- `DB_HOST`, `DB_PORT`, etc.: Database connection (if using Cloud SQL)
- `HF_HOME`: HuggingFace cache directory

**Frontend:**
- `BACKEND_URL`: Full URL to backend Cloud Run service
- `NODE_ENV`: `production`

## Health Checks

Both Dockerfiles include health checks:
- Backend: `http://localhost:8000/health`
- Frontend: Default Next.js health check

Cloud Run will use these automatically.

## Cost Optimization

- **Min instances: 0** - Scale to zero when not in use
- **CPU-only PyTorch** - Saves ~600MB (faster builds)
- **Cloud Run** - Pay per request (no idle costs)
- **Auto-scaling** - Handles traffic spikes automatically

## Performance Tips

1. **Cold starts**: Set `min-instances=1` if you need < 1s response times
2. **Memory**: Start with 2Gi for backend, increase if needed
3. **CPU**: 2 vCPU for backend is usually sufficient
4. **Concurrency**: Default 80 requests/instance is fine

## Testing Locally First

Always test with Docker Compose before deploying:

```bash
docker-compose up --build
```

Then test at:
- Frontend: http://localhost:3000
- Backend: http://localhost:8000/docs

## Monitoring

Monitor in GCP Console:
- Cloud Run metrics (requests, latency, errors)
- Cloud Logging for application logs
- Error Reporting for exceptions

## Troubleshooting

**Build timeouts**: Normal, PyTorch is large. Cloud Build has 10h timeout.

**Memory errors**: Increase memory allocation in Cloud Run settings.

**Cold starts slow**: This is normal (~10-30s first request). Use min-instances=1 if needed.

**Frontend can't reach backend**: Check `BACKEND_URL` environment variable.

## Next Steps

1. Test locally with `docker-compose up`
2. Build and push images
3. Deploy backend first
4. Deploy frontend with correct `BACKEND_URL`
5. Test end-to-end
6. Monitor and optimize


