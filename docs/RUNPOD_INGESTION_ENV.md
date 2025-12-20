# RunPod Ingestion Environment Variables

Complete list of environment variables needed to run ingestion on RunPod GPU workers.

## Required Environment Variables

### Database Connection
```bash
# PostgreSQL connection string (REQUIRED)
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@YOUR_HOST:5432/rag_app"

# OR if using Cloud SQL Proxy:
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"
```

### Google Cloud Storage (Documents)
```bash
# GCS bucket where documents are stored (REQUIRED)
export DOCS_GCS_BUCKET="arrow-documents-prod"

# GCS prefix/path within bucket (optional, defaults to bucket root "")
export DOCS_GCS_PREFIX="ROOT"  # or "documents/" or "" for bucket root

# Path to GCS service account JSON key file (REQUIRED)
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcs-key.json"
```

### RAG Index Storage
```bash
# GCS bucket where index artifacts are stored (default: arrow-rag-support-prod-rag)
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"

# GCS prefix for index artifacts (default: latest_model/)
export RAG_INDEX_GCS_PREFIX="latest_model/"

# Local directory where index will be saved (default: latest_model)
export RAG_INDEX_LOCAL_DIR="/workspace/latest_model"
```

### Environment Mode
```bash
# Set to "prod" for production ingestion (default: "dev")
export ENV="prod"
```

## Optional but Recommended

### HuggingFace Cache (Performance)
```bash
# HuggingFace model cache directory (default: ~/.cache/huggingface)
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="/workspace/.cache/huggingface"
```

### Anthropic API (Claude Rewriting)
```bash
# Optional: For Claude semantic rewriting during ingestion
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

### Ingestion Configuration
```bash
# Allow ingestion operations (for ingestion workers)
export ARROW_ALLOW_APP_INGESTION="true"

# Disable metadata updates during ingestion (recommended)
export DISABLE_METADATA_UPDATE="1"
```

### Performance Tuning (Optional)
```bash
# Max characters per document for smart chunking (default: 250000)
export MAX_DOC_CHARS_FOR_SMART_CHUNK="250000"

# Max characters per document (default: 250000)
export MAX_DOC_CHARS="250000"

# Max chunks per document (default: 5000)
export MAX_CHUNKS_PER_DOC="5000"

# Max seconds per document processing (default: 90)
export MAX_SECONDS_PER_DOC="90"

# Chunking heartbeat interval (default: 50)
export CHUNK_HEARTBEAT_EVERY="50"

# Enable chunk debug logging (default: 0)
export CHUNK_DEBUG="0"
```

## Complete RunPod Setup Script (Production Values)

Save this as `setup_runpod_ingestion.sh`:

```bash
#!/bin/bash
# RunPod Ingestion Environment Setup - Production Configuration

# ============================================
# REQUIRED - Database
# IMPORTANT: Cloud Run uses Unix socket, RunPod needs TCP
# Option 1: Use Cloud SQL Proxy (recommended)
#   Start proxy first: cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres
#   Then use localhost connection:
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"

# Option 2: Direct connection (if Cloud SQL allows external IPs)
# export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app"

# ============================================
# REQUIRED - GCS Configuration
# ============================================
export DOCS_GCS_BUCKET="arrow-rag-support-prod-docs"
export DOCS_GCS_PREFIX="ROOT"
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcs-key.json"

# ============================================
# REQUIRED - RAG Index Configuration
# ============================================
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"
export RAG_INDEX_LOCAL_DIR="/workspace/latest_model"

# ============================================
# REQUIRED - Environment
# ============================================
export ENV="prod"

# ============================================
# RECOMMENDED - HuggingFace Cache
# ============================================
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="/workspace/.cache/huggingface"

# ============================================
# OPTIONAL - Anthropic (Claude Rewriting)
# ============================================
# export ANTHROPIC_API_KEY="sk-ant-api03-..."

# ============================================
# OPTIONAL - Ingestion Configuration
# ============================================
export ARROW_ALLOW_APP_INGESTION="true"
export DISABLE_METADATA_UPDATE="1"

# ============================================
# OPTIONAL - Performance Tuning
# ============================================
# export MAX_DOC_CHARS_FOR_SMART_CHUNK="250000"
# export MAX_DOC_CHARS="250000"
# export MAX_CHUNKS_PER_DOC="5000"
# export MAX_SECONDS_PER_DOC="90"
# export CHUNK_HEARTBEAT_EVERY="50"
# export CHUNK_DEBUG="0"

echo "✅ Environment variables configured for RunPod ingestion"
echo "   DATABASE_URL: ${DATABASE_URL%%@*}@***"
echo "   DOCS_GCS_BUCKET: $DOCS_GCS_BUCKET"
echo "   RAG_INDEX_GCS_BUCKET: $RAG_INDEX_GCS_BUCKET"
echo "   RAG_INDEX_LOCAL_DIR: $RAG_INDEX_LOCAL_DIR"
echo "   ENV: $ENV"
```

## Quick Start Commands

### 1. Set up Cloud SQL Proxy (if using Unix socket connection)
```bash
# Install Cloud SQL Proxy if not already installed
# Download from: https://cloud.google.com/sql/docs/postgres/sql-proxy

# Authenticate
gcloud auth application-default login

# Start proxy in background
cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres &
```

### 2. Set up environment variables
```bash
# Source the setup script
source setup_runpod_ingestion.sh

# OR set manually (replace YOUR_PASSWORD and YOUR_API_KEY with actual values):
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"
export DOCS_GCS_BUCKET="arrow-rag-support-prod-docs"
export DOCS_GCS_PREFIX="ROOT"
export GOOGLE_APPLICATION_CREDENTIALS="/workspace/gcs-key.json"
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"
export RAG_INDEX_LOCAL_DIR="/workspace/latest_model"
export ENV="prod"
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="/workspace/.cache/huggingface"
export ARROW_ALLOW_APP_INGESTION="true"
export DISABLE_METADATA_UPDATE="1"
# export ANTHROPIC_API_KEY="sk-ant-api03-..."  # Optional: Uncomment and add your key
```

### 2. Upload GCS service account key
```bash
# Upload your service account JSON key to RunPod workspace
# Place it at: /workspace/gcs-key.json
# Set permissions:
chmod 600 /workspace/gcs-key.json
```

### 3. Verify GCS access
```bash
# Test GCS connection
python -c "
from google.cloud import storage
client = storage.Client()
buckets = list(client.list_buckets())
print(f'✅ GCS access working. Found {len(buckets)} buckets')
"
```

### 4. Run ingestion
```bash
# Navigate to backend directory
cd /workspace/rag_app.py

# Run full ingestion
python ingest.py

# OR with specific options:
python ingest.py --storage-dir /workspace/latest_model --data-dir /workspace/data
```

## Verification Checklist

Before running ingestion, verify:

- [ ] `DATABASE_URL` is set and points to PostgreSQL (not SQLite)
- [ ] `GOOGLE_APPLICATION_CREDENTIALS` points to valid service account JSON key
- [ ] Service account has Storage Object Admin role on `DOCS_GCS_BUCKET`
- [ ] Service account has Storage Object Admin role on `RAG_INDEX_GCS_BUCKET`
- [ ] `RAG_INDEX_LOCAL_DIR` exists and is writable
- [ ] `HF_HOME` directory exists and is writable (for model cache)
- [ ] Database connection works: `python -c "from backend.utils.db import SessionLocal; s = SessionLocal(); s.close(); print('✅ DB connected')"`

## Common Issues

### "DATABASE_URL not set"
- Ensure `DATABASE_URL` is exported before running `ingest.py`
- Check that it's a PostgreSQL connection string (not SQLite)

### "GCS client not available"
- Verify `GOOGLE_APPLICATION_CREDENTIALS` points to valid JSON file
- Check service account has correct IAM permissions
- Verify file permissions: `chmod 600 /workspace/gcs-key.json`

### "Storage directory not writable"
- Create directory: `mkdir -p /workspace/latest_model`
- Set permissions: `chmod 755 /workspace/latest_model`

### "HuggingFace cache error"
- Create cache directory: `mkdir -p /workspace/.cache/huggingface`
- Set `HF_HOME` environment variable

## Getting Production Values

To get your actual production environment variables from Cloud Run:

```bash
# Get all environment variables from Cloud Run service
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --format='value(spec.template.spec.containers[0].env)' | \
  grep -E "(DATABASE_URL|DOCS_GCS_BUCKET|DOCS_GCS_PREFIX|RAG_INDEX|ANTHROPIC_API_KEY)"

# Extract specific values (replace YOUR_PASSWORD with actual password from output):
# DATABASE_URL format: postgresql://rag_user:YOUR_PASSWORD@/rag_app?host=/cloudsql/...
# For RunPod, convert to: postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app
```

## Production Values Template

Based on your Cloud Run configuration structure:

```bash
# Database - IMPORTANT: Cloud Run uses Unix socket, RunPod needs TCP connection
# Option 1: Use Cloud SQL Proxy (recommended)
# First start proxy: cloud-sql-proxy arrow-rag-support-prod:us-central1:rag-postgres
# Then use (replace YOUR_PASSWORD with actual password):
export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"

# Option 2: Direct connection (if Cloud SQL allows external IPs)
# export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app"

# GCS Documents
export DOCS_GCS_BUCKET="arrow-rag-support-prod-docs"
export DOCS_GCS_PREFIX="ROOT"

# RAG Index
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"
export RAG_INDEX_LOCAL_DIR="/workspace/latest_model"

# Environment
export ENV="prod"

# HuggingFace Cache
export HF_HOME="/workspace/.cache/huggingface"
export TRANSFORMERS_CACHE="/workspace/.cache/huggingface"
export SENTENCE_TRANSFORMERS_HOME="/workspace/.cache/huggingface"

# Ingestion Config
export ARROW_ALLOW_APP_INGESTION="true"
export DISABLE_METADATA_UPDATE="1"

# Optional - Anthropic (Claude)
# export ANTHROPIC_API_KEY="sk-ant-api03-..."  # Get from Cloud Run output above
```

