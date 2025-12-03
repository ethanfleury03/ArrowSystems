#!/bin/bash
# Deployment script for arrow-rag-backend Cloud Run service
# This script deploys the service and configures scaling and env vars
# NOTE: GCS volumes are NOT used - index is bundled in the Docker image

set -e

PROJECT="arrow-rag-support-prod"
REGION="us-central1"
SERVICE="arrow-rag-backend"

# SAFETY CHECK – do not allow accidental GCS mounts
# Check for volume mount commands in deployment files (excluding this safety check itself)
if grep -rE "(--add-volume|volumeMounts:|volumes:.*rag-index|gcsfuse)" \
   --include="*.sh" --include="*.yaml" --include="*.yml" \
   deployment .github/workflows 2>/dev/null | \
   grep -v "SAFETY CHECK" | \
   grep -v "do not allow" | \
   grep -v "Remove any existing"; then
  echo "❌ ERROR: GCS volume mount detected — this system no longer uses gcsfuse."
  echo "   Index must be bundled in the Docker image at /app/latest_model/"
  exit 1
fi

echo "=========================================="
echo "Pre-Deploy: Verify secrets in Secret Manager"
echo "=========================================="
echo "🔐 Checking Secret Manager for JWT_SECRET_KEY..."
if ! gcloud secrets versions list JWT_SECRET_KEY --project="$PROJECT" --format="value(name)" >/dev/null 2>&1; then
  echo "❌ ERROR: JWT_SECRET_KEY does not exist in Secret Manager."
  echo "The backend will crash if deployed without this secret."
  echo ""
  echo "To create the secret:"
  echo "  gcloud secrets create JWT_SECRET_KEY --project=$PROJECT"
  echo "  echo -n 'your-secret-value' | gcloud secrets versions add JWT_SECRET_KEY --data-file=- --project=$PROJECT"
  echo ""
  echo "Generate a secure secret with:"
  echo "  python -c 'import secrets; print(secrets.token_urlsafe(64))'"
  exit 1
fi
echo "✅ JWT_SECRET_KEY found in Secret Manager"

echo ""
echo "🔐 Checking Secret Manager for DATABASE_URL..."
if ! gcloud secrets versions list DATABASE_URL --project="$PROJECT" --format="value(name)" >/dev/null 2>&1; then
  echo "❌ ERROR: DATABASE_URL does not exist in Secret Manager."
  exit 1
fi
echo "✅ DATABASE_URL found in Secret Manager"

echo ""
echo "🔐 Checking Secret Manager for FRONTEND_SESSION_SECRET..."
if ! gcloud secrets versions list FRONTEND_SESSION_SECRET --project="$PROJECT" --format="value(name)" >/dev/null 2>&1; then
  echo "❌ ERROR: FRONTEND_SESSION_SECRET does not exist in Secret Manager."
  exit 1
fi
echo "✅ FRONTEND_SESSION_SECRET found in Secret Manager"

echo ""
echo "=========================================="
echo "Step 1: Deploy base service from YAML"
echo "=========================================="
gcloud run services replace deployment/cloud-run-service.yaml \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "Step 2: Remove any existing GCS volume mounts (if present)"
echo "=========================================="
# Remove any existing volume mounts to ensure clean state
gcloud run services update $SERVICE \
  --remove-volume=rag-index \
  --remove-volume-mount=rag-index \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT 2>/dev/null || echo "No rag-index volume to remove (expected)"

gcloud run services update $SERVICE \
  --remove-volume=data-volume \
  --remove-volume-mount=data-volume \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT 2>/dev/null || echo "No data-volume volume to remove (expected)"

echo ""
echo "=========================================="
echo "Step 3: Update secrets from Google Secret Manager"
echo "=========================================="
gcloud run services update $SERVICE \
  --update-secrets=DATABASE_URL=DATABASE_URL:latest \
  --update-secrets=JWT_SECRET_KEY=JWT_SECRET_KEY:latest \
  --update-secrets=FRONTEND_SESSION_SECRET=FRONTEND_SESSION_SECRET:latest \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "Step 4: Set scaling and concurrency"
echo "=========================================="
gcloud run services update $SERVICE \
  --min-instances=1 \
  --cpu-throttling \
  --max-instances=10 \
  --concurrency=100 \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "✅ Deployment complete!"
echo "=========================================="
echo "Service URL:"
gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(status.url)"





