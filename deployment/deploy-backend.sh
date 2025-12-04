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
echo "Step 3: Set scaling, concurrency, and memory"
echo "=========================================="
gcloud run services update $SERVICE \
  --memory=4Gi \
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





