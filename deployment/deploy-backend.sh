#!/bin/bash
# Deployment script for arrow-rag-backend Cloud Run service
# This script deploys the service and configures volumes, scaling, and env vars

set -e

PROJECT="arrow-rag-support-prod"
REGION="us-central1"
SERVICE="arrow-rag-backend"

echo "=========================================="
echo "Step 1: Deploy base service from YAML"
echo "=========================================="
gcloud run services replace deployment/cloud-run-service.yaml \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "Step 2: Add RAG index volume mount"
echo "=========================================="
echo "Mounting bucket root to /app/latest_model/"
echo "Files at gs://arrow-rag-support-prod-rag/ (bucket root) will appear at /app/latest_model/"
gcloud run services update $SERVICE \
  --add-volume=name=rag-index-volume,type=gcs,bucket=arrow-rag-support-prod-rag \
  --add-volume-mount=volume=rag-index-volume,mount-path=/app/latest_model \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "Step 3: Add data volume mount"
echo "=========================================="
gcloud run services update $SERVICE \
  --add-volume=name=data-volume,type=gcs,bucket=ragapp-data \
  --add-volume-mount=volume=data-volume,mount-path=/app/data \
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





