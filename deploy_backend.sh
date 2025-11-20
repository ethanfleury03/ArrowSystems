#!/bin/bash
# Cloud Run Deployment Script - Local Docker Build
# This script builds Docker images locally, pushes to GCR, and deploys to Cloud Run
# WITHOUT using Cloud Build (cost-effective approach)

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# =============================================================================
# Configuration
# =============================================================================
# TODO: Replace <PROJECT_ID> with your actual GCP project ID
# Your project ID: arrow-rag-support-prod
PROJECT_ID="arrow-rag-support-prod"
REGION="us-central1"
SERVICE_NAME="arrow-rag-backend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# =============================================================================
# Validation
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Cloud Run Deployment (Local Build)${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if PROJECT_ID is set (not placeholder)
if [ "$PROJECT_ID" = "<PROJECT_ID>" ] || [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: PROJECT_ID not configured${NC}"
    echo ""
    echo "Please edit this script and replace <PROJECT_ID> with your GCP project ID:"
    echo "  PROJECT_ID=\"arrow-rag-support-prod\""
    echo ""
    exit 1
fi

# Check required environment variables
echo -e "${YELLOW}📋 Checking required environment variables...${NC}"
REQUIRED_VARS=("DATABASE_URL" "DOCS_BUCKET_NAME" "ANTHROPIC_API_KEY")
MISSING_VARS=()

for var in "${REQUIRED_VARS[@]}"; do
    if [ -z "${!var}" ]; then
        MISSING_VARS+=("$var")
    fi
done

if [ ${#MISSING_VARS[@]} -ne 0 ]; then
    echo -e "${RED}❌ Missing required environment variables:${NC}"
    for var in "${MISSING_VARS[@]}"; do
        echo "  - $var"
    done
    echo ""
    echo "Please export these variables before running the script:"
    echo "  export DATABASE_URL=\"postgresql://...\""
    echo "  export DOCS_BUCKET_NAME=\"your-bucket-name\""
    echo "  export ANTHROPIC_API_KEY=\"your-api-key\""
    echo ""
    exit 1
fi

echo -e "${GREEN}✅ All required environment variables are set${NC}"
echo ""

# Display configuration
echo -e "${YELLOW}📋 Deployment Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo "  Image: $IMAGE_NAME"
echo ""

# =============================================================================
# Google Cloud Authentication
# =============================================================================
echo -e "${YELLOW}🔐 Authenticating Docker to Google Container Registry...${NC}"
if ! gcloud auth configure-docker --quiet; then
    echo -e "${RED}❌ Failed to configure Docker authentication${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Docker authentication configured${NC}"
echo ""

# Check if gcloud is authenticated
echo -e "${YELLOW}🔍 Verifying Google Cloud authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo -e "${RED}❌ Not authenticated with Google Cloud${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi
echo -e "${GREEN}✅ Google Cloud authentication verified${NC}"
echo ""

# Set the project
echo -e "${YELLOW}🎯 Setting GCP project...${NC}"
gcloud config set project "$PROJECT_ID" --quiet
echo -e "${GREEN}✅ Project set to $PROJECT_ID${NC}"
echo ""

# =============================================================================
# Build Docker Image Locally
# =============================================================================
echo -e "${YELLOW}🏗️  Building Docker image locally...${NC}"
echo "  Dockerfile: backend/Dockerfile.backend"
echo "  Image: $IMAGE_NAME"
echo ""

# Build with --no-cache to ensure we don't reuse old bloated layers
if ! docker build --no-cache -f backend/Dockerfile.backend \
    -t "$IMAGE_NAME" \
    .; then
    echo -e "${RED}❌ Docker build failed${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker image built successfully${NC}"
echo ""

# =============================================================================
# Push Image to Google Container Registry
# =============================================================================
echo -e "${YELLOW}📤 Pushing image to Google Container Registry...${NC}"
echo "  Image: $IMAGE_NAME"
echo ""

if ! docker push "$IMAGE_NAME"; then
    echo -e "${RED}❌ Failed to push image to GCR${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Image pushed to GCR successfully${NC}"
echo ""

# =============================================================================
# Deploy to Cloud Run (Cheap Mode)
# =============================================================================
echo -e "${YELLOW}🚀 Deploying to Cloud Run...${NC}"
echo "  Service: $SERVICE_NAME"
echo "  Region: $REGION"
echo ""

# REMINDER: Before deploying, add Cloud Run's outbound IP to:
# Cloud SQL → Connections → Authorized Networks
# This is a one-time setup step that must be done manually in the GCP Console

if ! gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --add-cloudsql-instances "${PROJECT_ID}:${REGION}:rag-postgres" \
    --set-env-vars "DATABASE_URL=${DATABASE_URL}" \
    --set-env-vars "DOCS_BUCKET_NAME=${DOCS_BUCKET_NAME}" \
    --set-env-vars "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}" \
    --set-env-vars "APP_ENV=production" \
    --set-env-vars "ALLOWED_ORIGINS=https://support.arrowsystems.com" \
    --set-env-vars "LOG_LEVEL=INFO" \
    --set-env-vars "HF_HOME=/tmp/hf,TRANSFORMERS_CACHE=/tmp/hf,SENTENCE_TRANSFORMERS_HOME=/tmp/st"; then
    echo -e "${RED}❌ Cloud Run deployment failed${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""

# =============================================================================
# Get Service URL
# =============================================================================
echo -e "${YELLOW}🌐 Retrieving service URL...${NC}"
SERVICE_URL=$(gcloud run services describe "$SERVICE_NAME" \
    --region="$REGION" \
    --format="value(status.url)" 2>/dev/null || echo "")

if [ -n "$SERVICE_URL" ]; then
    echo ""
    echo -e "${GREEN}🎉 Your backend is now live!${NC}"
    echo ""
    echo -e "${BLUE}Service URL:${NC}"
    echo -e "${GREEN}$SERVICE_URL${NC}"
    echo ""
else
    echo -e "${YELLOW}⚠️  Could not retrieve service URL${NC}"
    echo "You can get it manually with:"
    echo "  gcloud run services describe $SERVICE_NAME --region=$REGION --format='value(status.url)'"
    echo ""
fi

# =============================================================================
# Helpful Commands
# =============================================================================
echo -e "${YELLOW}📊 Useful commands:${NC}"
echo ""
echo "View logs:"
echo "  gcloud run services logs tail $SERVICE_NAME --region=$REGION"
echo ""
echo "View service details:"
echo "  gcloud run services describe $SERVICE_NAME --region=$REGION"
echo ""
echo "Update service (re-run this script):"
echo "  bash deploy_backend.sh"
echo ""
echo -e "${GREEN}✨ Deployment script completed!${NC}"

