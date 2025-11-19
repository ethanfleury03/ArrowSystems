#!/bin/bash
# Cloud Run Deployment Script - Frontend (Local Docker Build)
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
PROJECT_ID="arrow-rag-support-prod"
REGION="us-central1"
SERVICE_NAME="arrow-rag-frontend"
IMAGE_NAME="gcr.io/${PROJECT_ID}/${SERVICE_NAME}"

# =============================================================================
# Validation
# =============================================================================
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Frontend Cloud Run Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: PROJECT_ID not configured${NC}"
    exit 1
fi

# Check required environment variables
echo -e "${YELLOW}📋 Checking required environment variables...${NC}"
REQUIRED_VARS=("NEXT_PUBLIC_API_URL" "SESSION_SECRET")
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
    echo "  export NEXT_PUBLIC_API_URL=\"https://arrow-rag-backend-<hash>-uc.a.run.app\""
    echo "  export SESSION_SECRET=\"your-secure-random-string-at-least-32-characters\""
    echo ""
    echo "Note: NEXT_PUBLIC_API_URL should be your backend Cloud Run service URL"
    echo "      Get it by running: gcloud run services describe arrow-rag-backend --region=us-central1 --format='value(status.url)'"
    echo ""
    exit 1
fi

# Validate SESSION_SECRET length
if [ ${#SESSION_SECRET} -lt 32 ]; then
    echo -e "${RED}❌ Error: SESSION_SECRET must be at least 32 characters long${NC}"
    echo "Current length: ${#SESSION_SECRET}"
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
echo "  Backend URL: $NEXT_PUBLIC_API_URL"
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
echo "  Dockerfile: frontend/Dockerfile"
echo "  Build context: ./frontend"
echo "  Image: $IMAGE_NAME"
echo ""

# Build with build-time environment variables
# NEXT_PUBLIC_API_URL is required at build time for Next.js validation
if ! docker build -f frontend/Dockerfile \
    --build-arg NEXT_PUBLIC_API_URL="${NEXT_PUBLIC_API_URL}" \
    --build-arg NODE_ENV=production \
    -t "$IMAGE_NAME" \
    ./frontend; then
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

if ! gcloud run deploy "$SERVICE_NAME" \
    --image "$IMAGE_NAME" \
    --region "$REGION" \
    --platform managed \
    --allow-unauthenticated \
    --set-env-vars "NODE_ENV=production" \
    --set-env-vars "NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}" \
    --set-env-vars "SESSION_SECRET=${SESSION_SECRET}" \
    --set-env-vars "PORT=3000" \
    --set-env-vars "HOSTNAME=0.0.0.0"; then
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
    echo -e "${GREEN}🎉 Your frontend is now live!${NC}"
    echo ""
    echo -e "${BLUE}Frontend URL:${NC}"
    echo -e "${GREEN}$SERVICE_URL${NC}"
    echo ""
    echo -e "${YELLOW}📝 Next Steps:${NC}"
    echo "  1. Update your domain/DNS to point to this URL"
    echo "  2. Verify the frontend can connect to backend at: $NEXT_PUBLIC_API_URL"
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
echo "  bash deploy_frontend.sh"
echo ""
echo -e "${GREEN}✨ Deployment script completed!${NC}"

