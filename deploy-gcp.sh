#!/bin/bash
# Google Cloud Deployment Script for RAG Application
# This script handles the complete deployment process to Google Cloud Run

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID="ragapp-476414"
REGION="us-central1"
SERVICE_NAME="rag-app"
IMAGE_NAME="gcr.io/$PROJECT_ID/$SERVICE_NAME"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🚀 Google Cloud RAG App Deployment${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if PROJECT_ID is set
if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: PROJECT_ID not set${NC}"
    echo "Please set your Google Cloud Project ID:"
    echo "export PROJECT_ID=your-project-id"
    echo ""
    echo "Or edit this script and set PROJECT_ID variable"
    exit 1
fi

echo -e "${YELLOW}📋 Configuration:${NC}"
echo "  Project ID: $PROJECT_ID"
echo "  Region: $REGION"
echo "  Service: $SERVICE_NAME"
echo "  Image: $IMAGE_NAME"
echo ""

# Check if gcloud is installed and authenticated
echo -e "${YELLOW}🔐 Checking Google Cloud authentication...${NC}"
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" | grep -q "@"; then
    echo -e "${RED}❌ Not authenticated with Google Cloud${NC}"
    echo "Please run: gcloud auth login"
    exit 1
fi

# Set the project
echo -e "${YELLOW}🎯 Setting project...${NC}"
gcloud config set project $PROJECT_ID

# Enable required APIs
echo -e "${YELLOW}🔧 Enabling required APIs...${NC}"
gcloud services enable cloudbuild.googleapis.com
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com

# Create secrets for API keys (if they don't exist)
echo -e "${YELLOW}🔑 Setting up secrets...${NC}"
echo "You'll need to set up your API keys as secrets:"
echo ""
echo "Run these commands to set up your secrets:"
echo "gcloud secrets create anthropic-api-key --data-file=- <<< 'your-anthropic-key'"
echo ""

# Build and deploy
echo -e "${YELLOW}🏗️  Building and deploying...${NC}"
gcloud builds submit --config cloudbuild.yaml .

# Get the service URL
echo -e "${YELLOW}🌐 Getting service URL...${NC}"
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")

echo ""
echo -e "${GREEN}✅ Deployment completed successfully!${NC}"
echo ""
echo -e "${BLUE}🌐 Your RAG application is available at:${NC}"
echo -e "${GREEN}$SERVICE_URL${NC}"
echo ""
echo -e "${YELLOW}📊 To view logs:${NC}"
echo "gcloud run services logs tail $SERVICE_NAME --region=$REGION"
echo ""
echo -e "${YELLOW}🔧 To update the service:${NC}"
echo "gcloud builds submit --config cloudbuild.yaml ."
echo ""
echo -e "${GREEN}🎉 Deployment complete!${NC}"
