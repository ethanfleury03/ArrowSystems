# Google Cloud Storage Buckets Setup
# This script creates the necessary GCS buckets for your RAG application

#!/bin/bash
set -e

PROJECT_ID="ragapp-476414"
REGION="us-central1"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}🗄️  Google Cloud Storage Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}❌ Error: PROJECT_ID not set${NC}"
    echo "Please set your Google Cloud Project ID:"
    echo "export PROJECT_ID=your-project-id"
    exit 1
fi

echo -e "${YELLOW}📋 Creating GCS buckets for project: $PROJECT_ID${NC}"
echo ""

# Create data bucket
echo -e "${YELLOW}📁 Creating data bucket...${NC}"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://ragapp-data || echo "Bucket may already exist"

# Create models bucket
echo -e "${YELLOW}🤖 Creating models bucket...${NC}"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://ragapp-models || echo "Bucket may already exist"

# Create logs bucket
echo -e "${YELLOW}📝 Creating logs bucket...${NC}"
gsutil mb -p $PROJECT_ID -c STANDARD -l $REGION gs://ragapp-logs || echo "Bucket may already exist"

# Set up lifecycle policies
echo -e "${YELLOW}⏰ Setting up lifecycle policies...${NC}"

# Data bucket - keep for 90 days
gsutil lifecycle set - <<EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 90}
    }
  ]
}
EOF gs://ragapp-data

# Models bucket - keep indefinitely (no lifecycle)
echo "Models bucket will keep data indefinitely"

# Logs bucket - keep for 30 days
gsutil lifecycle set - <<EOF
{
  "rule": [
    {
      "action": {"type": "Delete"},
      "condition": {"age": 30}
    }
  ]
}
EOF gs://ragapp-logs

echo ""
echo -e "${GREEN}✅ GCS buckets created successfully!${NC}"
echo ""
echo -e "${BLUE}📋 Buckets created:${NC}"
echo "  📁 Data: gs://ragapp-data"
echo "  🤖 Models: gs://ragapp-models"
echo "  📝 Logs: gs://ragapp-logs"
echo ""
echo -e "${YELLOW}📤 To upload your data:${NC}"
echo "gsutil -m cp -r ./data/* gs://ragapp-data/"
echo "gsutil -m cp -r ./latest_model/* gs://ragapp-models/"
