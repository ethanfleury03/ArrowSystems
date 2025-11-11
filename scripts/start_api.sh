#!/bin/bash

# Start FastAPI Backend for DuraFlex Technical Assistant
# Production-ready API server startup script

set -e

echo "🚀 Starting DuraFlex Technical Assistant API Server..."

# Check if required environment variables are set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "⚠️  Warning: ANTHROPIC_API_KEY not set. LLM features will be disabled."
fi

# Check if index exists in multiple locations
if [ ! -d "latest_model" ] && [ ! -d "../latest_model" ] && [ ! -d "/workspace/latest_model" ] && [ ! -d "/workspace/ArrowSystems/latest_model" ]; then
    echo "❌ Error: Index not found. Please run 'python -m backend.ingest' first."
    echo "   Or ensure the latest_model directory exists."
    exit 1
fi

# Create logs directory if it doesn't exist
mkdir -p logs

# Change to project root directory (backend package root)
cd "$(dirname "$0")/.."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export HF_HUB_ENABLE_HF_TRANSFER=0

# Start the API server
echo "📚 API documentation will be available at: http://localhost:8000/docs"
echo "🔍 Health check endpoint: http://localhost:8000/health"
echo ""

# Run with appropriate settings based on environment
if [ "$ENVIRONMENT" = "development" ]; then
    echo "🔧 Running in development mode with auto-reload..."
    python -m backend.api --host 0.0.0.0 --port 8000 --reload
else
    echo "🏭 Running in production mode..."
    python -m backend.api --host 0.0.0.0 --port 8000 --workers 2
fi
