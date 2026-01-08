#!/bin/bash
# Setup script for RunPod to connect to Cloud SQL
# Run this ON the RunPod instance, not on your local machine

set -e

echo "========================================"
echo "RunPod Database Connection Setup"
echo "========================================"
echo ""

# Check if we're on RunPod (or similar Linux environment)
if [ ! -f /etc/os-release ]; then
    echo "⚠️  This script is designed for Linux/RunPod environments"
    exit 1
fi

PROXY_NAME="cloud-sql-proxy"
CONNECTION_STRING="arrow-rag-support-prod:us-central1:rag-postgres"
ARCH=$(uname -m)

# Determine architecture
if [ "$ARCH" = "x86_64" ]; then
    PROXY_URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64"
elif [ "$ARCH" = "aarch64" ] || [ "$ARCH" = "arm64" ]; then
    PROXY_URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.arm64"
else
    echo "❌ Unsupported architecture: $ARCH"
    exit 1
fi

# Check if Cloud SQL Proxy exists
if ! command -v cloud-sql-proxy &> /dev/null && [ ! -f "./$PROXY_NAME" ]; then
    echo "⚠️  Cloud SQL Proxy not found!"
    echo ""
    echo "Downloading Cloud SQL Proxy for $ARCH..."
    if command -v curl &> /dev/null; then
        curl -L -o "$PROXY_NAME" "$PROXY_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$PROXY_NAME" "$PROXY_URL"
    else
        echo "❌ Please install curl or wget"
        exit 1
    fi
    chmod +x "$PROXY_NAME"
    echo "✅ Downloaded $PROXY_NAME"
fi

# Determine proxy executable path
if command -v cloud-sql-proxy &> /dev/null; then
    PROXY_CMD="cloud-sql-proxy"
else
    PROXY_CMD="./$PROXY_NAME"
fi

# Check if proxy is already running
if pgrep -f "cloud-sql-proxy" > /dev/null; then
    echo "⚠️  Cloud SQL Proxy is already running"
    PROXY_PID=$(pgrep -f "cloud-sql-proxy" | head -1)
    echo "   PID: $PROXY_PID"
    echo "   If you need to restart it, stop it first:"
    echo "   pkill cloud-sql-proxy"
else
    echo "Starting Cloud SQL Proxy..."
    echo "   Connection: $CONNECTION_STRING"
    echo "   Listening on: 127.0.0.1:5432"
    echo ""
    
    # Check if gcloud auth is set up
    if [ -z "$GOOGLE_APPLICATION_CREDENTIALS" ] && ! gcloud auth application-default print-access-token &> /dev/null; then
        echo "⚠️  Google Cloud authentication not found!"
        echo ""
        echo "You need to authenticate. Options:"
        echo ""
        echo "Option 1: Use Application Default Credentials (recommended)"
        echo "   gcloud auth application-default login"
        echo ""
        echo "Option 2: Set GOOGLE_APPLICATION_CREDENTIALS"
        echo "   export GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account-key.json"
        echo ""
        read -p "Press Enter to continue after setting up authentication..."
    fi
    
    # Start proxy in background
    nohup $PROXY_CMD "$CONNECTION_STRING" > /tmp/cloud-sql-proxy.log 2>&1 &
    PROXY_PID=$!
    sleep 3
    
    # Check if it started successfully
    if ps -p $PROXY_PID > /dev/null 2>&1; then
        echo "✅ Cloud SQL Proxy started (PID: $PROXY_PID)"
        echo "   Logs: /tmp/cloud-sql-proxy.log"
    else
        echo "❌ Failed to start Cloud SQL Proxy"
        echo "   Check logs: cat /tmp/cloud-sql-proxy.log"
        exit 1
    fi
fi

# Prompt for password if not set
if [ -z "$DATABASE_PASSWORD" ]; then
    read -sp "Enter database password: " DATABASE_PASSWORD
    echo ""
fi

# Set DATABASE_URL
export DATABASE_URL="postgresql://rag_user:${DATABASE_PASSWORD}@127.0.0.1:5432/rag_app"

echo ""
echo "✅ DATABASE_URL has been set!"
echo "   Connection: postgresql://rag_user:***@127.0.0.1:5432/rag_app"
echo ""
echo "To make this permanent in this session, add to your shell:"
echo "   export DATABASE_URL='postgresql://rag_user:${DATABASE_PASSWORD}@127.0.0.1:5432/rag_app'"
echo ""
echo "Test the connection:"
echo "   python check_db_connection.py"
echo ""
echo "Run your script:"
echo "   python update_failed_to_complete.py"
echo ""

