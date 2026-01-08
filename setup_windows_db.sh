#!/bin/bash
# Git Bash script to set up Cloud SQL Proxy and DATABASE_URL on Windows

set -e

echo "========================================"
echo "Windows Database Connection Setup"
echo "========================================"
echo ""

PROXY_NAME="cloud-sql-proxy.exe"
PROXY_URL="https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe"
CONNECTION_STRING="arrow-rag-support-prod:us-central1:rag-postgres"

# Check if Cloud SQL Proxy exists
if ! command -v cloud-sql-proxy.exe &> /dev/null && [ ! -f "./$PROXY_NAME" ]; then
    echo "⚠️  Cloud SQL Proxy not found!"
    echo ""
    echo "Downloading Cloud SQL Proxy..."
    if command -v curl &> /dev/null; then
        curl -L -o "$PROXY_NAME" "$PROXY_URL"
    elif command -v wget &> /dev/null; then
        wget -O "$PROXY_NAME" "$PROXY_URL"
    else
        echo "❌ Please install curl or wget, or download manually:"
        echo "   $PROXY_URL"
        exit 1
    fi
    chmod +x "$PROXY_NAME"
    echo "✅ Downloaded $PROXY_NAME"
fi

# Determine proxy executable path
if command -v cloud-sql-proxy.exe &> /dev/null; then
    PROXY_CMD="cloud-sql-proxy.exe"
else
    PROXY_CMD="./$PROXY_NAME"
fi

# Check if proxy is already running
if pgrep -f "cloud-sql-proxy" > /dev/null; then
    echo "⚠️  Cloud SQL Proxy is already running"
    echo "   If you need to restart it, stop it first:"
    echo "   pkill cloud-sql-proxy"
else
    echo "Starting Cloud SQL Proxy..."
    $PROXY_CMD "$CONNECTION_STRING" &
    sleep 3
    
    if pgrep -f "cloud-sql-proxy" > /dev/null; then
        echo "✅ Cloud SQL Proxy started"
    else
        echo "❌ Failed to start Cloud SQL Proxy"
        echo "   Make sure you're authenticated: gcloud auth application-default login"
        exit 1
    fi
fi

# Prompt for password
read -sp "Enter database password: " PASSWORD
echo ""

# Set DATABASE_URL
export DATABASE_URL="postgresql://rag_user:${PASSWORD}@127.0.0.1:5432/rag_app"

echo ""
echo "✅ DATABASE_URL has been set!"
echo "   Connection: postgresql://rag_user:***@127.0.0.1:5432/rag_app"
echo ""
echo "To make this permanent in this session, run:"
echo "   export DATABASE_URL='postgresql://rag_user:${PASSWORD}@127.0.0.1:5432/rag_app'"
echo ""
echo "Test the connection:"
echo "   python check_db_connection.py"
echo ""
echo "Run your script:"
echo "   python update_failed_to_complete.py"
echo ""

