#!/bin/bash

# DuraFlex Technical Assistant - Startup Script
# Usage: ./start.sh

echo "=================================="
echo "DuraFlex Technical Assistant"
echo "=================================="
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "✅ Virtual environment created"
fi

# Activate virtual environment
echo "📦 Activating virtual environment..."
source venv/bin/activate

# Check if dependencies are installed
if [ ! -f "venv/installed.flag" ]; then
    echo "📥 Installing dependencies..."
    pip install --upgrade pip
    pip install -r requirements.txt
    touch venv/installed.flag
    echo "✅ Dependencies installed"
else
    echo "✅ Dependencies already installed"
fi

# Check if storage exists
if [ ! -d "storage" ]; then
    echo ""
    echo "⚠️  Storage directory not found!"
    echo "You need to run ingestion first:"
    echo "  python ingest.py"
    echo ""
    read -p "Do you want to run ingestion now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        python ingest.py
    else
        echo "⚠️  Starting without index - queries will fail"
    fi
fi

# Start application
echo ""
echo "🚀 Starting DuraFlex Technical Assistant..."
echo "📍 Application will be available at: http://localhost:8501"
echo "🔐 Default login: admin / admin123"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

streamlit run app.py

