#!/bin/bash
##########################################
# First-Time Complete Setup Script
# Does EVERYTHING: models + dependencies + start
# For technicians who want one command to do it all
##########################################

echo "=========================================="
echo "🚀 DuraFlex Complete Setup"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Install dependencies"
echo "  2. Download AI models (~2GB)"
echo "  3. Start the application"
echo ""
echo "⏱️  Total time: 10-15 minutes"
echo ""
echo "💡 You can let this run while you grab lunch!"
echo ""

# Confirm
read -p "Ready to proceed? (yes/no): " CONFIRM
if [ "$CONFIRM" != "yes" ] && [ "$CONFIRM" != "y" ]; then
    echo "Setup cancelled."
    exit 0
fi

echo ""
echo "=========================================="
echo "📦 Step 1/3: Installing Dependencies"
echo "=========================================="
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found!"
    echo "   Please install Python 3.10 or newer from:"
    echo "   https://www.python.org/downloads/"
    exit 1
fi

echo "🐍 Python version: $($PYTHON --version)"
echo ""

# Check pip
if ! $PYTHON -m pip --version &> /dev/null; then
    echo "❌ Error: pip not found!"
    echo "   Please install pip"
    exit 1
fi

# Install requirements
if [ -f "requirements.txt" ]; then
    echo "📥 Installing Python packages from requirements.txt..."
    echo "   (This may take 2-3 minutes)"
    echo ""
    
    $PYTHON -m pip install -r requirements.txt --quiet
    
    if [ $? -eq 0 ]; then
        echo "✅ Dependencies installed successfully!"
    else
        echo "⚠️  Some packages may have failed to install"
        echo "   The app might still work - continuing..."
    fi
else
    echo "⚠️  requirements.txt not found"
    echo "   Assuming dependencies are already installed"
fi

echo ""
echo "=========================================="
echo "🤖 Step 2/3: Downloading AI Models"
echo "=========================================="
echo ""

# Run model setup script
if [ -f "setup_models.sh" ]; then
    chmod +x setup_models.sh
    bash setup_models.sh
    
    if [ $? -ne 0 ]; then
        echo ""
        echo "⚠️  Model download failed"
        echo "   The app will download models on first run (slower)"
        echo "   Continuing anyway..."
    fi
else
    echo "⚠️  setup_models.sh not found"
    echo "   Models will be downloaded on first app startup"
fi

echo ""
echo "=========================================="
echo "🚀 Step 3/3: Starting Application"
echo "=========================================="
echo ""

echo "💡 The application will open in your browser automatically"
echo ""
echo "Press Ctrl+C at any time to stop the app"
echo ""

# Make start.sh executable
chmod +x start.sh

# Small delay
sleep 2

# Start the app
echo "🎉 Launching DuraFlex Technical Assistant..."
echo ""

./start.sh

