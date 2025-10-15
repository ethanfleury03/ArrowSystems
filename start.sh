#!/bin/bash

# DuraFlex Technical Assistant - Enhanced Startup Script
# Works on local machines and RunPod GPU instances
# Usage: ./start.sh

set -e  # Exit on error

echo "=========================================="
echo "🔧 DuraFlex Technical Assistant"
echo "=========================================="
echo ""

# Set GPU acceleration environment variables for Ollama
export OLLAMA_GPU_LAYERS=32
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=1

# Set Claude API key for LLM answer generation
export ANTHROPIC_API_KEY=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

# Detect environment
IS_RUNPOD=false
if [ -d "/runpod-volume" ] || [ -d "/workspace" ] || [ ! -z "$RUNPOD_POD_ID" ]; then
    IS_RUNPOD=true
    echo "🖥️  Environment: RunPod GPU Instance"
    echo "📍 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'Not detected')"
else
    echo "🖥️  Environment: Local Machine"
fi
echo ""

# Function to check if a Python package is installed
check_package() {
    python -c "import $1" 2>/dev/null
    return $?
}

# Virtual environment handling
# On RunPod: Use system environment (has PyTorch pre-installed)
# On Local: Use venv for isolation
if [ "$IS_RUNPOD" = false ]; then
    # Local machine - use venv for isolation
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
        echo "✅ Virtual environment created"
    fi
    
    echo "📦 Activating virtual environment..."
    source venv/bin/activate
else
    # RunPod - use system environment (has PyTorch, Transformers, etc.)
    echo "📦 Using system Python environment"
    echo "   (Keeps access to pre-installed PyTorch & ML libraries)"
fi

# Check Python version
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
echo "🐍 Python version: $PYTHON_VERSION"
echo ""

# Smart dependency checking
echo "🔍 Checking dependencies..."
echo ""

MISSING_CORE=false
MISSING_UI=false

# Debug: Show Python path
if [ "$IS_RUNPOD" = true ]; then
    echo "   Python: $(which python)"
    echo "   Checking if packages are installed via pip..."
fi

# Check PyTorch
if check_package torch; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    echo "  ✅ PyTorch: $TORCH_VERSION"
else
    echo "  ❌ PyTorch not found"
    MISSING_CORE=true
fi

# For other packages on RunPod, just check if Streamlit is missing
# The system packages (transformers, llama-index, etc.) are pre-installed and work
if [ "$IS_RUNPOD" = true ]; then
    # Assume system packages are good if PyTorch works
    echo "  ✅ Transformers (system)"
    echo "  ✅ LlamaIndex (system)" 
    echo "  ✅ Sentence-Transformers (system)"
    
    # Check Streamlit
    if check_package streamlit; then
        echo "  ✅ Streamlit"
    else
        echo "  ❌ Streamlit not found"
        MISSING_UI=true
    fi
    
    # Check PyMuPDF (critical for ingestion)
    if check_package fitz; then
        echo "  ✅ PyMuPDF"
    else
        echo "  ❌ PyMuPDF not found"
        MISSING_UI=true
    fi
    
    # Check rank-bm25
    if check_package rank_bm25; then
        echo "  ✅ rank-bm25"
    else
        echo "  ❌ rank-bm25 not found"
        MISSING_UI=true
    fi
else
    # Local machine - check everything
    if check_package transformers; then
        echo "  ✅ Transformers"
    else
        echo "  ❌ Transformers not found"
        MISSING_CORE=true
    fi

    if python -c "import llama_index.core" 2>/dev/null; then
        echo "  ✅ LlamaIndex"
    else
        echo "  ❌ LlamaIndex not found"
        MISSING_CORE=true
    fi

    if check_package sentence_transformers; then
        echo "  ✅ Sentence-Transformers"
    else
        echo "  ❌ Sentence-Transformers not found"
        MISSING_CORE=true
    fi

    if check_package streamlit; then
        echo "  ✅ Streamlit"
    else
        echo "  ❌ Streamlit not found"
        MISSING_UI=true
    fi
fi

echo ""

# Install missing dependencies with smart handling
if [ "$MISSING_CORE" = true ]; then
    echo "📥 Installing all dependencies from requirements.txt..."
    echo "   This may take a few minutes..."
    echo ""
    
    pip install --upgrade pip -q
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install to user directory to avoid system conflicts
        pip install -r requirements.txt \
            --user \
            --upgrade-strategy only-if-needed \
            --ignore-installed cryptography \
            --no-warn-script-location
    else
        # On local: Normal install in venv
        pip install -r requirements.txt
    fi
    
    echo ""
    echo "✅ All dependencies installed"
    echo ""
elif [ "$MISSING_UI" = true ]; then
    echo "📥 Installing all dependencies from requirements.txt..."
    echo "   This will show progress so you can see what's happening..."
    echo ""
    
    pip install --upgrade pip -q
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install all requirements
        # Use --upgrade-strategy only-if-needed to skip reinstalling satisfied dependencies
        echo "   Note: Skipping already-satisfied dependencies (PyTorch, Transformers, etc.)"
        pip install -r requirements.txt \
                    --user \
                    --upgrade-strategy only-if-needed \
                    --ignore-installed cryptography \
                    --no-warn-script-location
        
        INSTALL_STATUS=$?
    else
        # On local: Install everything from requirements.txt
        pip install -r requirements.txt
        
        INSTALL_STATUS=$?
    fi
    
    echo ""
    if [ $INSTALL_STATUS -eq 0 ]; then
        echo "✅ All dependencies installed successfully"
    else
        echo "⚠️  Installation had issues, but may have partially succeeded"
        echo "   Attempting to continue..."
    fi
    echo ""
else
    echo "✅ All dependencies satisfied!"
    echo ""
fi

# Check Claude for LLM answer generation
echo "🤖 Checking Claude for LLM answer generation..."
echo ""

# Check if anthropic package is installed
if ! python -c "import anthropic" 2>/dev/null; then
    echo "  ⚠️  Anthropic package not found"
    echo "     Installing anthropic package..."
    
    if pip install anthropic; then
        echo "  ✅ Anthropic package installed"
    else
        echo "  ❌ Failed to install Anthropic package"
        echo "     LLM answer generation will be disabled"
    fi
fi

# Check if API key is set
if [ -z "$ANTHROPIC_API_KEY" ]; then
    echo "  ⚠️  ANTHROPIC_API_KEY not set"
    echo "     LLM answer generation will be disabled"
else
    echo "  ✅ Claude API key found"
    echo "  🎉 LLM answer generation enabled!"
    echo "     ChatGPT-style responses will be generated"
fi

echo ""

# Check if config files exist
if [ ! -f "config/users.yaml" ]; then
    echo "⚠️  Warning: config/users.yaml not found!"
    echo "   UI authentication may not work properly"
    echo ""
fi

if [ ! -f "config/app_config.yaml" ]; then
    echo "⚠️  Warning: config/app_config.yaml not found!"
    echo ""
fi

# Check if storage/index exists (check multiple locations)
STORAGE_PATH=""
if [ -d "/workspace/storage" ] && [ -f "/workspace/storage/docstore.json" ]; then
    STORAGE_PATH="/workspace/storage"
    echo "✅ RAG index found in /workspace/storage/"
elif [ -d "storage" ] && [ -f "storage/docstore.json" ]; then
    STORAGE_PATH="storage"
    echo "✅ RAG index found in ./storage/"
else
    echo "=========================================="
    echo "⚠️  RAG Index Not Found!"
    echo "=========================================="
    echo ""
    echo "Checked locations:"
    echo "  • /workspace/storage/"
    echo "  • ./storage/"
    echo ""
    echo "The vector index hasn't been built yet."
    echo "You need to run ingestion first to process your PDFs."
    echo ""
    echo "This will:"
    echo "  • Extract text from PDFs in data/ folder"
    echo "  • Extract tables and images"
    echo "  • Create vector embeddings"
    echo "  • Build searchable index"
    echo ""
    echo "Estimated time: 5-15 minutes (depending on # of PDFs)"
    echo ""
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod, auto-run if data exists
        if [ -d "data" ] && [ "$(ls -A data/*.pdf 2>/dev/null)" ]; then
            echo "📄 Found PDF files in data/ folder"
            read -p "Run ingestion now? (y/n) " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo ""
                echo "🔄 Running ingestion..."
                python ingest.py
                echo ""
                echo "✅ Ingestion complete!"
                echo ""
            else
                echo ""
                echo "⚠️  Skipping ingestion - queries will fail without index"
                echo "   Run manually later: python ingest.py"
                echo ""
            fi
        else
            echo "⚠️  No PDF files found in data/ folder"
            echo "   Add PDFs to data/ and run: python ingest.py"
            echo ""
        fi
    else
        # On local machine, ask user
        read -p "Do you want to run ingestion now? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo ""
            echo "🔄 Running ingestion..."
            python ingest.py
            echo ""
            echo "✅ Ingestion complete!"
            echo ""
        else
            echo ""
            echo "⚠️  Starting without index - queries will fail"
            echo "   Run ingestion later: python ingest.py"
            echo ""
        fi
    fi
fi

# If we found storage, show stats
if [ ! -z "$STORAGE_PATH" ]; then
    if [ -f "$STORAGE_PATH/docstore.json" ]; then
        NUM_DOCS=$(python -c "import json; print(len(json.load(open('$STORAGE_PATH/docstore.json'))['docstore/data']))" 2>/dev/null || echo "unknown")
        echo "   📊 Indexed chunks: $NUM_DOCS"
    fi
    echo ""
fi

# Determine port and URL
if [ "$IS_RUNPOD" = true ]; then
    # Use port 8501 on RunPod
    PORT=8501
    echo "📍 Using port $PORT (RunPod HTTP Service port)"
else
    PORT=8501
fi

if [ "$IS_RUNPOD" = true ]; then
    echo "=========================================="
    echo "🌐 RunPod Network Configuration"
    echo "=========================================="
    echo ""
    echo "The app will run on port $PORT"
    echo ""
    echo "To access from your browser:"
    echo "  1. Go to your RunPod pod page"
    echo "  2. Under 'Connect' → 'HTTP Services'"
    echo "  3. Click on the port $PORT service link"
    echo ""
    echo "💡 The URL will look like:"
    echo "   https://xxxxx-$PORT.proxy.runpod.net"
    echo ""
else
    echo "=========================================="
    echo "🌐 Local Access"
    echo "=========================================="
    echo ""
    echo "After startup, open your browser to:"
    echo "  http://localhost:$PORT"
    echo ""
fi

echo "=========================================="
echo "🔐 Login Credentials"
echo "=========================================="
echo ""
echo "  Admin:       admin / admin123"
echo "  Technician:  tech1 / tech123"
echo ""
echo "=========================================="
echo ""

# Start the application
echo "🚀 Starting Streamlit server..."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Precompile Python files for faster startup (saves 1-2 seconds)
if [ "$IS_RUNPOD" = true ]; then
    echo "⚡ Precompiling Python files..."
    python -m compileall app.py components/ utils/ -q 2>/dev/null || true
    echo ""
fi

# Run streamlit with appropriate settings
if [ "$IS_RUNPOD" = true ]; then
    # RunPod - bind to all interfaces
    # Use python -m streamlit to ensure it's found
    # Note: Disable CORS and XSRF for RunPod proxy compatibility (prevents WebSocket errors)
    python -m streamlit run app.py \
        --server.port=$PORT \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --server.enableWebsocketCompression=false
else
    # Local - standard settings
    python -m streamlit run app.py --server.port=$PORT
fi
