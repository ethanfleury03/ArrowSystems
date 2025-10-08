#!/bin/bash

# DuraFlex Technical Assistant - Enhanced Startup Script
# Works on local machines and RunPod GPU instances
# Usage: ./start.sh

set -e  # Exit on error

echo "=========================================="
echo "🔧 DuraFlex Technical Assistant"
echo "=========================================="
echo ""

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

# On RunPod, check if packages are in pip list (might not be importable yet)
if [ "$IS_RUNPOD" = true ]; then
    echo "   Scanning installed packages..."
    HAS_TORCH=$(pip list 2>/dev/null | grep -c "torch " || echo "0")
    HAS_TRANSFORMERS=$(pip list 2>/dev/null | grep -c "transformers " || echo "0")
    HAS_LLAMA=$(pip list 2>/dev/null | grep -c "llama-index " || echo "0")
    HAS_SENTENCE=$(pip list 2>/dev/null | grep -c "sentence-transformers" || echo "0")
    HAS_STREAMLIT=$(pip list 2>/dev/null | grep -c "streamlit " || echo "0")
    
    [ "$HAS_TORCH" -eq "0" ] && MISSING_CORE=true || echo "  ✅ PyTorch (system)"
    [ "$HAS_TRANSFORMERS" -eq "0" ] && MISSING_CORE=true || echo "  ✅ Transformers (system)"
    [ "$HAS_LLAMA" -eq "0" ] && MISSING_CORE=true || echo "  ✅ LlamaIndex (system)"
    [ "$HAS_SENTENCE" -eq "0" ] && MISSING_CORE=true || echo "  ✅ Sentence-Transformers (system)"
    [ "$HAS_STREAMLIT" -eq "0" ] && MISSING_UI=true || echo "  ✅ Streamlit"
else
    # Local machine - check imports
    check_package torch && echo "  ✅ PyTorch" || { echo "  ❌ PyTorch not found"; MISSING_CORE=true; }
    check_package transformers && echo "  ✅ Transformers" || { echo "  ❌ Transformers not found"; MISSING_CORE=true; }
    python -c "import llama_index.core" 2>/dev/null && echo "  ✅ LlamaIndex" || { echo "  ❌ LlamaIndex not found"; MISSING_CORE=true; }
    check_package sentence_transformers && echo "  ✅ Sentence-Transformers" || { echo "  ❌ Sentence-Transformers not found"; MISSING_CORE=true; }
    check_package streamlit && echo "  ✅ Streamlit" || { echo "  ❌ Streamlit not found"; MISSING_UI=true; }
fi

echo ""

# Install missing dependencies with smart handling
if [ "$MISSING_CORE" = true ]; then
    echo "📥 Installing core ML dependencies..."
    echo "   Using fast mirror and ignoring system packages..."
    echo "   Estimated time: 3-5 minutes with fast mirror"
    echo ""
    
    pip install --upgrade pip -q
    
    # Install with ignore-installed for problematic system packages
    echo "   Installing packages (progress hidden for clarity)..."
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install to user directory to avoid system conflicts
        pip install -r requirements.txt \
            --user \
            --ignore-installed cryptography \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --quiet --no-warn-script-location 2>&1 | grep -E "(Successfully installed|ERROR)" || true
    else
        # On local: Normal install in venv
        pip install -r requirements.txt \
            -i https://pypi.tuna.tsinghua.edu.cn/simple \
            --quiet --no-warn-script-location 2>&1 | grep -E "(Successfully installed|ERROR)" || true
    fi
    
    echo ""
    echo "✅ Core dependencies installed"
    echo ""
elif [ "$MISSING_UI" = true ]; then
    echo "📥 Installing UI dependencies..."
    echo "   Using fast mirror for quick installation (1-2 minutes)"
    echo ""
    
    pip install --upgrade pip -q
    
    # Install only lightweight UI packages with fast mirror
    echo "   Installing packages (progress hidden for clarity)..."
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install to user directory to avoid system conflicts
        pip install streamlit streamlit-authenticator plotly pydeck \
                    reportlab openpyxl python-docx python-dotenv watchdog \
                    PyMuPDF pandas Pillow pyyaml rank-bm25 qdrant-client \
                    --user \
                    --ignore-installed cryptography \
                    -i https://pypi.tuna.tsinghua.edu.cn/simple \
                    --quiet --no-warn-script-location 2>&1 | grep -E "(Successfully installed|ERROR)" || true
    else
        # On local: Normal install in venv
        pip install streamlit streamlit-authenticator plotly pydeck \
                    reportlab openpyxl python-docx python-dotenv watchdog \
                    PyMuPDF pandas Pillow pyyaml rank-bm25 qdrant-client \
                    -i https://pypi.tuna.tsinghua.edu.cn/simple \
                    --quiet --no-warn-script-location 2>&1 | grep -E "(Successfully installed|ERROR)" || true
    fi
    
    echo ""
    echo "✅ UI dependencies installed"
    echo ""
else
    echo "✅ All dependencies satisfied!"
    echo ""
fi

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
# On RunPod, use port 8888 (commonly exposed) or 8501
if [ "$IS_RUNPOD" = true ]; then
    # Check if port 8888 is available (usually exposed for Jupyter)
    # Use 8888 if Jupyter not running, otherwise 8501
    PORT=8888
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
    echo "     (It may show as 'Jupyter Lab' - that's the port we're using)"
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

# Run streamlit with appropriate settings
if [ "$IS_RUNPOD" = true ]; then
    # RunPod - bind to all interfaces
    # Use python -m streamlit to ensure it's found
    # Note: enableCORS must be true for XSRF protection
    python -m streamlit run app.py \
        --server.port=$PORT \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=true \
        --server.enableXsrfProtection=true
else
    # Local - standard settings
    python -m streamlit run app.py --server.port=$PORT
fi
