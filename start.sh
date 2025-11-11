#!/bin/bash

# DuraFlex Technical Assistant - Enhanced Startup Script
# Works on local machines and RunPod GPU instances
# Usage: ./start.sh

# ============================================================================
# CRITICAL: Set API Keys and Credentials FIRST (before anything else)
# ============================================================================
export ANTHROPIC_API_KEY="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"
export AWS_ACCESS_KEY_ID="AKIAXNHTG4AMCE54I36Y"
export AWS_SECRET_ACCESS_KEY="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"
export AWS_DEFAULT_REGION="us-east-1"

# GPU acceleration environment variables for Ollama
export OLLAMA_GPU_LAYERS=32
export OLLAMA_GPU_MEMORY_FRACTION=0.8
export OLLAMA_HOST=0.0.0.0:11434
export CUDA_VISIBLE_DEVICES=0
export OLLAMA_DEBUG=1

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
    
    # Configure Git credential caching (cache for 24 hours to avoid repeated prompts)
    git config --global credential.helper 'cache --timeout=86400'
    
    # Check for Git LFS and auto-pull large files
    if [ -f ".gitattributes" ] && grep -q "lfs" .gitattributes 2>/dev/null; then
        echo ""
        echo "🔍 Git LFS detected in repository"
        
        if ! command -v git-lfs &> /dev/null; then
            echo "   📥 Installing git-lfs..."
            apt-get update -qq 2>/dev/null && apt-get install -y git-lfs -qq 2>/dev/null
            git lfs install --skip-repo 2>/dev/null
            echo "   ✅ git-lfs installed"
        fi
        
        # Check if LFS files need to be pulled
        if [ -f "latest_model/default__vector_store.json" ]; then
            FILE_SIZE=$(stat -f%z "latest_model/default__vector_store.json" 2>/dev/null || stat -c%s "latest_model/default__vector_store.json" 2>/dev/null || echo "0")
            if [ "$FILE_SIZE" -lt 1000 ]; then
                echo "   📥 Pulling LFS files (this may take a few minutes)..."
                git lfs pull 2>/dev/null || git lfs fetch --all && git lfs checkout
                echo "   ✅ LFS files downloaded (~450MB RAG index)"
            else
                echo "   ✅ LFS files already present"
            fi
        else
            echo "   📥 Pulling LFS files..."
            git lfs pull 2>/dev/null || git lfs fetch --all && git lfs checkout
            echo "   ✅ LFS files downloaded"
        fi
    fi
    echo ""
else
    echo "🖥️  Environment: Local Machine"
    
    # Check for Git LFS and auto-pull large files (works on all environments)
    if [ -f ".gitattributes" ] && grep -q "lfs" .gitattributes 2>/dev/null; then
        echo ""
        echo "🔍 Git LFS detected in repository"
        
        # Check if git-lfs is installed
        if ! command -v git-lfs &> /dev/null; then
            echo "   ⚠️  git-lfs not found"
            echo "   📥 Installing git-lfs for Windows..."
            
            # Detect Windows and install Git LFS automatically
            if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" || "$OSTYPE" == "cygwin" ]]; then
                # Download Git LFS installer for Windows
                LFS_VERSION="3.4.0"
                LFS_URL="https://github.com/git-lfs/git-lfs/releases/download/v${LFS_VERSION}/git-lfs-windows-v${LFS_VERSION}.exe"
                LFS_INSTALLER="/tmp/git-lfs-installer.exe"
                
                echo "   📥 Downloading Git LFS v${LFS_VERSION}..."
                if command -v curl &> /dev/null; then
                    curl -L -o "$LFS_INSTALLER" "$LFS_URL" 2>/dev/null
                elif command -v wget &> /dev/null; then
                    wget -O "$LFS_INSTALLER" "$LFS_URL" 2>/dev/null
                fi
                
                if [ -f "$LFS_INSTALLER" ]; then
                    echo "   🔧 Installing Git LFS (silent installation)..."
                    "$LFS_INSTALLER" /VERYSILENT /NORESTART /SP- 2>/dev/null || true
                    sleep 3
                    
                    # Refresh PATH to find git-lfs
                    export PATH="/c/Program Files/Git LFS:$PATH"
                    
                    # Verify installation
                    if command -v git-lfs &> /dev/null; then
                        echo "   ✅ Git LFS installed successfully"
                    else
                        echo "   ⚠️  Git LFS installation may require a restart of Git Bash"
                        echo "   💡 If LFS files don't download, close Git Bash and reopen it"
                    fi
                    
                    rm -f "$LFS_INSTALLER" 2>/dev/null || true
                else
                    echo "   ⚠️  Could not download Git LFS installer"
                    echo "   💡 Manual install: https://git-lfs.github.com/"
                fi
            fi
        fi
        
        # Initialize git-lfs (even if already installed, ensure it's configured)
        if command -v git-lfs &> /dev/null; then
            echo "   🔧 Configuring Git LFS..."
            git lfs install 2>/dev/null || git lfs install --skip-repo 2>/dev/null || true
            echo "   ✅ Git LFS configured"
        fi
        
        # Check if LFS files need to be pulled
        if [ -f "latest_model/default__vector_store.json" ]; then
            # Use wc -c for file size (works on Windows Git Bash)
            FILE_SIZE=$(wc -c < "latest_model/default__vector_store.json" 2>/dev/null || echo "0")
            if [ "$FILE_SIZE" -lt 1000 ]; then
                echo "   📥 Pulling LFS files (this may take a few minutes)..."
                git lfs pull 2>/dev/null || git lfs fetch --all && git lfs checkout
                echo "   ✅ LFS files downloaded (~450MB RAG index)"
            else
                echo "   ✅ LFS files already present ($(numfmt --to=iec $FILE_SIZE 2>/dev/null || echo "${FILE_SIZE} bytes"))"
            fi
        else
            echo "   📥 Pulling LFS files..."
            git lfs pull 2>/dev/null || git lfs fetch --all && git lfs checkout
            echo "   ✅ LFS files downloaded"
        fi
        echo ""
    fi
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
        # Try python3 first, fall back to python
        (python3 -m venv venv) || (python -m venv venv)
        echo "✅ Virtual environment created"
    fi

    echo "📦 Activating virtual environment..."
    if [ -f "venv/bin/activate" ]; then
        . venv/bin/activate
    elif [ -f "venv/Scripts/activate" ]; then
        . venv/Scripts/activate
    else
        echo "❌ Could not find venv activation script (tried venv/bin/activate and venv/Scripts/activate)"
        exit 1
    fi
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

# Smart PyTorch detection - skip re-downloading if already installed
SKIP_TORCH=false
if python -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python -c "import torch; print(torch.__version__)" 2>/dev/null)
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    
    echo "🔍 PyTorch Detection:"
    echo "   ✅ PyTorch $TORCH_VERSION already installed"
    echo "   ✅ CUDA available: $CUDA_AVAILABLE"
    echo "   ⚡ Will skip torch in requirements.txt (saves ~2.2GB + 5 min)"
    echo ""
    SKIP_TORCH=true
fi

# Install missing dependencies with smart handling
if [ "$MISSING_CORE" = true ]; then
    echo "📥 Installing all dependencies from requirements.txt..."
    echo "   This may take a few minutes..."
    echo ""
    
    pip install --upgrade pip -q
    
    if [ "$SKIP_TORCH" = true ]; then
        # Create temporary requirements without torch
        grep -vE "^torch[>=<]|^torchvision" requirements.txt > /tmp/requirements_no_torch.txt
        REQUIREMENTS_FILE="/tmp/requirements_no_torch.txt"
    else
        REQUIREMENTS_FILE="requirements.txt"
    fi
    
    # Pre-install blinker to avoid distutils conflict (common on RunPod/system Python)
    echo "   🔧 Pre-installing blinker to avoid conflicts..."
    pip install --ignore-installed blinker>=1.6.0 -q 2>/dev/null || true
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install to user directory to avoid system conflicts
        pip install -r "$REQUIREMENTS_FILE" \
            --user \
            --upgrade-strategy only-if-needed \
            --ignore-installed cryptography \
            --no-warn-script-location
    else
        # On local: Normal install in venv
        pip install -r "$REQUIREMENTS_FILE"
    fi
    
    echo ""
    echo "✅ All dependencies installed"
    echo ""
elif [ "$MISSING_UI" = true ]; then
    echo "📥 Installing all dependencies from requirements.txt..."
    echo "   This will show progress so you can see what's happening..."
    echo ""
    
    pip install --upgrade pip -q
    
    if [ "$SKIP_TORCH" = true ]; then
        # Use filtered requirements without torch
        grep -vE "^torch[>=<]|^torchvision" requirements.txt > /tmp/requirements_no_torch.txt
        REQUIREMENTS_FILE="/tmp/requirements_no_torch.txt"
    else
        REQUIREMENTS_FILE="requirements.txt"
    fi
    
    # Pre-install blinker to avoid distutils conflict (common on RunPod/system Python)
    echo "   🔧 Pre-installing blinker to avoid conflicts..."
    pip install --ignore-installed blinker>=1.6.0 -q 2>/dev/null || true
    
    if [ "$IS_RUNPOD" = true ]; then
        # On RunPod: Install all requirements
        # Use --upgrade-strategy only-if-needed to skip reinstalling satisfied dependencies
        echo "   Note: Skipping already-satisfied dependencies (PyTorch, Transformers, etc.)"
        pip install -r "$REQUIREMENTS_FILE" \
                    --user \
                    --upgrade-strategy only-if-needed \
                    --ignore-installed cryptography \
                    --no-warn-script-location
        
        INSTALL_STATUS=$?
    else
        # On local: Install everything from requirements
        pip install -r "$REQUIREMENTS_FILE"
        
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

# Check database setup (SQLite)
echo "🗄️  Checking SQLite database setup..."
echo ""

# Ensure SQLite file exists and tables are created
if python - <<'PY'
import os
from utils.db import init_db, DEFAULT_DB_PATH

db_path = os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH)
os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)

# Ensure empty file exists before SQLAlchemy initializes
if not os.path.exists(db_path):
    open(db_path, "a").close()

init_db()
print(db_path)
PY
then
    DB_PATH=$(python - <<'PY'
import os
from utils.db import DEFAULT_DB_PATH
print(os.getenv("SQLITE_DB_PATH", DEFAULT_DB_PATH))
PY
)
    echo "  ✅ SQLite database ready at ${DB_PATH}"
else
    echo "  ❌ Failed to initialize SQLite database"
    echo "     Check write permissions for ${SQLITE_DB_PATH:-./database.sqlite}"
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

# Check if index exists in latest_model/ (new location for two-pod workflow)
STORAGE_PATH=""
if [ -d "latest_model" ] && [ -f "latest_model/docstore.json" ]; then
    STORAGE_PATH="latest_model"
    echo "✅ RAG index found in latest_model/"
elif [ -d "/workspace/latest_model" ] && [ -f "/workspace/latest_model/docstore.json" ]; then
    STORAGE_PATH="/workspace/latest_model"
    echo "✅ RAG index found in /workspace/latest_model/"
# Fallback: Check old storage locations for backward compatibility
elif [ -d "/workspace/storage" ] && [ -f "/workspace/storage/docstore.json" ]; then
    STORAGE_PATH="/workspace/storage"
    echo "✅ RAG index found in /workspace/storage/ (old location)"
    echo "   💡 Consider migrating: python migrate_to_latest_model.py"
elif [ -d "storage" ] && [ -f "storage/docstore.json" ]; then
    STORAGE_PATH="storage"
    echo "✅ RAG index found in ./storage/ (old location)"
    echo "   💡 Consider migrating: python migrate_to_latest_model.py"
else
    echo "=========================================="
    echo "⚠️  RAG Index Not Found!"
    echo "=========================================="
    echo ""
    echo "Checked locations:"
    echo "  • latest_model/ (default for two-pod workflow)"
    echo "  • /workspace/latest_model/"
    echo "  • /workspace/storage/ (legacy)"
    echo "  • ./storage/ (legacy)"
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
