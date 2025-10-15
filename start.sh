#!/bin/bash

# DuraFlex Technical Assistant - Streamlined Setup Script
# Works on local machines and RunPod GPU instances
# Usage: ./start.sh

set -e  # Exit on error

echo "=========================================="
echo "🔧 DuraFlex Technical Assistant"
echo "=========================================="
echo ""

# Set Claude API key for LLM answer generation
export ANTHROPIC_API_KEY=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

# Detect environment
IS_RUNPOD=false
if [ -d "/runpod-volume" ] || [ -d "/workspace" ] || [ ! -z "$RUNPOD_POD_ID" ]; then
    IS_RUNPOD=true
    echo "🖥️  Environment: RunPod GPU Instance"
    echo "📍 GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"
    echo ""
else
    echo "🖥️  Environment: Local Machine"
    echo ""
fi

# Check Python environment
echo "📦 Using system Python environment"
echo "   (Keeps access to pre-installed PyTorch & ML libraries)"
echo "🐍 Python version: $(python --version | cut -d' ' -f2)"
echo ""

# Check dependencies
echo "🔍 Checking dependencies..."
echo ""

# Check if packages are installed
echo "   Python: $(which python)"
echo "   Checking if packages are installed via pip..."

# Check PyTorch
if python -c "import torch; print(f'✅ PyTorch: {torch.__version__}')" 2>/dev/null; then
    :
else
    echo "  ❌ PyTorch not found"
    echo "     Please install PyTorch first"
    exit 1
fi

# Check other packages
python -c "
import sys
packages = [
    ('transformers', 'Transformers'),
    ('llama_index', 'LlamaIndex'),
    ('sentence_transformers', 'Sentence-Transformers'),
    ('streamlit', 'Streamlit'),
    ('fitz', 'PyMuPDF'),
    ('rank_bm25', 'rank-bm25')
]

for package, name in packages:
    try:
        if package == 'fitz':
            import fitz
        else:
            __import__(package)
        print(f'  ✅ {name}')
    except ImportError:
        print(f'  ❌ {name} not found')
        sys.exit(1)
"

if [ $? -eq 0 ]; then
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
    echo "⚠️  User configuration not found"
    echo "     Creating default user configuration..."
    mkdir -p config
    cat > config/users.yaml << EOF
users:
  admin:
    username: "admin"
    name: "Administrator"
    password: "admin123"
    role: "admin"
  tech1:
    username: "tech1"
    name: "Technician"
    password: "tech123"
    role: "technician"
EOF
    echo "✅ Default user configuration created"
    echo ""
fi

# Check if RAG index exists
if [ -d "storage" ] && [ -f "storage/index_store.json" ]; then
    echo "✅ RAG index found in /workspace/storage/"
    echo "   📊 Indexed chunks: $(find storage -name "*.json" -exec wc -l {} + | tail -1 | awk '{print $1}')"
    echo ""
else
    echo "⚠️  RAG index not found"
    echo "     Please run the indexing process first"
    echo "     Run: python index_documents.py"
    echo ""
fi

# Set port based on environment
if [ "$IS_RUNPOD" = true ]; then
    PORT=8501
    echo "📍 Using port 8501 (RunPod HTTP Service port)"
    echo ""
    echo "=========================================="
    echo "🌐 RunPod Network Configuration"
    echo "=========================================="
    echo ""
    echo "The app will run on port 8501"
    echo ""
    echo "To access from your browser:"
    echo "  1. Go to your RunPod pod page"
    echo "  2. Under 'Connect' → 'HTTP Services'"
    echo "  3. Click on the port 8501 service link"
    echo ""
    echo "💡 The URL will look like:"
    echo "   https://xxxxx-8501.proxy.runpod.net"
    echo ""
    echo "=========================================="
    echo "🔐 Login Credentials"
    echo "=========================================="
    echo ""
    echo "  Admin:       admin / admin123"
    echo "  Technician:  tech1 / tech123"
    echo ""
    echo "=========================================="
    echo ""
else
    PORT=8501
    echo "📍 Using port 8501"
    echo ""
fi

echo "🚀 Starting Streamlit server..."
echo ""
echo "Press Ctrl+C to stop the server"
echo ""
echo "=========================================="
echo ""

# Start Streamlit
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
