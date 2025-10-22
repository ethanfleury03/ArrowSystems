#!/bin/bash
##########################################
# AI Models Download Script
# Run this once to pre-download AI models
# Saves 5-10 minutes on first app startup
##########################################

echo "=========================================="
echo "🤖 DuraFlex AI Models Setup"
echo "=========================================="
echo ""
echo "This will download AI models (~2GB)"
echo "Estimated time: 5-10 minutes"
echo "You only need to do this once!"
echo ""
echo "💡 Tip: Run this before leaving work,"
echo "   then the app starts instantly tomorrow!"
echo ""

# Detect Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "❌ Error: Python not found!"
    echo "   Please install Python 3.10 or newer"
    exit 1
fi

echo "🐍 Using Python: $($PYTHON --version)"
echo ""

# Check if models already exist
MODEL_DIR="$HOME/.cache/huggingface/hub"
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR 2>/dev/null)" ]; then
    echo "✅ Models directory exists and has content"
    echo "   Checking if models are complete..."
    echo ""
fi

# Download models
echo "📥 Downloading AI models..."
echo "   This may take 5-10 minutes depending on your internet speed"
echo ""

$PYTHON << 'PYTHON_SCRIPT'
import sys
import os

print("=" * 60)
print("Step 1/2: Downloading Embedding Model")
print("=" * 60)
print()

try:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding
    
    print("📦 Downloading BAAI/bge-large-en-v1.5 (~1.2 GB)...")
    embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-large-en-v1.5",
        cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
    )
    print("✅ Embedding model downloaded successfully!")
    print()
except Exception as e:
    print(f"❌ Error downloading embedding model: {e}")
    print()
    sys.exit(1)

print("=" * 60)
print("Step 2/2: Downloading Reranker Model")
print("=" * 60)
print()

try:
    from sentence_transformers import SentenceTransformer
    
    print("📦 Downloading cross-encoder/ms-marco-MiniLM-L-6-v2 (~80 MB)...")
    reranker = SentenceTransformer(
        'cross-encoder/ms-marco-MiniLM-L-6-v2',
        cache_folder=os.path.expanduser("~/.cache/huggingface/hub")
    )
    print("✅ Reranker model downloaded successfully!")
    print()
except Exception as e:
    print(f"❌ Error downloading reranker model: {e}")
    print()
    sys.exit(1)

print("=" * 60)
print("✅ ALL MODELS DOWNLOADED SUCCESSFULLY!")
print("=" * 60)
print()
print("📊 Models are cached in:")
print(f"   {os.path.expanduser('~/.cache/huggingface/hub')}")
print()
print("🚀 You can now run ./start.sh")
print("   The app will start in ~30 seconds (instead of 5-10 minutes)!")
print()
PYTHON_SCRIPT

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "🎉 Setup Complete!"
    echo "=========================================="
    echo ""
    echo "✅ AI models are ready!"
    echo ""
    echo "Next steps:"
    echo "  1. Run: ./start.sh"
    echo "  2. App will open in your browser"
    echo "  3. Start asking questions!"
    echo ""
    echo "💡 The models are now cached, so the app"
    echo "   will start quickly every time from now on."
else
    echo "❌ Setup Failed"
    echo "=========================================="
    echo ""
    echo "Please check the error messages above."
    echo "Common issues:"
    echo "  - No internet connection"
    echo "  - Missing Python packages (run: pip install -r requirements.txt)"
    echo "  - Disk space (need ~2GB free)"
fi
echo "=========================================="
echo ""

