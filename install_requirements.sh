#!/bin/bash
# Safe requirements installation script
# Handles common package conflicts automatically

echo "🔧 Installing Python requirements with conflict handling..."

# Upgrade pip first
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Handle blinker conflict specifically (common in RunPod/system Python)
echo "🔧 Pre-installing blinker to avoid distutils conflict..."
pip install --ignore-installed blinker>=1.6.0

# Install all requirements
echo "📦 Installing requirements..."
pip install -r requirements.txt

# Verify critical packages
echo "✅ Verifying installation..."
python -c "import streamlit; print(f'✓ Streamlit {streamlit.__version__}')" 2>/dev/null || echo "⚠️  Streamlit not installed"
python -c "import anthropic; print(f'✓ Anthropic {anthropic.__version__}')" 2>/dev/null || echo "⚠️  Anthropic not installed"
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>/dev/null || echo "⚠️  PyTorch not installed"
python -c "from llama_index.core import VectorStoreIndex; print('✓ LlamaIndex')" 2>/dev/null || echo "⚠️  LlamaIndex not installed"

echo ""
echo "✅ Installation complete!"

