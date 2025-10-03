#!/bin/bash

# RunPod Setup Script for RAG App
echo "🚀 Setting up RAG App on RunPod..."

# Update system packages
sudo apt update && sudo apt upgrade -y

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

# Create necessary directories
mkdir -p storage
mkdir -p data

# Set up environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

echo "✅ Setup complete! Ready to run the RAG app."
echo "📝 Next steps:"
echo "1. Run: python ingest.py (to create vector database)"
echo "2. Run: streamlit run app.py --server.port 8501 --server.address 0.0.0.0"
