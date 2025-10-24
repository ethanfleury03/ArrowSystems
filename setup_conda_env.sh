#!/bin/bash

# =============================================================================
# Arrow Systems AI/RAG Development Environment Setup
# =============================================================================
# This script creates a reproducible Python 3.11 environment with Conda
# for AI/RAG development with GPU acceleration support.

set -e  # Exit on any error

echo "🚀 Setting up Arrow Systems AI/RAG Development Environment"
echo "=========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    print_error "This script is designed for Linux systems only."
    exit 1
fi

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    print_status "Conda not found. Installing Miniconda..."
    
    # Download and install Miniconda
    cd /tmp
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
    chmod +x miniconda.sh
    
    # Install Miniconda silently
    ./miniconda.sh -b -p $HOME/miniconda3
    
    # Initialize conda for bash
    $HOME/miniconda3/bin/conda init bash
    
    # Add conda to PATH for current session
    export PATH="$HOME/miniconda3/bin:$PATH"
    
    print_success "Miniconda installed successfully"
else
    print_success "Conda is already installed"
    export PATH="$(conda info --base)/bin:$PATH"
fi

# Update conda
print_status "Updating conda..."
conda update -n base -c defaults conda -y

# Remove existing environment if it exists
if conda env list | grep -q "arrow"; then
    print_warning "Environment 'arrow' already exists. Removing it..."
    conda env remove -n arrow -y
fi

# Create new conda environment with Python 3.11
print_status "Creating conda environment 'arrow' with Python 3.11..."
conda create -n arrow python=3.11 -y

# Activate the environment
print_status "Activating environment 'arrow'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate arrow

# Install PyTorch with CUDA 12.1 support
print_status "Installing PyTorch 2.5.1 with CUDA 12.1 support..."
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install core ML packages
print_status "Installing core ML packages..."
conda install -c conda-forge numpy=1.26.4 scikit-learn=1.7.2 pandas=2.1.4 -y

# Install HuggingFace ecosystem
print_status "Installing HuggingFace ecosystem..."
pip install transformers==4.57.1
pip install sentence-transformers==5.1.2
pip install accelerate==1.11.0
pip install huggingface-hub==0.36.0
pip install safetensors==0.6.2

# Install LlamaIndex ecosystem
print_status "Installing LlamaIndex ecosystem..."
pip install llama-index==0.14.5
pip install llama-index-core==0.14.5
pip install llama-index-embeddings-huggingface==0.6.1
pip install llama-index-llms-huggingface==0.6.1
pip install llama-index-readers-file==0.5.4

# Install API clients
print_status "Installing API clients..."
pip install anthropic==0.71.0
pip install openai==2.6.1

# Install web frameworks
print_status "Installing web frameworks..."
pip install fastapi==0.120.0
pip install "uvicorn[standard]==0.38.0"
pip install streamlit==1.50.0
pip install streamlit-authenticator==0.4.2

# Install document processing
print_status "Installing document processing libraries..."
pip install PyMuPDF==1.26.5
pip install reportlab==4.4.4
pip install openpyxl==3.1.5
pip install python-docx==1.2.0

# Install visualization
print_status "Installing visualization libraries..."
pip install plotly==6.3.1
pip install pydeck==0.9.1
pip install Pillow==11.3.0

# Install utilities
print_status "Installing utility packages..."
pip install python-dotenv==1.1.1
pip install watchdog==6.0.0
pip install tqdm==4.67.1
pip install nltk==3.9.2
pip install rank-bm25==0.2.2
pip install symspellpy==6.9.0
pip install tabulate==0.9.0
pip install pyyaml==6.0.3

# Install database and cloud
print_status "Installing database and cloud packages..."
pip install boto3==1.40.58

# Verify installation
print_status "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
python -c "import streamlit; print(f'Streamlit version: {streamlit.__version__}')"
python -c "import fastapi; print(f'FastAPI version: {fastapi.__version__}')"

print_success "Environment setup completed successfully!"
echo ""
echo "🎉 Arrow Systems AI/RAG Environment is ready!"
echo "=============================================="
echo ""
echo "To activate the environment, run:"
echo "  conda activate arrow"
echo ""
echo "To deactivate the environment, run:"
echo "  conda deactivate"
echo ""
echo "To test GPU support, run:"
echo "  python -c \"import torch; print(f'CUDA available: {torch.cuda.is_available()}')\""
echo ""
echo "Environment location: $(conda info --envs | grep arrow | awk '{print $2}')"
