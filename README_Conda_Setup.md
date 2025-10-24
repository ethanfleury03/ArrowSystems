# Arrow Systems AI/RAG Development Environment

This repository contains a complete, reproducible Python development environment for AI/RAG projects using Conda.

## 🚀 Quick Start

### Option 1: Automated Setup (Recommended)
```bash
# Make the setup script executable
chmod +x setup_conda_env.sh

# Run the automated setup
./setup_conda_env.sh
```

### Option 2: Manual Setup with environment.yml
```bash
# Create environment from YAML file
conda env create -f environment.yml

# Activate the environment
conda activate arrow
```

### Option 3: Manual Setup with requirements.txt
```bash
# Create new environment
conda create -n arrow python=3.11 -y
conda activate arrow

# Install PyTorch with CUDA
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y

# Install other dependencies
pip install -r requirements_conda.txt
```

## 📋 Environment Specifications

- **Python**: 3.11
- **PyTorch**: 2.5.1 with CUDA 12.1 support
- **CUDA**: 12.1 (GPU acceleration)
- **Environment Name**: `arrow`

## 🔧 Key Packages

### Core ML/AI
- PyTorch 2.5.1 (with CUDA 12.1)
- Transformers 4.57.1
- Sentence-Transformers 5.1.2
- Accelerate 1.11.0
- HuggingFace Hub 0.36.0
- Safetensors 0.6.2
- Scikit-learn 1.7.2

### LlamaIndex Ecosystem
- Llama-Index 0.14.5
- Llama-Index-Core 0.14.5
- Llama-Index-Embeddings-HuggingFace 0.6.1
- Llama-Index-LLMs-HuggingFace 0.6.1
- Llama-Index-Readers-File 0.5.4

### API Clients
- Anthropic 0.71.0
- OpenAI 2.6.1

### Web Frameworks
- FastAPI 0.120.0
- Uvicorn[standard] 0.38.0
- Streamlit 1.50.0
- Streamlit-Authenticator 0.4.2

### Document Processing
- PyMuPDF 1.26.5
- Pandas 2.1.4
- ReportLab 4.4.4
- OpenPyXL 3.1.5
- Python-DocX 1.2.0

### Visualization
- Plotly 6.3.1
- PyDeck 0.9.1
- Pillow 11.3.0

## 🎯 Usage Commands

### Activate Environment
```bash
conda activate arrow
```

### Deactivate Environment
```bash
conda deactivate
```

### Test GPU Support
```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Test Key Packages
```bash
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import streamlit; print(f'Streamlit: {streamlit.__version__}')"
python -c "import fastapi; print(f'FastAPI: {fastapi.__version__}')"
```

### Export Environment
```bash
# Export to YAML
conda env export -n arrow > environment_exported.yml

# Export to requirements.txt
conda list -n arrow --export > requirements_exported.txt
```

## 🔍 Troubleshooting

### CUDA Issues
If CUDA is not detected:
```bash
# Check CUDA installation
nvidia-smi

# Reinstall PyTorch with CUDA
conda activate arrow
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia -y
```

### Environment Conflicts
If you encounter conflicts:
```bash
# Remove existing environment
conda env remove -n arrow

# Recreate from scratch
conda env create -f environment.yml
```

### Package Updates
To update packages:
```bash
conda activate arrow
conda update --all
pip list --outdated
pip install --upgrade <package_name>
```

## 📁 File Structure

```
├── setup_conda_env.sh      # Automated setup script
├── environment.yml         # Conda environment definition
├── requirements_conda.txt  # Pip-compatible requirements
└── README_Conda_Setup.md  # This documentation
```

## 🎉 Benefits

- ✅ **Reproducible**: Exact same environment every time
- ✅ **GPU Support**: CUDA 12.1 acceleration
- ✅ **No Conflicts**: Conda handles dependency resolution
- ✅ **Fast Setup**: Automated installation
- ✅ **Cross-Platform**: Works on Linux, macOS, Windows
- ✅ **Isolated**: No system package conflicts

## 🚀 Next Steps

1. Run the setup script: `./setup_conda_env.sh`
2. Activate the environment: `conda activate arrow`
3. Test GPU support: `python -c "import torch; print(torch.cuda.is_available())"`
4. Start developing your AI/RAG application!

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Verify CUDA installation: `nvidia-smi`
3. Check conda installation: `conda --version`
4. Recreate environment if needed: `conda env remove -n arrow && conda env create -f environment.yml`
