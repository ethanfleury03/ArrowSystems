@echo off
REM Safe requirements installation script for Windows
REM Handles common package conflicts automatically

echo 🔧 Installing Python requirements with conflict handling...

REM Upgrade pip first
echo 📦 Upgrading pip...
python -m pip install --upgrade pip

REM Handle blinker conflict specifically
echo 🔧 Pre-installing blinker to avoid distutils conflict...
pip install --ignore-installed blinker>=1.6.0

REM Install all requirements
echo 📦 Installing requirements...
pip install -r requirements.txt

REM Verify critical packages
echo ✅ Verifying installation...
python -c "import streamlit; print(f'✓ Streamlit {streamlit.__version__}')" 2>nul || echo ⚠️  Streamlit not installed
python -c "import anthropic; print(f'✓ Anthropic {anthropic.__version__}')" 2>nul || echo ⚠️  Anthropic not installed
python -c "import torch; print(f'✓ PyTorch {torch.__version__}')" 2>nul || echo ⚠️  PyTorch not installed
python -c "from llama_index.core import VectorStoreIndex; print('✓ LlamaIndex')" 2>nul || echo ⚠️  LlamaIndex not installed

echo.
echo ✅ Installation complete!
pause

