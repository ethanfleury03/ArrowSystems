# RunPod Setup Instructions for RAG App

## 🚀 Complete Step-by-Step Guide

### 1. **Upload Your Files to RunPod**
```bash
# Upload the entire rag_app.py folder to your RunPod instance
# You can use RunPod's file manager or SCP/SFTP
```

### 2. **Connect to RunPod Terminal**
- Open your RunPod instance
- Access the terminal/SSH

### 3. **Navigate to Your App Directory**
```bash
cd rag_app.py
```

### 4. **Make Setup Script Executable**
```bash
chmod +x run_setup.sh
```

### 5. **Run the Setup Script**
```bash
./run_setup.sh
```

### 6. **Create Vector Database (First Time Only)**
```bash
python ingest.py
```

### 7. **Start the RAG App**
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### 8. **Access Your App**
- Open RunPod's web interface
- Go to the "Connect" tab
- Click on the Streamlit port (8501)
- Your RAG app will open in a new tab!

## 🔧 **Troubleshooting**

### If you get CUDA errors:
```bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
```

### If models don't download:
```bash
# Clear HuggingFace cache and retry
rm -rf ~/.cache/huggingface
python ingest.py
```

### If you need to reset everything:
```bash
rm -rf storage/
python ingest.py
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

## 📁 **File Structure (Should Look Like This)**
```
rag_app.py/
├── app.py
├── ingest.py
├── requirements.txt
├── run_setup.sh
├── runpod_instructions.md
├── data/ (your PDF files)
├── storage/ (vector database - created after ingest.py)
└── README.md
```

## 🎯 **Expected Performance**
- **Model Download**: 5-10 minutes (first time only)
- **App Startup**: 30-60 seconds
- **Query Response**: 2-5 seconds per question
- **GPU Usage**: RTX 4090 will handle this easily!

## 💡 **Pro Tips**
1. **Keep the terminal open** while the app is running
2. **Use Ctrl+C** to stop the app when needed
3. **Check GPU usage** with `nvidia-smi` if needed
4. **The app will remember** your vector database between sessions
