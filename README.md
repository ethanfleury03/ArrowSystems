# RAG Streamlit App

A fully offline RAG (Retrieval-Augmented Generation) assistant that uses HuggingFace embeddings to query your company documents.

## Features

- 🤖 **Fully Offline**: No API keys required
- 📚 **Document Querying**: Ask questions about your 24 company documents
- 💬 **Chat Memory**: Remembers conversation context
- 🚀 **Fast**: Uses cached embeddings and vector index
- 🎨 **Modern UI**: Clean Streamlit interface

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Create Vector Index (if not already done)

```bash
python ingest.py
```

This will:
- Load all PDFs from the `data/` directory
- Create embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Build and persist a vector index in the `storage/` directory

### 3. Run the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Alternative Setup

You can also use the setup script:

```bash
python setup.py
```

This will install dependencies and create the index automatically.

## Troubleshooting

### Common Issues

1. **"RuntimeError: no running event loop"**
   - Fixed by properly configuring the embedding model and using `@st.cache_resource`

2. **"OpenAI API key errors"**
   - Fixed by explicitly setting the HuggingFace embedding model in Settings

3. **"Torch errors"**
   - Fixed by forcing CPU usage and proper dependency versions

4. **"Index loading errors"**
   - Make sure you've run `python ingest.py` first
   - Check that the `storage/` directory exists and contains index files

### Performance Tips

- The app uses CPU by default for stability
- If you have a GPU, you can modify `device="cpu"` to `device="cuda"` in `app.py`
- The embedding model is cached to avoid reloading on each query

## File Structure

```
rag_app.py/
├── app.py              # Main Streamlit application
├── ingest.py           # Script to create vector index
├── requirements.txt    # Python dependencies
├── setup.py           # Automated setup script
├── data/              # PDF documents (24 files)
└── storage/           # Vector index storage
    ├── docstore.json
    ├── index_store.json
    └── vector_store.json
```

## Dependencies

- `streamlit`: Web app framework
- `llama-index`: RAG framework
- `llama-index-embeddings-huggingface`: HuggingFace embeddings
- `sentence-transformers`: Embedding model
- `torch`: PyTorch for ML operations
- `transformers`: HuggingFace transformers

## Usage

1. Start the app with `streamlit run app.py`
2. Type questions about your company documents
3. The assistant will search through the documents and provide answers
4. Use "Reset Chat" to clear conversation history
5. Use "Reload Index" in the sidebar to refresh the vector index

## Tips for Better Results

- Ask specific questions about procedures, specifications, or troubleshooting
- Use keywords from your documents
- The assistant can answer follow-up questions with context
- Try questions like:
  - "How do I install the DuraFlex system?"
  - "What are the electrical requirements?"
  - "How do I troubleshoot print quality issues?"

