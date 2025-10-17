# RunPod Serverless Deployment Guide

This guide explains how to deploy your RAG app to RunPod Serverless.

## Files Overview

- `handler.py` - Main entrypoint for RunPod Serverless
- `storage/` - Directory containing the vector database (created by `ingest.py`)
- `requirements.txt` - Python dependencies
- `test_handler.py` - Local testing script

## Deployment Steps

### 1. Prepare Your Files

Ensure you have run `ingest.py` to create the vector database:
```bash
python ingest.py
```

This creates the `storage/` directory with your vector index.

### 2. Upload to RunPod

Upload these files to your RunPod Serverless environment:
- `handler.py` (main entrypoint)
- `storage/` directory (vector database)
- `requirements.txt`

### 3. Configure RunPod Serverless

In your RunPod Serverless template:
- **Handler**: `handler.handler`
- **Python Version**: 3.9+ (recommended 3.10)
- **GPU**: Optional but recommended for faster inference

### 4. Environment Variables

No special environment variables are required. The handler uses:
- HuggingFace models (downloaded automatically)
- Local storage directory for the vector index

### 5. Test Your Deployment

Send a test request to your RunPod endpoint:

```json
{
  "input": {
    "question": "What is the DuraFlex system?"
  }
}
```

Expected response:
```json
{
  "output": "Detailed answer about DuraFlex system..."
}
```

## Local Testing

Before deploying, test locally:

```bash
python test_handler.py
```

This will run various test cases to ensure the handler works correctly.

## Performance Notes

- First request may be slower due to model loading
- Subsequent requests will be faster
- GPU acceleration is used when available
- Vector index is loaded once and cached in memory

## Troubleshooting

### Common Issues

1. **"Storage directory not found"**
   - Run `python ingest.py` to create the vector database

2. **"Model download failed"**
   - Ensure internet connection for initial model download
   - Check HuggingFace Hub access

3. **Memory issues**
   - Consider using smaller models
   - Increase RunPod memory allocation

### Logs

The handler includes detailed logging. Check RunPod logs for:
- Model initialization status
- Index loading confirmation
- Query processing details
- Error messages with context

## Customization

To modify the handler:

1. **Change embedding model**: Update `model_name` in `initialize_models()`
2. **Change LLM model**: Update `model_name` in `initialize_models()`
3. **Adjust response parameters**: Modify `generate_kwargs` in the LLM configuration
4. **Change retrieval settings**: Update `similarity_top_k` in `query_index()`
