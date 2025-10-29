# Quick Fix Guide for Docker Container Issues

## Issues Found

### 1. ✅ FIXED: HuggingFace Permission Errors
**Problem:** Container runs as `appuser` but HuggingFace tries to write to `/root/.cache/`

**Solution:** Added environment variables to use `/app/.cache/huggingface` instead:
```dockerfile
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface
ENV HF_DATASETS_CACHE=/app/.cache/huggingface
```

### 2. ✅ FIXED: Claude API Key Not Passed
**Problem:** `ANTHROPIC_API_KEY` needs to be passed to the container

**Solution:** Updated `run-local.ps1` to automatically pass the API key if it's set in your environment

## How to Use

### Option 1: Set API Key in PowerShell (Recommended)
```powershell
# Set your Claude API key
$env:ANTHROPIC_API_KEY = "sk-ant-api03-..."

# Run the container
.\run-local.ps1
```

### Option 2: Use .env File
Create a `.env` file in your project root:
```
ANTHROPIC_API_KEY=sk-ant-api03-...
```

The startup script will automatically load it.

### Option 3: Pass via Docker Run Command
```powershell
docker run -it --rm `
    -p 8501:8501 `
    -e ANTHROPIC_API_KEY="sk-ant-api03-..." `
    -e PYTHONPATH=/app `
    rag-app:local
```

## Rebuild Required

After updating the Dockerfile, rebuild the image:
```powershell
.\build-local.ps1
```

## What Each Error Means

### Permission Errors (FIXED)
- **Before:** `Permission denied: '/root/.cache/huggingface/hub/...'`
- **After:** Models will cache to `/app/.cache/huggingface` which appuser can write to
- **Impact:** Models will download and cache correctly

### Claude Connection Errors
- **Cause:** Missing `ANTHROPIC_API_KEY` environment variable
- **Impact:** Claude features disabled (intent classification, document evaluation, answer generation)
- **Fallback:** Application uses pattern-matching fallbacks instead
- **Fix:** Set API key as shown above

## Testing

After rebuilding and running:
1. Check logs for: `✅ Claude Intent Classifier initialized`
2. Check logs for: `✅ Claude Answer Generator initialized`
3. If you see warnings, API key isn't set correctly

