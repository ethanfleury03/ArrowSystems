# RAG Query Issue Analysis & Fix Plan

## Problem Summary

The `/query` endpoint is returning raw chunk dumps with "According to <pdf> [n]..." instead of synthesized LLM answers.

## Root Cause Analysis

### 1. "According to..." Formatting Location

**File**: `backend/orchestrator.py`  
**Function**: `ResponseGenerator._build_chunk_based_answer()`  
**Line**: 2155

```python
answer_parts.append(f"According to {source_name} {source_id}:\n{combined_text}")
```

**Condition**: This fallback is used when:
- `answer_generator` is `None`, OR
- `answer_generator.claude_client` is `None`

**Code Path**: 
```
/query → rag_pipeline.query() → orchestrator.orchestrate_query() 
→ response_generator.generate_structured_response() 
→ _build_answer() → _build_chunk_based_answer() [FALLBACK]
```

### 2. LLM Synthesis Not Being Called

**File**: `backend/orchestrator.py`  
**Function**: `ResponseGenerator._build_answer()`  
**Line**: 2096

```python
if answer_generator and answer_generator.claude_client:
    # LLM synthesis happens here
else:
    # Falls back to chunk-based answer
    return self._build_chunk_based_answer(query, context, intent)
```

**Issue**: In `orchestrate_query()` at line 4892, `answer_generator=self.answer_generator` is passed, but:
- If `LLM_WARMUP_ON_STARTUP != "1"`, `self.answer_generator` is initialized as `None` (lazy init)
- The lazy initialization (`_ensure_answer_generator()`) is **never called** before passing to `generate_structured_response()`
- Even if initialized, `claude_client` might be `None` if:
  - `ANTHROPIC_API_KEY` is missing
  - Claude API connection fails during initialization
  - Anthropic package not installed

### 3. Prompt Loading (Not the Issue)

The "2 prompts are loaded" message is likely from a different component (query rewriting/decomposition). The answer synthesis uses **hardcoded prompts** in `ClaudeAnswerGenerator._build_answer_prompt()` (line 3391), not a prompt registry.

### 4. Reranker Loading Issue

**File**: `backend/orchestrator.py`  
**Function**: `RAGOrchestrator.initialize_models()`  
**Lines**: 3850-3906

**Problem**: CrossEncoder from `sentence-transformers` may not fully respect `HF_HUB_OFFLINE=1` and might attempt network calls if:
- Cache directory structure doesn't match expected format
- Model files are missing (config.json, model.safetensors, etc.)
- Cache directory path mismatch between build-time and runtime

**Current Cache Dir**: `/app/.cache/huggingface` (from Dockerfile)  
**Runtime Cache Dir**: Set via `self.cache_dir` (default: `/root/.cache/huggingface/hub`)

**Mismatch Risk**: Dockerfile downloads to `/app/.cache/huggingface`, but runtime might look in `/root/.cache/huggingface/hub`.

### 5. Dockerfile Verification

**File**: `deployment/Dockerfile.api`  
**Lines**: 47-102

The Dockerfile DOES pre-download the reranker, but:
- Uses `cache_folder=cache` where `cache = os.getenv("SENTENCE_TRANSFORMERS_HOME") or os.getenv("HF_HOME") or "/app/.cache/huggingface"`
- Downloads to `/app/.cache/huggingface` (not `/app/.cache/huggingface/hub`)
- Runtime code expects cache in `self.cache_dir` which defaults to `/root/.cache/huggingface/hub`

## Fixes Required

### Fix 1: Ensure Answer Generator is Initialized

**File**: `backend/orchestrator.py`  
**Location**: `orchestrate_query()` method, before calling `generate_structured_response()`

**Change**: Call `_ensure_answer_generator()` before passing to response generator.

### Fix 2: Improve Reranker Offline Loading

**File**: `backend/orchestrator.py`  
**Location**: `initialize_models()` method, reranker loading section

**Changes**:
1. Verify cache directory structure before loading
2. Add explicit `local_files_only=True` if supported by CrossEncoder
3. Better error handling with cache verification
4. Ensure cache directory path matches Dockerfile

### Fix 3: Fix Cache Directory Path Consistency

**Files**: 
- `deployment/Dockerfile.api` (build-time cache)
- `backend/orchestrator.py` (runtime cache)

**Change**: Ensure both use the same cache directory path.

### Fix 4: Add Diagnostic Logging

**File**: `backend/orchestrator.py`

**Changes**:
1. Log when answer_generator is None
2. Log when claude_client is None
3. Log reranker cache verification
4. Add startup diagnostic endpoint

## Fixes Applied

### Fix 1: Ensure Answer Generator is Initialized ✅

**File**: `backend/orchestrator.py`  
**Location**: `orchestrate_query()` method, line ~4888

**Change**: Call `_ensure_answer_generator()` before passing to response generator.

```python
# CRITICAL FIX: Ensure answer generator is initialized before use
answer_generator = self._ensure_answer_generator()
if answer_generator and not answer_generator.claude_client:
    logger.warning(...)
response = self.response_generator.generate_structured_response(
    ...
    answer_generator=answer_generator,  # Use ensured answer generator
    ...
)
```

### Fix 2: Enhanced Logging in _build_answer() ✅

**File**: `backend/orchestrator.py`  
**Location**: `ResponseGenerator._build_answer()`, line ~2095

**Changes**:
- Log when `answer_generator` is None
- Log when `claude_client` is None
- Log successful LLM generation
- Better error logging with traceback

### Fix 3: Improved Reranker Offline Loading ✅

**File**: `backend/orchestrator.py`  
**Location**: `initialize_models()`, reranker loading section, line ~3850

**Changes**:
1. Verify cache directory structure before loading
2. Check for model files (config.json, *.safetensors, *.bin)
3. Normalize cache directory path to match Dockerfile
4. Set `HF_HUB_CACHE` environment variable explicitly
5. Better error handling with traceback logging

### Fix 4: Cache Directory Path Consistency ✅

**Files**: 
- `deployment/Dockerfile.api` (build-time cache)
- `backend/orchestrator.py` (runtime cache)

**Changes**:
1. Dockerfile now uses `/app/.cache/huggingface/hub` consistently
2. Runtime code checks for `/app/.cache/huggingface` and uses it if available
3. Both set `HF_HOME`, `HF_HUB_CACHE`, `TRANSFORMERS_CACHE`, `SENTENCE_TRANSFORMERS_HOME`

### Fix 5: Dockerfile Reranker Pre-download Enhancement ✅

**File**: `deployment/Dockerfile.api`  
**Location**: Model pre-download section, line ~56

**Changes**:
1. Verify reranker model files exist after download
2. Use consistent cache directory structure (`/app/.cache/huggingface/hub`)
3. Set all required environment variables during build
4. Make reranker download failure fatal (to catch issues early)

## Verification Checklist

### Pre-Deployment Verification

1. **Dockerfile Build Verification**:
   ```bash
   docker build -f deployment/Dockerfile.api -t rag-test .
   # Check logs for:
   # - "[BUILD] VERIFICATION_MARKER: reranker_download_done"
   # - Reranker model files found in cache
   ```

2. **Cache Directory Structure**:
   ```bash
   docker run --rm rag-test find /app/.cache/huggingface -name "*.safetensors" -o -name "*.bin" | grep reranker
   # Should show reranker model files
   ```

### Runtime Verification

1. **Reranker Loads Offline**:
   - Check logs for: `[RAG] reranker_model_load_done`
   - Should NOT see: `couldn't connect to https://huggingface.co`
   - Should see: `reranker_cache_verified`

2. **Answer Generator Initialization**:
   - Check logs for: `llm_answer_generation_start` (if API key is set)
   - If API key missing: `claude_client_not_available` warning
   - Should NOT see: `answer_generator_is_none` (after fix)

3. **Query Returns Synthesized Answer**:
   - Send test query to `/query`
   - Response should NOT contain: `"According to <pdf> [n]..."`
   - Response should be a clean, synthesized answer
   - If LLM unavailable, should see clear warning in logs

4. **Diagnostic Endpoint**:
   ```bash
   curl http://localhost:8080/api/model_cache_status
   # Should show:
   # - embedding_model.exists: true
   # - reranker_model.exists: true
   # - cache_dir matches Dockerfile location
   ```

### Expected Log Markers

**Successful LLM Synthesis**:
```
🤖 Generating LLM answer...
llm_answer_generated_successfully answer_length=...
```

**LLM Unavailable (Expected Fallback)**:
```
llm_answer_generation_skipped reason=claude_client_not_available
using_chunk_based_answer_fallback reason=llm_unavailable_or_failed
```

**Reranker Loaded Successfully**:
```
[RAG] reranker_cache_verified files_found=...
[RAG] reranker_model_load_done duration=...
reranker_loaded device=cpu cache_verified=true
```

**Reranker Load Failed**:
```
[RAG] reranker_cache_missing cache_dir=...
[RAG] reranker_load_FAILED: OSError: ...
```

## Files Modified

1. `backend/orchestrator.py`:
   - Line ~4888: Ensure answer generator before use
   - Line ~2095: Enhanced logging in `_build_answer()`
   - Line ~3850: Improved reranker loading with cache verification
   - Line ~3529: Cache directory normalization

2. `deployment/Dockerfile.api`:
   - Line ~56: Enhanced reranker pre-download with verification

## Testing Commands

```bash
# 1. Build image and verify models are cached
docker build -f deployment/Dockerfile.api -t rag-fix-test .
docker run --rm rag-fix-test ls -la /app/.cache/huggingface/hub/models--BAAI--bge-reranker-large/

# 2. Check diagnostic endpoint (after deployment)
curl http://your-service/api/model_cache_status | jq

# 3. Test query endpoint
curl -X POST http://your-service/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test question"}'

# 4. Check logs for verification markers
# Look for: reranker_cache_verified, llm_answer_generated_successfully
```

