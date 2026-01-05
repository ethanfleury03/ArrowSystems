# Smoke Tests for Image Removal

This document describes how to validate that image ingestion is completely disabled and the RAG system doesn't expect images anywhere.

## Overview

The smoke test script (`scripts/smoke_no_images.py`) validates:
1. **No image ingestion**: Fresh index builds contain zero image nodes
2. **No image retrieval**: Query results never include image nodes
3. **Config validation**: Ensures `non_text.extract_images: false` is set

## Prerequisites

- Python 3.8+
- Dependencies installed (`pip install -r backend/requirements.txt`)
- Test documents (1-3 PDFs, optional DOCX files)

## Setting Up Test Documents

Create a small test folder with a few documents:

```bash
mkdir -p test_docs
# Copy 1-3 PDF files to test_docs/
cp data/some_document.pdf test_docs/
```

**Recommended**: Use small PDFs (1-5 pages) to keep test fast. The test will process all PDFs and DOCX files in the directory.

## Running the Smoke Test

### Basic Usage

```bash
python scripts/smoke_no_images.py test_docs/
```

### With Custom Options

```bash
python scripts/smoke_no_images.py test_docs/ \
    --config config.yaml \
    --persist-dir /tmp/test_index \
    --query "temperature regulation"
```

### Arguments

- `test_docs_dir` (required): Directory containing test PDF/DOCX files
- `--config` (default: `config.yaml`): Path to config file
- `--persist-dir` (default: temporary): Where to save the test index
- `--query` (default: `"temperature regulation"`): Test query to run

## Expected Output

### PASS Example

```
✅ Config check passed: extract_images is disabled

============================================================
STEP 1: Building index from test documents
============================================================
🔧 Initializing models...
📚 Building index from: test_docs
✅ Index built successfully

============================================================
STEP 2: Inspecting docstore.json for image nodes
============================================================
📊 Total nodes: 45
📊 Nodes by content_type:
   figure_caption: 2
   table: 3
   text: 40
✅ No image nodes found in docstore.json

============================================================
STEP 3: Running query and checking retrieval results
============================================================
🔍 Query: 'temperature regulation'
   Using TechnicalRAGPipeline.hybrid_search()
📊 Retrieved 10 results
✅ No image nodes in query results

============================================================
✅ PASS: All checks passed
============================================================
   - Config: extract_images disabled
   - Docstore: 45 total nodes, 0 image nodes
   - Query: 10 results, 0 image nodes
============================================================
```

### FAIL Examples

**Config has images enabled:**
```
❌ FAIL: Config has extract_images enabled (should be false)
   Edit config.yaml and set non_text.extract_images: false
```

**Image nodes found in docstore:**
```
📊 Nodes by content_type:
   image: 5
   text: 40

❌ FAIL: Found 5 image node(s) in docstore.json
   Expected: 0 image nodes
```

**Image nodes in query results:**
```
❌ FAIL: Query returned 2 image node(s):
   Result 3: node_id=abc123, content_type=image
   Result 7: node_id=def456, content_type=image
```

## What to Do If It Fails

### If Config Check Fails

1. Open `config.yaml`
2. Find the `non_text` section
3. Set `extract_images: false`
4. Re-run the test

### If Image Nodes Found in Docstore

This means ingestion is still creating image nodes. Check:

1. **Config is loaded correctly**: Verify `non_text.extract_images: false` in config.yaml
2. **Code changes applied**: Ensure `NonTextExtractor.extract_images_from_pdf()` has early return
3. **No cached results**: Delete any existing index and rebuild

### If Image Nodes in Query Results

This means retrieval is returning image nodes. Check:

1. **Defense-in-depth filtering**: Verify `HybridRetriever._filter_image_nodes()` is called
2. **Old index**: If using an old index, rebuild from scratch
3. **Content type filtering**: Ensure `hybrid_search()` excludes "image" from defaults

## Repo-Wide Audit

To check for any remaining image references in code:

```bash
# Search for image-related code
rg -n '"image"|content_type.*image|image__vector_store|extract_images|get_images\(' backend frontend
```

### What to Look For

**✅ OK (defense-in-depth):**
- Code that checks `content_type == "image"` to **exclude** images
- Comments mentioning images are excluded
- Optional file listings (e.g., `image__vector_store.json` as optional)

**❌ FIX (assumes images exist):**
- Default `content_types` lists that include `"image"`
- UI filters that expect image content
- Code paths that assume images will be present

### Audit Checklist

- [ ] No default `content_types` lists include `"image"`
- [ ] All retrieval paths filter out `content_type == "image"`
- [ ] Config defaults to `extract_images: false`
- [ ] Documentation updated to reflect images are disabled

## Quick Validation Steps

Run these commands locally to validate the image removal:

```bash
# 1. Verify config
grep -A 2 "non_text:" config.yaml | grep extract_images
# Should show: extract_images: false

# 2. Run smoke test
python scripts/smoke_no_images.py test_docs/

# 3. Run audit
rg -n '"image"|content_type.*image|image__vector_store|extract_images|get_images\(' backend frontend
# Review output for any issues (see "What to Look For" above)
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're running from the repo root:

```bash
cd /path/to/ArrowSystems
python scripts/smoke_no_images.py test_docs/
```

### Model Initialization Fails

The script will continue if model initialization fails, but queries may not work. Ensure:
- GPU/CPU environment is set up correctly
- Required model files are available
- Dependencies are installed

### Temporary Directory Issues

If you see permission errors with temp directories, use `--persist-dir` to specify a custom location:

```bash
python scripts/smoke_no_images.py test_docs/ --persist-dir /tmp/my_test_index
```

