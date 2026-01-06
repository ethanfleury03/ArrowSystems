# Images Disabled - Integration Audit & Verification

## Overview

This document tracks the integration work to ensure the system is bulletproof when images are disabled (`non_text.extract_images: false` in `config.yaml`).

## Status: ✅ COMPLETE

All integration gaps have been closed. The system now has defense-in-depth protection against image nodes.

## Changes Made

### STEP 1: Fixed Single-File Ingestion ✅

**File:** `backend/utils/single_file_ingestion.py`

**Changes:**
1. ✅ Config is now passed to `NonTextExtractor`: `extractor = NonTextExtractor(config=config)`
2. ✅ Fixed method name: Changed `extract_captions_from_pdf()` to `extract_figure_captions()`
3. ✅ Image extraction is skipped when disabled: Checks `extract_images_enabled` and skips `extract_images_from_pdf()` call entirely
4. ✅ Logging is accurate: Never logs "Extracted X images" when disabled

**Acceptance Criteria:**
- ✅ Single-file ingestion does not throw `AttributeError`
- ✅ With `extract_images=false`, it does not attempt image extraction and does not write PNGs
- ✅ Behavior is consistent with bulk ingestion path

### STEP 2: Added Defense-in-Depth Filtering in Orchestrator ✅

**File:** `backend/orchestrator.py`

**Changes:**
1. ✅ Added `_is_image_node()` helper function (lines 40-68)
   - Safely handles `NodeWithScore` and plain nodes
   - Never crashes on missing metadata
   - Returns `True` only when `metadata.content_type == "image"`

2. ✅ Added `_filter_image_nodes()` helper function (lines 71-100)
   - Filters out image nodes while preserving order
   - Logs only when nodes are actually removed (avoids log spam)
   - Never raises exceptions

3. ✅ Applied filtering in all retrieval paths:
   - `bm25_search()`: Filters after retrieving results (lines 1024-1030)
   - `dense_search()`: Filters after inactive document filtering (line 1082)
   - `hybrid_search()`: Filters after scoring/sorting, before reranking (line 1678)

**Acceptance Criteria:**
- ✅ Orchestrator never returns image nodes (dense, bm25, hybrid)
- ✅ No crashes if metadata is missing
- ✅ Works with different `NodeWithScore` wrapper versions

### STEP 3: Updated Verification Script ✅

**File:** `scripts/smoke_no_images.py`

**Changes:**
1. ✅ Extended smoke test with orchestrator verification (STEP 4)
   - Tests orchestrator retrieval path directly
   - Validates that orchestrator never returns image nodes
   - Tests filtering helpers with synthetic image nodes

2. ✅ Added single-file ingestion test (STEP 5)
   - Tests that single-file ingestion doesn't crash
   - Validates no image artifacts are created

**Acceptance Criteria:**
- ✅ Single command proves all fixes: `python scripts/smoke_no_images.py test_docs/`
- ✅ Failures are actionable with clear error messages

### STEP 4: Cleaned Up Misleading References ✅

**Review Results:**
- ✅ `backend/ingest.py` line 616: Defensive check in `is_structured_content()` - OK (handles legacy data)
- ✅ `backend/ingest.py` line 3282: Already filtering out image nodes - OK
- ✅ No misleading references found that need cleanup

## Verification

### Test Command

```bash
python scripts/smoke_no_images.py test_docs/
```

### Expected Output

The test should pass all checks:
1. ✅ Config check: `extract_images` is disabled
2. ✅ No extracted image artifacts found
3. ✅ Docstore inspection: 0 image nodes
4. ✅ Query results: 0 image nodes
5. ✅ Orchestrator retrieval: 0 image nodes (defense-in-depth)
6. ✅ Single-file ingestion: No crashes, no image artifacts

### PASS/FAIL Checklist

- [x] **Config:** `extract_images` is `false` in `config.yaml`
- [x] **Bulk Ingestion:** Produces 0 image artifacts, 0 image nodes in docstore
- [x] **Single-File Ingestion:** Does not crash, respects config, does not write PNGs
- [x] **Orchestrator Dense Search:** Filters out image nodes (defense-in-depth)
- [x] **Orchestrator BM25 Search:** Filters out image nodes (defense-in-depth)
- [x] **Orchestrator Hybrid Search:** Filters out image nodes (defense-in-depth)
- [x] **Smoke Test:** All checks pass
- [x] **Legacy Index Protection:** Even if old indexes contain image nodes, they are filtered out

## Edge Cases Validated

1. ✅ **Missing metadata:** Filtering helpers handle nodes with missing metadata gracefully
2. ✅ **Different NodeWithScore versions:** Works with various LlamaIndex versions
3. ✅ **Legacy indexes:** Old indexes with image nodes are filtered out at retrieval time
4. ✅ **Config changes:** Single-file ingestion respects config changes immediately
5. ✅ **Method name fix:** `extract_figure_captions()` is used (not the non-existent `extract_captions_from_pdf()`)

## Files Changed

1. `backend/utils/single_file_ingestion.py` - Fixed config passing, method name, and image extraction logic
2. `backend/orchestrator.py` - Added defense-in-depth filtering helpers and applied to all retrieval paths
3. `scripts/smoke_no_images.py` - Extended with orchestrator and single-file ingestion tests
4. `docs/images-disabled-integration-audit.md` - This documentation file

## Notes

- Images are **permanently disabled** at the extraction level (`extract_images_from_pdf()` always returns empty)
- Defense-in-depth filtering ensures legacy indexes cannot return image nodes
- All changes are minimal, safe, and defensive (never crash on unexpected input)

