# Step 5 Infinite Loop Fix

## Problem

Step 5 chunking was hanging indefinitely on certain documents (e.g., "DuraBolt Installation Guide"). The process showed:
- High CPU on one thread
- No progress
- Always stopped at the same document index
- No index artifacts written (stuck in chunking)

## Root Cause

**Bug in `_preserve_structured_chunks()`:**
- When a single line exceeds `chunk_size` (e.g., 512 chars) AND `current_chunk` is empty
- The code enters the "chunk is full" branch but doesn't increment `i` because it only increments inside `if current_chunk:`
- This causes an infinite loop pegging 1 CPU core forever

**Why single long lines exist:**
- Preprocessing (`normalize_whitespace`, `normalize_technical_content`) was collapsing newlines
- Whole pages (e.g., TOC pages) became single long lines (logs showed `raw_len=5780` → one "line" after newline removal)

## Fixes Implemented

### A) Fixed `_preserve_structured_chunks()` Infinite Loop

**Changes:**
1. **Explicit handling for long single lines:**
   - Added check: `if line_size > self.chunk_size:`
   - Flushes `current_chunk` if non-empty
   - Splits the long line using `base_splitter.split_text(line)` or fallback simple slicer
   - Appends chunks and increments `i` → **guarantees forward progress**

2. **Safety guard for infinite loop detection:**
   - Tracks `prev_i` each loop iteration
   - If `i` doesn't change, logs warning and forces `i += 1` to prevent hard hangs
   - Last-resort guard (should never trigger after the long-line fix)

3. **Fallback for empty current_chunk edge case:**
   - Added `else` branch when `current_chunk` is empty but we're in the "chunk full" path
   - Adds line as single chunk and forces progress

**Code location:** `backend/ingest.py`, `SmartChunkSplitter._preserve_structured_chunks()`, lines ~872-920

### B) Preserved Newlines in Preprocessing

**Changes:**
1. **`normalize_whitespace()`:**
   - Changed from processing entire text to processing line-by-line
   - Preserves `\n` characters between lines
   - Only normalizes spaces/tabs within each line
   - Collapses excessive newlines (`\n{3,}` → `\n\n`) but keeps line structure

2. **`normalize_technical_content()`:**
   - Changed from processing entire text to processing line-by-line
   - Preserves `\n` characters between lines
   - Only normalizes spacing/punctuation within each line
   - Maintains line structure for chunking

**Code location:** `backend/ingest.py`, `TextPreprocessor.normalize_whitespace()` and `normalize_technical_content()`

### C) Added Per-Doc Stage Timing Logs

**Changes:**
- Added timing around key stages in `get_nodes_from_documents()`:
  - `clean_text`
  - `is_low_content_page`
  - `split_text`
  - `chunk_loop` (loop over chunks + `should_skip_node`)

**Log format (when `CHUNK_DEBUG=1`):**
```json
{
  "event": "chunking_doc_done",
  "document_id": 123,
  "source_gcs": "gs://bucket/doc.pdf",
  "file_name": "doc.pdf",
  "page_label": "1",
  "section_number": "",
  "raw_len": 5780,
  "elapsed_s": 2.5,
  "chunks": 45,
  "nodes_emitted": 42,
  "idx": 0,
  "total": 100,
  "stage_clean_text_s": 0.123,
  "stage_is_low_content_page_s": 0.001,
  "stage_split_text_s": 0.456,
  "stage_chunk_loop_s": 1.234
}
```

**Usage:**
```bash
CHUNK_DEBUG=1 python ingest.py
```

**Code location:** `backend/ingest.py`, `SmartChunkSplitter.get_nodes_from_documents()`, lines ~1130-1230

### D) Added Repro Test

**Self-test function:**
- `_test_chunker_infinite_loop_fix()` in `backend/ingest.py`
- Tests 6000+ char string with no newlines
- Verifies:
  - Returns chunks quickly (< 1 second)
  - Returns multiple chunks
  - All chunks are non-empty
  - No infinite loop

**Run test:**
```bash
RUN_CHUNK_SELFTEST=1 python ingest.py
```

**Code location:** `backend/ingest.py`, lines ~3899-3970

## Testing

### Manual Test
```bash
# Run self-test
RUN_CHUNK_SELFTEST=1 python ingest.py

# Expected output:
# ✅ Test passed!
#    - Input: 6000 chars (no newlines)
#    - Output: N chunks
#    - Total output chars: >= 6000
#    - Elapsed: < 1.0s
```

### Integration Test
```bash
# Run full ingestion with debug logging
CHUNK_DEBUG=1 python ingest.py

# Monitor for:
# - No hangs at Step 5
# - Stage timings in logs
# - Progress continues past previously problematic documents
```

## Acceptance Criteria

✅ **Infinite loop eliminated:**
- Documents with long single lines are chunked safely
- `_preserve_structured_chunks()` always makes forward progress
- Safety guard prevents hard hangs

✅ **Newlines preserved:**
- Preprocessing maintains line structure
- Whole pages don't become single long lines
- Chunking has proper structure to work with

✅ **Diagnostics available:**
- `CHUNK_DEBUG=1` shows stage timings
- Can identify which stage is slow for problematic docs
- No document content logged (metadata-only)

✅ **Test passes:**
- Self-test completes in < 1 second
- Returns multiple chunks for 6000 char input
- No infinite loop detected

✅ **Behavior stable:**
- No changes to database schema
- No changes to ingestion semantics
- Existing structured-preservation behavior maintained

## Files Modified

- `backend/ingest.py`:
  - `SmartChunkSplitter._preserve_structured_chunks()` - Fixed infinite loop
  - `TextPreprocessor.normalize_whitespace()` - Preserve newlines
  - `TextPreprocessor.normalize_technical_content()` - Preserve newlines
  - `SmartChunkSplitter.get_nodes_from_documents()` - Added stage timing
  - Added `_test_chunker_infinite_loop_fix()` - Self-test function

## Related Issues

- Step 5 hanging on specific documents
- High CPU usage during chunking
- Preprocessing collapsing newlines
- Single long lines causing chunking failures

