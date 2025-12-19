# SmartChunkSplitter Performance Fix Summary

## Changes Made

### 1. Fixed Logging Overhead in Hot Loops

All f-string logging calls in `SmartChunkSplitter.get_nodes_from_documents` have been replaced with parameterized logging to avoid string formatting overhead in hot loops.

**Changed logging callsites:**

1. **Line 913**: `logger.warning(f"Error cleaning text for {doc_name}: {e}")` 
   → `logger.warning("Error cleaning text for %s: %s", doc_name, e)`

2. **Line 920**: `logger.debug(f"Skipping low-content page: {doc_name}")`
   → `logger.debug("Skipping low-content page: %s", doc_name)`

3. **Line 924**: `logger.debug(f"Error checking low-content for {doc_name}: {e}")`
   → `logger.debug("Error checking low-content for %s: %s", doc_name, e)`

4. **Line 930**: `logger.warning(f"Using simple chunker for huge document: {doc_name} (len={len(text)})")`
   → `logger.warning("Using simple chunker for huge document: %s (len=%d)", doc_name, len(text))`

5. **Line 941**: `logger.error(f"Error splitting text for {doc_name}: {e}")`
   → `logger.error("Error splitting text for %s: %s", doc_name, e)`

6. **Line 959**: `logger.debug(f"Skipping chunk {chunk_idx} from {doc_name}: {skip_reason}")`
   → `logger.debug("Skipping chunk %s from %s: %s", chunk_idx, doc_name, skip_reason)`

7. **Line 993**: `logger.error(f"Error creating node for {doc_name}, chunk {chunk_idx}: {e}")`
   → `logger.error("Error creating node for %s, chunk %s: %s", doc_name, chunk_idx, e)`

8. **Line 1003**: `logger.error(f"Error processing document {doc.metadata.get('file_name', 'unknown')}: {e}")`
   → `logger.error("Error processing document %s: %s", doc_meta.get('file_name', 'unknown'), e)`

### 2. Added Progress Log File

- Progress log file: `/workspace/ingest_work/chunk_progress.log` (overrideable via `CHUNK_PROGRESS_LOG` env var)
- Log entries are flushed immediately for `tail -f` monitoring
- Log format:
  - `START idx=<i> src=<source> len=<raw_text_len>` - Document processing started
  - `DONE idx=<i> secs=<elapsed> chunks=<n> nodes=<n_emitted>` - Document completed
  - `SKIP idx=<i> reason=<reason>` - Document skipped
  - `ERROR idx=<i> secs=<elapsed> error=<error>` - Error occurred
  - `HUGE idx=<i> len=<len> using_simple_chunker=1` - Huge document using fallback
  - `HEARTBEAT idx=<i>` - Periodic heartbeat

### 3. Added Huge Text Fallback

- Environment variable: `MAX_DOC_CHARS` (default: 250000)
- Documents exceeding the limit use a simple deterministic chunker instead of smart chunking
- Simple chunker uses:
  - Chunk size: `max(512, self.chunk_size)`
  - Overlap: `max(128, self.chunk_overlap)`
- Single WARNING log when fallback is used

### 4. Added Heartbeat

- Environment variable: `CHUNK_HEARTBEAT_EVERY` (default: 25)
- Writes `HEARTBEAT idx=<i>` to progress log every N documents
- Ensures progress visibility even if processing stalls

### 5. Safe Metadata Access

- All `doc.metadata.get(...)` calls replaced with `doc_meta = doc.metadata or {}` pattern
- Prevents crashes when metadata is None

## Monitoring Commands

### Monitor Progress Log
```bash
tail -f /workspace/ingest_work/chunk_progress.log
```

### Monitor Process Status
```bash
# Find the process ID first
ps aux | grep ingest.py

# Then monitor it
ps -p <pid> -o pid,etime,pcpu,pmem,stat
```

### Alternative: Continuous Monitoring
```bash
watch -n 5 'ps -p <pid> -o pid,etime,pcpu,pmem,stat'
```

## Environment Variables

- `CHUNK_PROGRESS_LOG`: Override progress log path (default: `/workspace/ingest_work/chunk_progress.log`)
- `MAX_DOC_CHARS`: Maximum document size before using simple chunker (default: 250000)
- `CHUNK_HEARTBEAT_EVERY`: Heartbeat frequency in documents (default: 25)

## Expected Improvements

1. **Performance**: Eliminated f-string formatting overhead in hot loops (thousands/millions of iterations)
2. **Visibility**: Progress log provides deterministic progress tracking even if tqdm freezes
3. **Diagnostics**: Can identify exact document causing slowdown via last START line
4. **Resilience**: Huge documents automatically use fast fallback chunker

## Testing

1. Run ingestion: `python ingest.py`
2. In another terminal, monitor progress: `tail -f /workspace/ingest_work/chunk_progress.log`
3. Verify START/DONE lines incrementing
4. Check for HUGE entries if any documents exceed size limit
5. Verify HEARTBEAT lines appear every 25 documents

