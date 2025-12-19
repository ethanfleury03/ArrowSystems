# Step 5 Diagnostics and Safety Features Guide

## Overview

This guide describes the safety features and diagnostics added to Step 5 chunking to prevent infinite stalls and enable debugging.

## Features Implemented

### 1. Safe Stack Dump on Signal

**What it does:**
- Enables `faulthandler` for stack traces on crashes
- Registers `SIGUSR1` to dump all thread stacks without terminating the process
- Prints PID at startup for easy debugging

**Usage:**
```bash
# Find the process ID (printed at startup)
# Then send SIGUSR1 to dump stacks
kill -USR1 <pid>
```

**Output:**
- Stack traces printed to stderr/logs
- Process continues running (non-terminating)

### 2. Progress Log File

**What it does:**
- Writes structured progress log to file (default: `/workspace/ingest_work/chunk_progress.log`)
- Independent of tqdm/screen output (works even if terminal freezes)
- Flushes immediately for real-time monitoring

**Log Format:**
```
event=chunking_doc_start idx=0 total=100 file_name=doc.pdf source_gcs=gs://... document_id=123 page_label=1 section_number= raw_len=50000
event=chunking_doc_done idx=0 total=100 file_name=doc.pdf source_gcs=gs://... elapsed_s=2.5 chunks_count=45 nodes_emitted=42
event=chunking_heartbeat idx=49 total=100 nodes_so_far=2100
```

**Monitor progress:**
```bash
tail -f /workspace/ingest_work/chunk_progress.log
```

### 3. Per-Document Timeout Guards

**What it does:**
- Hard timeout per document (default: 90 seconds, configurable via `MAX_SECONDS_PER_DOC`)
- Automatically falls back to base splitter if timeout triggers
- Skips document only if fallback also fails

**Configuration:**
```bash
MAX_SECONDS_PER_DOC=120  # Increase timeout if needed
```

**Behavior:**
- On timeout: attempts fallback to `base_splitter.split_text()`
- If fallback succeeds: continues with fallback chunks
- If fallback fails: skips document and logs error

### 4. Chunk Explosion and Huge Doc Guards

**What it does:**
- Prevents pathological documents from generating millions of chunks
- Automatically falls back to base splitter when limits exceeded

**Configuration:**
```bash
MAX_DOC_CHARS_FOR_SMART_CHUNK=250000  # Bypass smart chunking for huge docs
MAX_CHUNKS_PER_DOC=5000                # Max chunks before fallback
MAX_STRUCTURED_LINES=200               # Max lines in structured block
```

**Behavior:**
- Documents > `MAX_DOC_CHARS_FOR_SMART_CHUNK`: use base splitter directly
- Chunk count > `MAX_CHUNKS_PER_DOC`: fallback to base splitter
- Structured block > `MAX_STRUCTURED_LINES`: fallback to base splitter for that block

### 5. Hot-Loop Performance Optimizations

**What it does:**
- Replaces expensive regex scans with short-circuiting counters
- Gates expensive checks (TOC, first-page) to only run when needed

**Optimizations:**
- `should_skip_node`: Short-circuiting alpha char counter (stops at threshold)
- TOC detection: Only on first page/chunk AND if keywords present in first 500 chars
- First-page check: Only when `page_label in {'1', 'i', 'I'}`
- `is_low_content_page`: Non-allocating word counter with early exit

### 6. Repro Mode for Isolating Failing Docs

**What it does:**
- Filter documents to specific source or range for quick reproduction
- Enables fast iteration when debugging specific documents

**Usage:**

**Option A: Filter by source GCS path**
```bash
CHUNK_ONLY_SOURCE_GCS=gs://bucket/path/to/doc.pdf python ingest.py
```

**Option B: Filter by document index range**
```bash
CHUNK_DOC_RANGE_START=100 CHUNK_DOC_RANGE_END=150 python ingest.py
```

**Note:** In repro mode, you can stop after Step 5 (no need to wait for embedding/index build).

## Environment Variables

All environment variables with defaults:

| Variable | Default | Description |
|----------|---------|-------------|
| `CHUNK_DEBUG` | `0` | Set to `1` to enable detailed diagnostic logging |
| `CHUNK_PROGRESS_LOG` | `/workspace/ingest_work/chunk_progress.log` | Path to progress log file |
| `CHUNK_HEARTBEAT_EVERY` | `50` | Heartbeat frequency (every N documents) |
| `MAX_SECONDS_PER_DOC` | `90` | Hard timeout per document (seconds) |
| `MAX_DOC_CHARS_FOR_SMART_CHUNK` | `250000` | Max doc size for smart chunking |
| `MAX_CHUNKS_PER_DOC` | `5000` | Max chunks before fallback |
| `MAX_STRUCTURED_LINES` | `200` | Max lines in structured block |
| `CHUNK_ONLY_SOURCE_GCS` | (none) | Filter to specific source GCS path |
| `CHUNK_DOC_RANGE_START` | (none) | Start index for doc range filter |
| `CHUNK_DOC_RANGE_END` | (none) | End index for doc range filter |

## Run Commands

### Normal Run
```bash
python ingest.py
```

### Debug Run (with detailed logging)
```bash
CHUNK_DEBUG=1 python ingest.py
```

### Monitor Progress Log
```bash
# In another terminal
tail -f /workspace/ingest_work/chunk_progress.log
```

### Dump Stacks (when process appears stuck)
```bash
# Find PID (printed at startup or via ps)
ps aux | grep ingest.py

# Send SIGUSR1 to dump stacks (non-terminating)
kill -USR1 <pid>

# Check logs/stderr for stack traces
```

### Repro Mode (isolate specific doc)
```bash
# Filter to specific document
CHUNK_ONLY_SOURCE_GCS=gs://bucket/path/to/problematic.pdf CHUNK_DEBUG=1 python ingest.py

# Or filter to doc range
CHUNK_DOC_RANGE_START=100 CHUNK_DOC_RANGE_END=150 CHUNK_DEBUG=1 python ingest.py
```

### Custom Timeout (for very large documents)
```bash
MAX_SECONDS_PER_DOC=300 python ingest.py  # 5 minute timeout
```

## Troubleshooting

### Process appears frozen but CPU is high

1. **Check progress log:**
   ```bash
   tail -f /workspace/ingest_work/chunk_progress.log
   ```
   - If log is updating: process is working, just slow
   - If log stopped: identify last document from log

2. **Dump stacks:**
   ```bash
   kill -USR1 <pid>
   ```
   - Look for function names in stack trace
   - Common culprits: `should_skip_node`, `is_table_of_contents`, `clean_text`

3. **Use repro mode:**
   - Identify problematic doc from progress log
   - Run with `CHUNK_ONLY_SOURCE_GCS` or `CHUNK_DOC_RANGE_*` to isolate

### Document timing out repeatedly

1. **Increase timeout:**
   ```bash
   MAX_SECONDS_PER_DOC=180 python ingest.py
   ```

2. **Check if document is huge:**
   - Look for `chunking_doc_huge` events in progress log
   - Consider lowering `MAX_DOC_CHARS_FOR_SMART_CHUNK` to force base splitter

3. **Check for chunk explosion:**
   - Look for `chunking_doc_chunk_explosion` events
   - Consider lowering `MAX_CHUNKS_PER_DOC` to trigger fallback earlier

### Progress log not updating

1. **Check file permissions:**
   ```bash
   ls -la /workspace/ingest_work/chunk_progress.log
   ```

2. **Check disk space:**
   ```bash
   df -h /workspace/ingest_work
   ```

3. **Override log path:**
   ```bash
   CHUNK_PROGRESS_LOG=/tmp/chunk_progress.log python ingest.py
   ```

## Acceptance Criteria

✅ **Stack dumps work:**
- `kill -USR1 <pid>` prints stack traces without terminating process

✅ **Progress log is observable:**
- `tail -f` shows continuous updates independent of tqdm
- Last entry shows exact document being processed

✅ **Timeouts prevent hangs:**
- Documents exceeding timeout trigger fallback or skip
- Ingestion continues after timeout

✅ **Guards prevent explosions:**
- No document generates >5000 chunks
- Huge documents use base splitter automatically

✅ **Repro mode isolates issues:**
- Can target specific document or range
- Fast iteration for debugging

## Notes

- All features are opt-in via environment variables (safe defaults)
- No changes to database schema or serving behavior
- Only affects ingestion CLI path and chunking internals
- No logging of chunk text content (metadata-only)

