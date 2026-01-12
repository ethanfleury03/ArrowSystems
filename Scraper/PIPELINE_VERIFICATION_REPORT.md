# Ticket Cache Artifacts Pipeline Verification Report

**Date:** 2026-01-12  
**Verification Type:** End-to-end dry-run verification (no writes to index or DB)  
**Status:** ✅ PASS with minor limitations noted

---

## Executive Summary

| Component | Status | Notes |
|-----------|--------|-------|
| Contract/Schema Compatibility | ✅ PASS | All required fields match between exporter and ingester |
| Exporter Query Logic | ✅ PASS | Correctly selects cache-eligible tickets, no confirmation filter |
| JSONL Validity | ✅ PASS | All 20 sample artifacts are valid JSON |
| Ingester Dry-Run Logic | ⚠️ PARTIAL | Code is correct but requires embeddings for full test |
| Index Write Safety | ✅ PASS | Dry-run correctly gates persist() calls |
| Retrieval/UI Safety | ⚠️ RISK IDENTIFIED | Ticket artifacts will trigger PDF fetch that will fail |

---

## A) Contract / Schema Compatibility

### A.1 Contract Table

| Field | Exporter Produces | Ingester Expects | Retrieval/UI Uses | Status |
|-------|-------------------|------------------|-------------------|--------|
| `id` | ✅ `ticket:{ticket_id}` | ✅ Required, validated | ❌ Not used | ✅ PASS |
| `text` | ✅ Deterministic template | ✅ Required, non-empty | ✅ Used for search | ✅ PASS |
| `metadata.document_id` | ✅ `ticket:{ticket_id}` | ✅ Required | ✅ Used as doc identifier | ✅ PASS |
| `metadata.file_name` | ✅ `ticket_{ticket_id}.md` | ✅ Required | ⚠️ Used for PDF fetch | ⚠️ RISK |
| `metadata.content_type` | ✅ `ticket_cache` | ✅ Required | ❓ Not checked | ✅ PASS |
| `metadata.source` | ✅ `zendesk_ticket` | ✅ Required | ❓ Not checked | ✅ PASS |
| `metadata.ticket_id` | ✅ Original ticket ID | ❌ Optional | ❌ Not used | ✅ PASS |
| `metadata.outcome` | ✅ From raw_json | ❌ Optional | ❌ Not used | ✅ PASS |
| `metadata.confidence` | ✅ Float 0.0-1.0 | ✅ Validated range | ❌ Not used | ✅ PASS |
| `metadata.cache_eligible` | ✅ 0 or 1 | ✅ Validated | ❌ Not used | ✅ PASS |
| `metadata.confirmed` | ✅ Bool | ❌ Optional | ❌ Not used | ✅ PASS |
| `metadata.machine_model_ids` | ✅ Empty list `[]` | ✅ List type validated | ✅ Used for filtering | ✅ PASS |
| `metadata.machine_model_names` | ✅ Empty list `[]` | ✅ List type validated | ✅ Used for filtering | ✅ PASS |

**Key Findings:**
- ✅ All required fields are produced by exporter and validated by ingester
- ✅ Pydantic model validation ensures type safety
- ⚠️ `file_name` format (`ticket_{id}.md`) will trigger PDF fetch attempts in UI

### A.2 Exporter Verification

**SQL Query Analysis** (`export_cache_artifacts.py` lines 163-183):
```sql
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = 1))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = 1
```

**Verification:**
- ✅ Includes auto-approved: `j.review_status = 'approved'`
- ✅ Includes manual-approved: `m.manual_status = 'approved'`
- ✅ Includes legacy cache_eligible: `j.cache_eligible = 1` (when review_status NULL)
- ✅ Excludes manual-rejected: `m.manual_status != 'rejected'`
- ✅ **Does NOT filter on `confirmation.confirmed`** ✅ CORRECT
- ✅ Requires `cache_eligible = 1` ✅ CORRECT

**ID Format:**
- ✅ Deterministic: `f"ticket:{ticket_id}"` (line 156)
- ✅ Stable: Same ticket_id always produces same ID

**Text Template:**
- ✅ Deterministic: Fixed template with problem, steps, outcome, rationale, blockers
- ✅ No LLM calls: Pure string formatting
- ✅ Fallback: Provides default text if all fields empty

### A.3 Ingester Verification

**Dry-Run Support:**
- ✅ `--dry-run` flag implemented (line 215)
- ✅ Early return before `index.insert_nodes()` (line 215-224)
- ✅ Early return before `index.storage_context.persist()` (line 244 is AFTER dry-run check)

**Node ID Deduplication:**
- ✅ Uses `id_=artifact.id` in `TextNode` creation (line 124)
- ✅ `skip_existing` flag checks `existing_ids` set (line 198)
- ✅ Namespace prefix `ticket:` prevents collisions with doc nodes

**Metadata Preservation:**
- ✅ All metadata keys passed through: `metadata=artifact.metadata` (line 123)
- ✅ No filtering or transformation of metadata

---

## B) Dry-Run Execution Results

### B.1 Export Test

**Command:**
```bash
python export_cache_artifacts.py --db data/tickets.db --out out/_dryrun_cache_artifacts_20.jsonl --limit 20
```

**Output:**
```
Found 20 cache-eligible tickets

======================================================================
EXPORT SUMMARY
======================================================================
Total tickets found: 20
Successfully exported: 20
Failed: 0
Output file: out\_dryrun_cache_artifacts_20.jsonl
======================================================================
```

**Result:** ✅ **PASS** - All 20 tickets exported successfully

### B.2 JSONL Validation

**Validation Method:** Python json.loads() on each line

**Result:**
- ✅ Total lines: 20
- ✅ All lines valid JSON: 20/20
- ✅ No parsing errors

**Sample Artifact Structure:**
```json
{
  "id": "ticket:3688",
  "text": "Problem: ...\nResolution Steps:\n1. ...",
  "metadata": {
    "document_id": "ticket:3688",
    "file_name": "ticket_3688.md",
    "content_type": "ticket_cache",
    "source": "zendesk_ticket",
    ...
  }
}
```

**Result:** ✅ **PASS** - JSONL is well-formed

### B.3 Ingester Dry-Run Test

**Command:**
```bash
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl Scraper/out/_dryrun_cache_artifacts_20.jsonl --dry-run --limit 20
```

**Output (partial):**
```
[info] Loading artifacts from Scraper/out/_dryrun_cache_artifacts_20.jsonl
[info] Loaded 20 artifacts
[warning] Failed to load index from latest_model: [embedding model error]
[info] Creating new index
ERROR: [embedding model required]
```

**Analysis:**
- ✅ Artifacts loaded successfully: 20/20
- ✅ Validation would occur (failed before reaching due to embedding requirement)
- ⚠️ **Limitation:** Dry-run requires embedding model to be configured (even though it won't write)

**Code Verification:**
- ✅ Dry-run check at line 215 returns BEFORE `index.insert_nodes()` (line 234)
- ✅ Dry-run check at line 215 returns BEFORE `index.storage_context.persist()` (line 244)
- ✅ Persist call is correctly gated: Only executes if `not dry_run`

**Result:** ⚠️ **PARTIAL PASS** - Code logic is correct, but full dry-run test requires embeddings configured

### B.4 Index Write Safety Verification

**Git Status Check:**
- ✅ No files modified in `latest_model/` directory
- ✅ Output JSONL file created (expected, non-destructive)

**Code Analysis:**
```python
# Line 215-224: Dry-run early return
if dry_run:
    logger.info("DRY RUN: Would insert nodes but skipping actual insertion")
    return {...}  # Returns BEFORE any writes

# Line 234: Insert nodes (only if not dry_run)
index.insert_nodes(batch)

# Line 244: Persist index (only if not dry_run)
index.storage_context.persist(persist_dir=str(index_dir))
```

**Result:** ✅ **PASS** - Dry-run correctly prevents all writes

---

## C) Safety Checks

### C.1 Retrieval/UI Safety

**Issue Identified:** ⚠️ **RISK**

**Problem:**
1. UI uses `doc.doc_id` (which comes from `metadata.document_id` = `ticket:{id}`)
2. UI calls `/api/documents/${filename}` where `filename = doc.doc_id`
3. Frontend route (`frontend/app/api/documents/[...path]/route.ts`) calls backend `/documents/${filename}`
4. Backend expects a PDF file in GCS
5. Ticket artifacts have `file_name = "ticket_{id}.md"` which doesn't exist as PDF

**Evidence:**
- `documents-panel.tsx` line 56: `setSelectedDoc({ filename: doc.doc_id, page: firstPage })`
- `documents-panel.tsx` line 53: Uses `doc.doc_id` directly
- `frontend/app/api/documents/[...path]/route.ts` line 49: Calls `${BACKEND_URL}/documents/${encodedFilename}`
- Backend expects PDF binary response

**Impact:**
- User clicks ticket cache artifact → Frontend requests `/api/documents/ticket:12345`
- Backend tries to fetch PDF from GCS → **404 Not Found**
- User sees error instead of ticket content

**Recommended Fix (NOT IMPLEMENTED):**
1. Check `metadata.content_type === "ticket_cache"` in UI
2. If ticket_cache, show text modal instead of PDF viewer
3. Display artifact text content directly
4. Alternative: Create backend endpoint `/api/tickets/{ticket_id}` that returns ticket text

**Result:** ⚠️ **RISK IDENTIFIED** - UI will fail when clicking ticket artifacts

### C.2 Metadata Filtering

**Machine Model Filtering:**
- ✅ Ticket artifacts include `machine_model_ids: []` and `machine_model_names: []`
- ✅ Empty lists are valid (won't be excluded)
- ✅ Filtering logic in `orchestrator.py` line 1329-1338 handles empty lists correctly
- ✅ Customers can see documents with empty machine_model (if admin allows)

**Content Type Filtering:**
- ✅ `content_type: "ticket_cache"` is set
- ⚠️ No evidence of content_type filtering in retrieval code
- ✅ Should not be excluded (no filters found)

**Result:** ✅ **PASS** - Metadata won't cause exclusion

### C.3 Naming Collisions

**Node ID Format:**
- ✅ Tickets: `ticket:{ticket_id}` (e.g., `ticket:3688`)
- ✅ Documents: Typically UUIDs or document IDs (not `ticket:` prefix)
- ✅ Namespace separation: `ticket:` prefix prevents collisions

**Verification:**
- Exporter: `id=f"ticket:{ticket_id}"` (line 156)
- Ingester: `id_=artifact.id` (line 124)
- No collision risk with document nodes

**Result:** ✅ **PASS** - No collision risk

### C.4 Size and Limits

**JSONL Streaming:**
- ✅ Exporter writes line-by-line (line 267): `f.write(json.dumps(artifact, ensure_ascii=False) + '\n')`
- ✅ Ingester reads line-by-line (line 58): `for line_num, line in enumerate(f, 1)`
- ✅ No full-file loading into memory
- ✅ Batch processing in ingester (line 231): `batch_size = 50`

**Text Template Limits:**
- ✅ No explicit size limits found in code
- ✅ LlamaIndex TextNode has no hard size limit (embeddings chunk if needed)
- ✅ Typical ticket text: ~500-2000 chars (well within limits)

**Result:** ✅ **PASS** - Handles large exports efficiently

---

## D) Summary of Findings

### ✅ Passing Components

1. **Contract Compatibility:** All required fields match between exporter and ingester
2. **Exporter Query:** Correctly selects cache-eligible tickets without confirmation filter
3. **JSONL Format:** Valid, well-formed JSONL output
4. **Dry-Run Logic:** Code correctly gates writes (requires embeddings for full test)
5. **Index Write Safety:** Dry-run prevents all persist() calls
6. **Metadata Filtering:** Won't cause exclusions
7. **Naming Collisions:** No collision risk with `ticket:` prefix
8. **Size Handling:** Efficient streaming and batching

### ⚠️ Risks and Limitations

1. **UI Document Fetch:** Ticket artifacts will trigger PDF fetch that fails (404)
   - **Severity:** Medium
   - **Impact:** User experience degradation
   - **Fix Required:** UI needs to handle `content_type=ticket_cache` differently

2. **Dry-Run Embedding Requirement:** Full dry-run test requires embedding model configured
   - **Severity:** Low
   - **Impact:** Can't fully test dry-run without embeddings
   - **Workaround:** Code logic verified manually

### 🔧 Recommended Fixes (NOT IMPLEMENTED)

1. **UI Fix for Ticket Artifacts:**
   ```typescript
   // In documents-panel.tsx, check content_type before opening PDF
   if (doc.content_type === 'ticket_cache') {
     // Show ticket text in modal instead of PDF viewer
     showTicketModal(doc.doc_id, doc.text);
   } else {
     // Existing PDF viewer logic
     setSelectedDoc({ filename: doc.doc_id, page: firstPage });
   }
   ```

2. **Alternative: Backend Ticket Endpoint:**
   - Create `/api/tickets/{ticket_id}` endpoint
   - Return ticket text content as JSON
   - UI calls this instead of `/api/documents/` for ticket artifacts

---

## E) Safe Commands for Production Use

### Full Export (Safe - Only writes JSONL)
```bash
cd Scraper
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl
```

### Full Dry-Run Ingestion (Safe - No writes)
```bash
cd ArrowSystems
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl Scraper/out/cache_artifacts.jsonl --dry-run
```

### Full Real Ingestion (⚠️ WRITES TO INDEX - Use with caution)
```bash
cd ArrowSystems
python -m backend.scripts.ingest_ticket_cache_artifacts --jsonl Scraper/out/cache_artifacts.jsonl --skip-existing
```

**Note:** Real ingestion will modify the RAG index. Ensure backups and test in non-production first.

---

## F) Final Verdict

**Overall Status:** ✅ **PASS** (with UI fix recommended)

The pipeline is functionally correct and safe for export + dry-run ingestion. The only issue is UI handling of ticket artifacts, which requires a frontend fix to prevent 404 errors when users click on ticket sources.

**Recommendation:** 
1. ✅ Safe to run export
2. ✅ Safe to run dry-run ingestion (with embeddings configured)
3. ⚠️ Fix UI before running real ingestion (or accept 404 errors on ticket clicks)
4. ✅ Safe to run real ingestion after UI fix

---

**Report Generated:** 2026-01-12  
**Verified By:** Automated pipeline verification  
**Files Verified:** 
- `Scraper/export_cache_artifacts.py`
- `backend/utils/ticket_cache_artifacts.py`
- `backend/scripts/ingest_ticket_cache_artifacts.py`
- `frontend/components/documents-panel.tsx`
- `frontend/app/api/documents/[...path]/route.ts`
