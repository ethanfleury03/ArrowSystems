# Testing New Index and Verifying Metadata

## Quick Test Commands

### 1. Test Index via API Endpoints (curl/HTTP)

**If your backend is running locally:**
```bash
# Base URL (adjust port if needed)
BASE_URL="http://localhost:8080"
```

**If your backend is on Cloud Run:**
```bash
# Replace with your actual Cloud Run URL
BASE_URL="https://your-service-url.run.app"
```

#### A) Check RAG Status
```bash
curl -X GET "${BASE_URL}/rag/status" | jq
```

**Expected response:**
```json
{
  "status": "ready",
  "rag_enabled": true,
  "initialized": true,
  "storage_dir": "/path/to/index",
  "last_error": null
}
```

#### B) Run Self-Test (Verifies Index is Queryable)
```bash
curl -X GET "${BASE_URL}/rag/self-test" | jq
```

**Expected response:**
```json
{
  "status": "ok",
  "rag_enabled": true,
  "test_query": "test",
  "num_results": 1,
  "storage_dir": "/path/to/index",
  "last_error": null
}
```

#### C) Validate Index Files
```bash
curl -X GET "${BASE_URL}/rag/validate-index" | jq
```

**Expected response:**
```json
{
  "storage_path": "/path/to/index",
  "directory_exists": true,
  "files_validated": {
    "docstore.json": {"exists": true, "size_bytes": 1234567, "valid_json": true},
    "default__vector_store.json": {"exists": true, "size_bytes": 2345678, "valid_json": true},
    "index_store.json": {"exists": true, "size_bytes": 345678, "valid_json": true}
  },
  "all_valid": true,
  "missing_files": [],
  "corrupted_files": []
}
```

#### D) Test Actual Query
```bash
curl -X POST "${BASE_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is DuraFlex?",
    "top_k": 5
  }' | jq
```

**Expected:** Returns query results with sources, answers, and metadata.

---

### 2. Verify Index Files in GCS (gsutil commands)

**Check if index files exist in GCS:**
```bash
# List files in latest_model
gsutil ls -lh gs://arrow-rag-support-prod-rag/latest_model/

# Check specific files
gsutil ls gs://arrow-rag-support-prod-rag/latest_model/*.json

# Get file sizes
gsutil du -h gs://arrow-rag-support-prod-rag/latest_model/

# Download and inspect docstore (contains metadata)
gsutil cp gs://arrow-rag-support-prod-rag/latest_model/docstore.json /tmp/docstore.json
python3 -c "import json; data=json.load(open('/tmp/docstore.json')); print(f'Total nodes: {len(data[\"docstore/data\"])}')"
```

**Verify backup was created:**
```bash
# List backups
gsutil ls gs://arrow-rag-support-prod-rag/old_model/

# Check latest backup
gsutil ls -lh gs://arrow-rag-support-prod-rag/old_model/2025-12-19T04-06-17Z/
```

---

### 3. Verify Metadata Was Applied Correctly

#### A) Check Metadata in Index (Python script)

Create a test script `verify_index_metadata.py`:

```python
#!/usr/bin/env python3
"""
Verify metadata in the index.
Checks that document_id, machine_model_ids, machine_model_names are present.
"""

import json
import sys
from pathlib import Path
from collections import defaultdict

# Path to index (adjust if needed)
INDEX_DIR = Path("/workspace/ingest_work/index_artifact")
# Or if downloaded from GCS:
# INDEX_DIR = Path("/tmp/index_artifact")

DOCSTORE_FILE = INDEX_DIR / "docstore.json"

if not DOCSTORE_FILE.exists():
    print(f"❌ Index not found at {DOCSTORE_FILE}")
    print("   Download from GCS first:")
    print("   gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/ /tmp/index_artifact/")
    sys.exit(1)

print("=" * 70)
print("Index Metadata Verification")
print("=" * 70)
print()

# Load docstore
with open(DOCSTORE_FILE, 'r') as f:
    docstore = json.load(f)

nodes = docstore.get("docstore/data", {})
print(f"📊 Total nodes in index: {len(nodes)}")
print()

# Check metadata fields
metadata_stats = {
    "has_document_id": 0,
    "has_machine_model_ids": 0,
    "has_machine_model_names": 0,
    "has_source_gcs": 0,
    "missing_document_id": 0,
    "missing_machine_models": 0,
}

document_ids = set()
machine_models_by_doc = defaultdict(set)
source_gcs_paths = set()

for node_id, node_data in nodes.items():
    metadata = node_data.get("metadata", {})
    
    # Check document_id
    doc_id = metadata.get("document_id")
    if doc_id:
        metadata_stats["has_document_id"] += 1
        document_ids.add(doc_id)
    else:
        metadata_stats["missing_document_id"] += 1
    
    # Check machine_model_ids
    mm_ids = metadata.get("machine_model_ids")
    if mm_ids:
        metadata_stats["has_machine_model_ids"] += 1
        if doc_id:
            machine_models_by_doc[doc_id].update(mm_ids if isinstance(mm_ids, list) else [mm_ids])
    
    # Check machine_model_names
    mm_names = metadata.get("machine_model_names")
    if mm_names:
        metadata_stats["has_machine_model_names"] += 1
    
    # Check source_gcs
    source_gcs = metadata.get("source_gcs") or metadata.get("gcs_path")
    if source_gcs:
        metadata_stats["has_source_gcs"] += 1
        source_gcs_paths.add(source_gcs)

print("📋 Metadata Coverage:")
print(f"   Nodes with document_id: {metadata_stats['has_document_id']} ({metadata_stats['has_document_id']/len(nodes)*100:.1f}%)")
print(f"   Nodes with machine_model_ids: {metadata_stats['has_machine_model_ids']} ({metadata_stats['has_machine_model_ids']/len(nodes)*100:.1f}%)")
print(f"   Nodes with machine_model_names: {metadata_stats['has_machine_model_names']} ({metadata_stats['has_machine_model_names']/len(nodes)*100:.1f}%)")
print(f"   Nodes with source_gcs: {metadata_stats['has_source_gcs']} ({metadata_stats['has_source_gcs']/len(nodes)*100:.1f}%)")
print()

if metadata_stats["missing_document_id"] > 0:
    print(f"⚠️  WARNING: {metadata_stats['missing_document_id']} nodes missing document_id")
    print()

print(f"📄 Unique document_ids: {len(document_ids)}")
print(f"📄 Unique source_gcs paths: {len(source_gcs_paths)}")
print()

# Sample some nodes to show metadata
print("📝 Sample node metadata (first 3 nodes):")
for i, (node_id, node_data) in enumerate(list(nodes.items())[:3]):
    metadata = node_data.get("metadata", {})
    print(f"\n   Node {i+1}:")
    print(f"     document_id: {metadata.get('document_id', 'MISSING')}")
    print(f"     file_name: {metadata.get('file_name', 'MISSING')}")
    print(f"     source_gcs: {metadata.get('source_gcs', metadata.get('gcs_path', 'MISSING'))}")
    print(f"     machine_model_ids: {metadata.get('machine_model_ids', 'MISSING')}")
    print(f"     machine_model_names: {metadata.get('machine_model_names', 'MISSING')}")
    print(f"     page_label: {metadata.get('page_label', 'N/A')}")

print()
print("=" * 70)
print("✅ Verification complete")
print("=" * 70)
```

**Run it:**
```bash
python3 verify_index_metadata.py
```

#### B) Check Database Metadata

```bash
# Run the verification script
python backend/scripts/verify_document_counts.py
```

This shows:
- Total documents in database
- Documents with GCS paths
- Which documents match the ingest query

#### C) Query Specific Document Metadata via API

```bash
# Get all documents (shows metadata)
curl -X GET "${BASE_URL}/documents" | jq '.documents[] | {filename, chunk_count, page_count, machine_model}' | head -20

# Get chunks for a specific document
curl -X GET "${BASE_URL}/admin/chunks/document?filename=DuraBolt%20Installation%20Guide_v6.0_18Aug2025.pdf" | jq '.chunks[0].metadata'
```

---

### 4. Verify Metadata from Your Ingestion Logs

From your logs, I can see metadata samples were logged. Check if they match:

**Expected metadata fields per node:**
- `document_id`: Should be present (e.g., "40", "41", "42")
- `machine_model_ids`: Array of integers (e.g., [2], [10, 11])
- `machine_model_names`: Array of strings (e.g., ["Duraflex"], ["DuraBolt", "DuraCore"])
- `source_gcs`: GCS path (e.g., "gs://arrow-rag-support-prod-docs/...")
- `file_name`: Filename

**From your logs, I see samples like:**
```json
{
  "event": "chunk_metadata_sample",
  "source_gcs": "gs://arrow-rag-support-prod-docs/DuraBolt Installation Guide_v6.0_18Aug2025.pdf",
  "document_id": "40",
  "machine_model": ["DuraBolt"],
  "machine_model_ids": [10],
  "machine_model_names": ["DuraBolt"]
}
```

This indicates metadata **was** being applied during chunking.

---

### 5. Quick Verification Checklist

Run these in order:

```bash
# 1. Check index files exist in GCS
gsutil ls gs://arrow-rag-support-prod-rag/latest_model/*.json

# 2. Test RAG status endpoint
curl "${BASE_URL}/rag/status" | jq '.status'  # Should return "ready"

# 3. Run self-test
curl "${BASE_URL}/rag/self-test" | jq '.status'  # Should return "ok"

# 4. Validate index files
curl "${BASE_URL}/rag/validate-index" | jq '.all_valid'  # Should return true

# 5. Test a query
curl -X POST "${BASE_URL}/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "DuraFlex", "top_k": 3}' | jq '.sources[0].metadata'

# 6. Check document counts
python backend/scripts/verify_document_counts.py
```

---

## Expected Results

### ✅ Success Indicators:

1. **Index Files:**
   - All 6 JSON files exist in GCS
   - File sizes are reasonable (not 0 bytes)
   - `docstore.json` contains ~17,427 nodes

2. **API Endpoints:**
   - `/rag/status` returns `"status": "ready"`
   - `/rag/self-test` returns `"status": "ok"` with `num_results > 0`
   - `/rag/validate-index` returns `"all_valid": true`
   - `/query` returns results with metadata

3. **Metadata:**
   - >95% of nodes have `document_id`
   - >95% of nodes have `machine_model_ids` and `machine_model_names`
   - >95% of nodes have `source_gcs`
   - Unique document_ids match expected count (~55 documents)

### ⚠️ Warning Signs:

- `all_valid: false` → Index files corrupted
- `status: "error"` → RAG not initialized
- `num_results: 0` → Index empty or query failed
- Many nodes missing `document_id` → Metadata not applied correctly

---

## Troubleshooting

**If metadata is missing:**
- Check if `DISABLE_METADATA_UPDATE=1` was set (this only affects DB updates, not node metadata)
- Verify documents were loaded from database (not just GCS/local)
- Check that `document_id` exists in source documents

**If index doesn't load:**
- Verify files exist in GCS: `gsutil ls gs://arrow-rag-support-prod-rag/latest_model/`
- Check `/rag/validate-index` for corrupted files
- Verify `RAG_INDEX_LOCAL_DIR` environment variable points to correct path

**If queries return no results:**
- Check `/rag/self-test` to verify index is queryable
- Verify embedding model loaded correctly
- Check that nodes were actually indexed (17,427 nodes expected)

