# RAG Diagnostic Tools

This directory contains diagnostic and utility scripts for validating and troubleshooting the RAG system.

## diagnose_rag_contract.py

Validates ingestion/query contract alignment across five critical checks:

1. **Index file naming and presence** - Verifies required index files exist
2. **Filename normalization alignment** - Compares DB filenames vs chunk metadata filenames
3. **Machine filtering split-brain** - Checks document-level vs chunk-level machine filtering
4. **Chunk machine_model_ids health** - Validates chunk metadata completeness
5. **Storage directory resolution** - Ensures consistent path resolution

### Usage

#### Download from GCS (Production Index) - Recommended

**Always use `--download-from-gcs` to analyze your current production index from GCP:**

```bash
# Download and analyze production index as ADMIN
python -m backend.tools.diagnose_rag_contract --download-from-gcs --role ADMIN

# Download and analyze production index as CUSTOMER
python -m backend.tools.diagnose_rag_contract \
  --download-from-gcs \
  --role CUSTOMER \
  --user-machine "EZCut 330"

# Multiple machines
python -m backend.tools.diagnose_rag_contract \
  --download-from-gcs \
  --role CUSTOMER \
  --user-machine "EZCut 330" \
  --user-machine "DuraFlex"
```

The script will:
1. Download the index from your GCS bucket (`settings.RAG_INDEX_GCS_BUCKET` / `settings.RAG_INDEX_GCS_PREFIX`)
2. Save it to the configured local directory (or a temporary directory)
3. Run all diagnostic checks on the downloaded production index

#### Local Index Check (Legacy - Use GCS Instead)

**Note:** These examples use local `latest_model` directory which may contain outdated data. Use `--download-from-gcs` for production analysis.

```bash
# From repo root
python -m backend.tools.diagnose_rag_contract --storage-dir latest_model --role ADMIN

# CUSTOMER check with local index
python -m backend.tools.diagnose_rag_contract \
  --storage-dir latest_model \
  --role CUSTOMER \
  --user-machine "EZCut 330"
```

This checks:
- Index files are present
- Filenames align between DB and chunks
- Chunk metadata is healthy
- Storage paths are consistent

This additionally checks:
- `allowed_filenames` is non-empty for CUSTOMER
- Chunks match allowed filenames (filename normalization)
- Machine filtering logic works correctly

### Interpreting Results

#### Exit Codes
- `0` = All checks PASS
- `1` = Critical FAIL (missing index files, empty allowed_filenames, etc.)
- `2` = WARN only (non-critical issues)

#### Common Failure Patterns

**FAIL on "Index files present"**
- Missing required files: `docstore.json`, `index_store.json`, `default__vector_store.json`
- **Fix**: Verify ingestion completed successfully and check storage directory path

**FAIL on "Machine filtering (doc vs chunk)" with `allowed_filenames=0`**
- CUSTOMER role sees zero allowed documents
- **Fix**: Check `Document.machine_model` field in database - may be empty/stale or user machines don't match

**FAIL on "Machine filtering (doc vs chunk)" with `allowed_filenames>0` but `chunks_in_allowed=0`**
- Documents are allowed but no chunks match
- **Fix**: Filename normalization mismatch - DB filenames don't match chunk metadata `file_name` values

**WARN/FAIL on "Filename alignment" where `base_intersection >> raw_intersection`**
- Basename matching improves overlap significantly
- **Fix**: Normalization bug - unify basename handling in ingestion and query code

### Example Output

```
Using storage_dir: /path/to/latest_model

 ⚠️ Missing file_name key: 0 nodes
 ⚠️ Empty file_name string: 0 nodes
 Canonical match rate: 100.0%

 Top 25 offending filenames (chunks not in DB):
   (none - all chunks match DB)

 ❌ ADMIN match rate 95.2% < 95% threshold - filename alignment issue

Docstore sample nodes:
  1. node_id=abc123... file_name=document.pdf machine_model_ids=[1, 2] content_type=text page_label=1
  2. node_id=def456... file_name=manual.pdf machine_model_ids=[] content_type=text page_label=2

==============================================================================================================
Summary:
==============================================================================================================
Check                                    | Status | Key numbers          | Notes
--------------------------------------------------------------------------------------------------------------
Storage path resolution                  | PASS   | -                    | chosen=/path/to/latest_model
Index files present                      | PASS   | 3 json files         | Required OK. Vector-like: ['default__vector_store.json']
Filename alignment                       | PASS   | DB=50 chunks=1000 missing_key=0 empty=0 | canonical_intersection=50/50 match_rate=100.0%
Machine filtering (doc vs chunk)        | PASS   | 45/1000 (match_rate=4.5%) | allowed_filenames=50 chunks_in_allowed=45/1000
Chunk machine_model_ids                 | PASS   | total_nodes=1000     | missing=0, empty=200, non_list=0
==============================================================================================================

✅ PASS: All checks passed.
```

### New Diagnostic Features

The enhanced diagnostics now include:

1. **Canonical Filename Comparison**
   - Compares canonicalized filenames between DB and index
   - Shows match rate percentage
   - Identifies top 25 offending filenames (chunks not in DB)

2. **Missing/Empty Filename Detection**
   - Counts nodes missing `file_name` key
   - Counts nodes with empty `file_name` string
   - FAILs if >5% of nodes have missing/empty filenames

3. **Stricter ADMIN Role Checks**
   - FAILs if match rate < 95% for ADMIN role
   - Ensures filename alignment is working correctly

4. **Enhanced Machine Filtering Analysis**
   - Shows match rate for machine filtering
   - Identifies empty filename issues
   - Better diagnostics for CUSTOMER role filtering

### Troubleshooting

**GCS Download Issues:**
- Ensure `google-cloud-storage` is installed: `pip install google-cloud-storage`
- Verify GCS credentials are configured (service account key or default credentials)
- Check environment variables: `RAG_INDEX_GCS_BUCKET` and `RAG_INDEX_GCS_PREFIX`
- The script uses the same download logic as production, so if production works, this should too

**Filename Mismatches:**
- Check ingestion code: How `file_name` is set in chunk metadata
- Check database: What `Document.file_name` values are stored
- Check normalization: Whether paths are included vs just basenames

**CUSTOMER Sees Zero Results:**
- Check `Document.machine_model` field - should be JSON array or list
- Verify user's assigned machines match document machine models
- Check `Document.is_active` - inactive docs are filtered out
- Verify chunks have non-empty `machine_model_ids` for customer-visible documents

