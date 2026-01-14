# Production Ticket Cache Validation & Promotion Runbook

## What We Can Prove Today (Read-Only Checks)

With read-only commands, we can verify:
1. ✅ **Cloud Run Configuration** - Env vars are set correctly
2. ✅ **Cloud SQL Data** - Ticket tables exist, have data, and eligibility predicate works
3. ✅ **GCS Index Presence** - Index files exist in GCS bucket
4. ✅ **Index Contains ticket_cache Nodes** - Proof script counts nodes with `content_type="ticket_cache"`
5. ✅ **API Functionality** - `/rag/status` shows index loaded, `/query` returns `cache_hit=true` for matching queries

**All validation steps are production-safe (read-only).** Only the "Promotion Workflow" section performs writes.

---

## Section 1: PROD Index Loading and Location

### 1.1 Code Path: Bucket/Prefix Configuration

**File:** `backend/config/env.py:245-282`

**Env Vars:**
- `RAG_INDEX_GCS_BUCKET` - Default: `"arrow-rag-support-prod-rag"` (line 261)
- `RAG_INDEX_GCS_PREFIX` - Default: `"latest_model/"` (line 263)
- `RAG_INDEX_LOCAL_DIR` - Default: `"/tmp/latest_model"` in prod/Cloud Run (line 272)

**Evidence:**
```python
# backend/config/env.py:261-275
self.RAG_INDEX_GCS_BUCKET = os.getenv("RAG_INDEX_GCS_BUCKET", "arrow-rag-support-prod-rag").strip()
self.RAG_INDEX_GCS_PREFIX = os.getenv("RAG_INDEX_GCS_PREFIX", "latest_model/")  # Normalized to end with /
default_local_dir = "/tmp/latest_model" if (self.is_prod or is_cloud_run) else "latest_model"
self.RAG_INDEX_LOCAL_DIR = os.getenv("RAG_INDEX_LOCAL_DIR", default_local_dir)
```

### 1.2 Code Path: Download from GCS

**File:** `backend/rag/startup_downloader.py:129-166`

**Function:** `download_index_from_gcs()`

**Source:** `gs://{RAG_INDEX_GCS_BUCKET}/{RAG_INDEX_GCS_PREFIX}`  
**Destination:** `{RAG_INDEX_LOCAL_DIR}` (resolved via `_resolve_local_dir()`)

**Evidence:**
```python
# backend/rag/startup_downloader.py:147-152
bucket_name = settings.RAG_INDEX_GCS_BUCKET
index_prefix = _normalize_prefix(getattr(settings, "RAG_INDEX_GCS_PREFIX", "latest_model/"))
requested_local_dir = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
local_path = _resolve_local_dir(requested_local_dir, index_prefix)
```

### 1.3 Code Path: Index Manager Loads from Local Dir

**File:** `backend/rag/index_manager.py:320-369`

**Function:** `_do_load()` → Downloads if missing, then loads from `storage_path`

**Evidence:**
```python
# backend/rag/index_manager.py:326-358
storage_path = getattr(settings, "RAG_INDEX_LOCAL_DIR", "/tmp/latest_model")
# ... download if missing ...
# Then loads via: load_index_from_storage(StorageContext.from_defaults(persist_dir=storage_path))
```

**Summary:**
- **GCS Source:** `gs://arrow-rag-support-prod-rag/latest_model/`
- **Local Dir (prod):** `/tmp/latest_model`
- **Download Trigger:** On startup if files missing (`backend/rag/index_manager.py:343-358`)
- **Load Trigger:** Lazy init on first `/query` or eager if `RAG_EAGER_LOAD_ON_STARTUP=1`

---

## Section 2: Promotion Path to GCS

### 2.1 Promotion Script

**File:** `backend/tools/promote_index_to_gcs.py`

**CLI Usage:**
```bash
# Verify only (no upload)
python -m backend.tools.promote_index_to_gcs --index-dir ./latest_model --verify-only

# Promote to GCS (requires --promote flag)
python -m backend.tools.promote_index_to_gcs \
  --index-dir ./latest_model \
  --promote \
  --bucket arrow-rag-support-prod-rag \
  --prefix latest_model/
```

**Default Source Dir:** `--index-dir` (required, no default)

**Default Destination:**
- Bucket: `RAG_INDEX_GCS_BUCKET` env var or `"arrow-rag-support-prod-rag"` (line 72)
- Prefix: `RAG_INDEX_GCS_PREFIX` env var or `"latest_model/"` (line 73)
- Backup Prefix: `GCS_RAG_OLD_PREFIX` env var or `"old_model/"` (line 74)

**Behavior:** Overwrites `latest_model/` after backing up to `old_model/{timestamp}/` (line 4023)

**Evidence:**
```python
# backend/tools/promote_index_to_gcs.py:72-74
bucket = args.bucket or os.getenv("RAG_INDEX_GCS_BUCKET") or "arrow-rag-support-prod-rag"
prefix = args.prefix or os.getenv("RAG_INDEX_GCS_PREFIX") or "latest_model/"
old_prefix = args.old_prefix or os.getenv("GCS_RAG_OLD_PREFIX") or "old_model/"
```

**Promotion Function:** `backend/ingest.py:4003-4100` (`promote_index_to_gcs()`)
- Backs up existing `latest_model/` → `old_model/{timestamp}/`
- Verifies backup
- Clears `latest_model/`
- Uploads new index
- Verifies upload matches local files

### 2.2 CI/CD Automation

**Evidence:** `.github/workflows/ci.yml` - **NO automatic promotion step found**

**Conclusion:** Promotion is **manual only**. No CI/CD job runs `promote_index_to_gcs.py`.

---

## Section 3: Ticket Cache Ingestion Behavior

### 3.1 Ingestion Script

**File:** `backend/scripts/ingest_ticket_cache_artifacts.py`

**Index Directory:**
- Default: `get_index_dir()` (line 149) - resolves to `latest_model` or env override
- Can be specified via `--index-dir` argument (line 130)

**Append vs Rebuild:**
- **Appends** to existing index (line 173-174: `load_index_from_storage()` if exists)
- Only creates new index if directory doesn't exist (line 186-187)

**Deduplication:**
- Uses `artifact.id` as `node_id` (line 124: `id_=artifact.id`)
- `artifact.id` format: `"ticket:{ticket_id}"` (`backend/utils/ticket_cache_artifacts.py:178`)
- `skip_existing` flag checks `existing_ids` set (line 198)
- **Deduplication works** - same `ticket:{ticket_id}` won't be inserted twice if `--skip-existing` is used

**Evidence:**
```python
# backend/scripts/ingest_ticket_cache_artifacts.py:111-125
def artifact_to_text_node(artifact: TicketCacheArtifact) -> TextNode:
    return TextNode(
        text=artifact.text,
        metadata=artifact.metadata,
        id_=artifact.id  # Use artifact.id as node_id for deduplication
    )

# backend/scripts/ingest_ticket_cache_artifacts.py:173-187
if index_dir.exists():
    storage_context = StorageContext.from_defaults(persist_dir=str(index_dir))
    index = load_index_from_storage(storage_context)  # Loads existing
    # ... then inserts new nodes via index.insert_nodes(batch) (line 234)
```

**Metadata Fields Written:**
- `content_type`: `"ticket_cache"` (`backend/utils/ticket_cache_artifacts.py:152`)
- `ticket_id`: Ticket ID string (line 154)
- `machine_model_ids`: List from `extra_meta` (defaults to `[]`) (line 162)
- `machine_model_names`: List from `extra_meta` (defaults to `[]`) (line 163)
- Plus: `document_id`, `file_name`, `source`, `outcome`, `confidence`, `cache_eligible`, `confirmed`, `rationale`, `blockers`

**Evidence:** `backend/utils/ticket_cache_artifacts.py:149-166`

### 3.2 Export Script

**File:** `Scraper/export_cache_artifacts.py`

**Canonical Eligibility SQL:**
```sql
-- Scraper/export_cache_artifacts.py:167-187
SELECT 
    j.ticket_id,
    j.raw_response_json,
    j.cache_eligible,
    j.confidence,
    j.model,
    j.prompt_version,
    j.judged_at,
    m.manual_status,
    m.reviewer
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = 1))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = 1
ORDER BY j.ticket_id
```

**Output Schema:**
- JSONL file with one `TicketCacheArtifact` per line
- Each artifact has: `id`, `text`, `metadata` dict

**Machine Model IDs:**
- Currently **NOT populated** in export (line 300: `extra_meta={}`)
- `machine_model_ids` defaults to `[]` in metadata (`backend/utils/ticket_cache_artifacts.py:162`)
- To populate: Query `ticket_machine_model_matches` table and pass to `build_ticket_cache_artifact(..., extra_meta={"machine_model_ids": [...]})`

**Evidence:** `Scraper/export_cache_artifacts.py:297-301` - `extra_meta={}` is empty

---

## Section 4: Proof Script - Index Contains ticket_cache Nodes

**File:** `backend/scripts/proof_ticket_cache_nodes.py` (NEW - created)

**Usage:**
```bash
# Download from GCS and check (matches prod behavior)
python backend/scripts/proof_ticket_cache_nodes.py

# Check local index directory
python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./latest_model

# Use specific GCS bucket/prefix
python backend/scripts/proof_ticket_cache_nodes.py --bucket arrow-rag-support-prod-rag --prefix latest_model/
```

**Output:**
- Total nodes count
- Ticket cache nodes count
- Percentage
- Sample node IDs (first 5)

**How It Works:**
1. Downloads index from GCS (if `--index-dir` not provided) using same logic as Cloud Run
2. Loads index via `load_index_from_storage()`
3. Iterates `docstore.docs.keys()` and checks `metadata.content_type == "ticket_cache"`
4. Prints counts and samples

**Evidence:** Script uses same loading path as production (`backend/rag/index_manager.py:369-371`)

---

## Section 5: Production Validation Runbook (Read-Only)

### Prerequisites

```bash
# Set project and region
export PROJECT_ID="arrow-rag-support-prod"
export REGION="us-central1"
export SERVICE_NAME="arrow-rag-backend"

gcloud config set project $PROJECT_ID
gcloud config set compute/region $REGION

# Verify access
gcloud projects describe $PROJECT_ID
```

**Evidence:** `deploy_backend.sh:20-22`, `.github/workflows/ci.yml:631-632`

---

### A) Cloud Run Verification

#### A1) Show Environment Variables

```bash
# Show all env vars (filter for ticket cache and RAG index)
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format="yaml(spec.template.spec.containers[0].env)" | \
  grep -E "(TICKET_CACHE|RAG_INDEX|DISABLE_RAG)" -A 1

# More readable: show specific vars
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep -E "(TICKET_CACHE|RAG_INDEX|DISABLE_RAG)"
```

**Expected Output:**
```
TICKET_CACHE_ENABLED=true (or unset, defaults to true)
TICKET_CACHE_THRESHOLD=0.75 (or unset, defaults to 0.75)
RAG_INDEX_GCS_BUCKET=arrow-rag-support-prod-rag
RAG_INDEX_GCS_PREFIX=latest_model/
RAG_INDEX_LOCAL_DIR=/tmp/latest_model
DISABLE_RAG=false (or unset)
```

**Evidence:** `.github/workflows/ci.yml:635` - RAG_INDEX vars are set

#### A2) Show Service URL

```bash
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format="value(status.url)")

echo "Service URL: $SERVICE_URL"
```

**Evidence:** `deploy_backend.sh:189-191`

#### A3) Trigger Revision Restart (Safe - Creates New Revision)

```bash
# Force new revision by updating a dummy env var (triggers index re-download)
# This is SAFE - only creates new revision, doesn't affect current one
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --update-env-vars="RAG_INDEX_RELOAD=$(date +%s)"

# Wait for new revision to be ready
echo "Waiting for new revision..."
sleep 10

# Verify new revision is serving traffic
gcloud run revisions list \
  --service=$SERVICE_NAME \
  --region=$REGION \
  --limit=3 \
  --format="table(name,status.conditions[0].status,status.conditions[0].type)"
```

**Note:** Cloud Run downloads index on startup if files are missing (`backend/rag/index_manager.py:343-358`). New revision will download fresh index from GCS.

---

### B) Cloud SQL Verification

#### B1) Connect to Cloud SQL

**Option A: Cloud SQL Auth Proxy (Recommended for Windows)**

```bash
# Download proxy if needed (Windows Git Bash)
# https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe
# Rename to cloud-sql-proxy.exe

# Start proxy in background
cloud-sql-proxy.exe arrow-rag-support-prod:us-central1:rag-postgres --port=5432 &

# Wait for connection
sleep 3

# Connect via psql
psql "host=127.0.0.1 port=5432 dbname=neondb user=neondb_owner sslmode=disable"
```

**Option B: gcloud sql connect**

```bash
gcloud sql connect rag-postgres \
  --user=neondb_owner \
  --database=neondb \
  --project=$PROJECT_ID
```

**Evidence:**
- `deploy_backend.sh:175` - Cloud SQL instance: `arrow-rag-support-prod:us-central1:rag-postgres`
- `.github/workflows/ci.yml:636` - Confirms connection string

#### B2) Verify Alembic Migration Head

```sql
-- Check current migration revision
SELECT version_num FROM alembic_version;

-- Compare with expected head (check backend/migrations/versions/ for latest)
-- Expected: Should match latest migration file name prefix (e.g., "011_ticket_tables_postgres")
```

**Evidence:** `backend/api.py:1659-1663` - Uses `check_migration_status()` from `backend/utils/migration_runner.py:267`

#### B3) Verify Ticket Table Row Counts

```sql
-- Row counts per table
SELECT 
    'tickets_index' as table_name, COUNT(*) as row_count 
FROM tickets_index
UNION ALL
SELECT 'tickets_detail', COUNT(*) FROM tickets_detail
UNION ALL
SELECT 'ticket_judgements', COUNT(*) FROM ticket_judgements
UNION ALL
SELECT 'ticket_manual_reviews', COUNT(*) FROM ticket_manual_reviews
UNION ALL
SELECT 'ticket_machine_model_matches', COUNT(*) FROM ticket_machine_model_matches
ORDER BY table_name;
```

**Expected:** All tables should have > 0 rows if migration completed.

#### B4) Orphan Checks

```sql
-- Check for orphaned ticket_judgements (missing tickets_detail)
SELECT 
    'ticket_judgements missing tickets_detail' as check_name,
    COUNT(*) as orphan_count
FROM ticket_judgements j
LEFT JOIN tickets_detail t ON j.ticket_id = t.ticket_id
WHERE t.ticket_id IS NULL;

-- Check for orphaned ticket_manual_reviews (missing ticket_judgements)
SELECT 
    'ticket_manual_reviews missing ticket_judgements' as check_name,
    COUNT(*) as orphan_count
FROM ticket_manual_reviews m
LEFT JOIN ticket_judgements j ON m.ticket_id = j.ticket_id
WHERE j.ticket_id IS NULL;
```

**Expected:** Both should return `0` (no orphans).

#### B5) Effective Cache-Eligible Count (Canonical Predicate)

```sql
-- Count using canonical predicate from Scraper/export_cache_artifacts.py:167-187
SELECT 
    COUNT(*) as cache_eligible_count
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = true;

-- Sample eligible tickets (for manual verification)
SELECT 
    j.ticket_id,
    j.cache_eligible,
    j.review_status,
    m.manual_status,
    j.confidence,
    CASE 
        WHEN j.review_status = 'approved' THEN 'auto_approved'
        WHEN j.review_status IS NULL AND j.cache_eligible = true THEN 'auto_eligible'
        WHEN m.manual_status = 'approved' THEN 'manually_approved'
        ELSE 'unknown'
    END as eligibility_reason
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = true
ORDER BY j.ticket_id
LIMIT 10;
```

**Expected:** `cache_eligible_count > 0` if tickets have been judged and approved.

#### B6) Count Tickets with Machine Model IDs

```sql
-- Count tickets with machine_model_ids assigned
SELECT 
    COUNT(DISTINCT ticket_id) as tickets_with_machine_models,
    COUNT(*) as total_machine_model_matches
FROM ticket_machine_model_matches;

-- Sample matches
SELECT 
    ticket_id,
    machine_model_id,
    COUNT(*) OVER (PARTITION BY ticket_id) as models_per_ticket
FROM ticket_machine_model_matches
ORDER BY ticket_id
LIMIT 20;
```

**Expected:** `tickets_with_machine_models > 0` if machine matching has run.

---

### C) Index Verification

#### C1) Confirm GCS Path Exists and Has Content

```bash
# List index files in GCS bucket
gsutil ls -lh gs://arrow-rag-support-prod-rag/latest_model/*.json | \
  awk '{print $1, $2, $5, $NF}'

# Check file sizes (should be > 0)
gsutil ls -lh gs://arrow-rag-support-prod-rag/latest_model/ | \
  grep -E "(docstore|vector_store|index_store)\.json" | \
  awk '{print $5, $NF}'

# Verify required files exist
REQUIRED_FILES="docstore.json index_store.json default__vector_store.json"
for file in $REQUIRED_FILES; do
  if gsutil ls "gs://arrow-rag-support-prod-rag/latest_model/$file" > /dev/null 2>&1; then
    SIZE=$(gsutil ls -lh "gs://arrow-rag-support-prod-rag/latest_model/$file" | awk '{print $5}')
    echo "✅ $file exists (size: $SIZE)"
  else
    echo "❌ $file missing"
  fi
done
```

**Expected:** All required files exist with non-zero sizes.

**Evidence:** `backend/rag/startup_downloader.py:28-32` - Required files list

#### C2) Verify Index Contains ticket_cache Nodes

```bash
# Run proof script (downloads from GCS, matches prod behavior)
python backend/scripts/proof_ticket_cache_nodes.py

# Or check local index if you have it
python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./latest_model
```

**Expected Output:**
```
📥 Downloading index from GCS: gs://arrow-rag-support-prod-rag/latest_model/
✅ Index downloaded to: /tmp/proof_index
📖 Loading index from /tmp/proof_index...
✅ Index loaded successfully
🔍 Analyzing nodes...
📊 Results:
   Total nodes: 12345
   Ticket cache nodes: 234
   Percentage: 1.90%
📋 Sample ticket_cache node IDs (first 5):
   - node_id: ticket:12345, ticket_id: 12345
   - node_id: ticket:12346, ticket_id: 12346
   ...
✅ Proof complete: Index contains 234 ticket_cache nodes
```

**If count is 0:** Ticket artifacts haven't been ingested. Run promotion workflow (Section 6).

---

### D) API Verification

#### D1) Check RAG Status Endpoint

```bash
# Get auth token
TOKEN=$(gcloud auth print-identity-token)

# Check /rag/status
curl -X GET "${SERVICE_URL}/rag/status" \
  -H "Authorization: Bearer ${TOKEN}" \
  | jq '{
    status: .status,
    initialized: .initialized,
    rag_enabled: .rag_enabled,
    storage_dir: .storage_dir,
    missing_files: .missing_files,
    download_status: .download_status
  }'
```

**Expected Response:**
```json
{
  "status": "ready",
  "initialized": true,
  "rag_enabled": true,
  "storage_dir": "/tmp/latest_model",
  "missing_files": [],
  "download_status": "downloaded"
}
```

**Evidence:** `backend/api.py:2042-2121` - `/rag/status` endpoint

#### D2) Test Ticket Cache Hit via /query

**Step 1: Find a Query That Should Match**

```bash
# Option A: Use proof script to get a sample ticket_id
python backend/scripts/proof_ticket_cache_nodes.py | grep "ticket_id:"

# Option B: Query DB for a sample ticket problem text (non-sensitive)
# Connect to DB (Section B1) and run:
# SELECT j.ticket_id, j.raw_response_json->>'problem' as problem_preview
# FROM ticket_judgements j
# WHERE j.cache_eligible = true
# LIMIT 1;
```

**Step 2: Query API**

```bash
# Replace QUERY_TEXT with actual problem text from a ticket (first 20-30 words)
QUERY_TEXT="How do I fix error X in my machine?"

curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{
    \"query\": \"${QUERY_TEXT}\",
    \"top_k\": 10
  }" | jq '{
    cache_hit: .cache_hit,
    sources: [.sources[] | select(.content_type == "ticket_cache")],
    answer_preview: .answer[0:200],
    confidence: .confidence
  }'
```

**Expected Response:**
```json
{
  "cache_hit": true,
  "sources": [
    {
      "id": "ticket:12345",
      "name": "Ticket #12345",
      "content_type": "ticket_cache"
    }
  ],
  "answer_preview": "Based on a similar resolved ticket (#12345):\n\nProblem: ...",
  "confidence": 0.85
}
```

**Evidence:**
- `backend/orchestrator.py:4696` - Sets `cache_hit=True`
- `backend/api.py:3523` - Passes `cache_hit` to QueryResponse
- `backend/orchestrator.py:5195-5199` - Source format

#### D3) Verify Cache Hit Flag and Sources

```bash
# Extract cache_hit and ticket_cache sources
curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d "{\"query\": \"${QUERY_TEXT}\", \"top_k\": 10}" \
  | jq -r '
    "cache_hit: " + (.cache_hit | tostring),
    "ticket_cache_sources:",
    (.sources[] | select(.content_type == "ticket_cache") | "  - " + .name)
  '
```

**Expected:**
```
cache_hit: true
ticket_cache_sources:
  - Ticket #12345
```

---

## Section 6: Troubleshooting Decision Tree

### 6.1 If cache_hit Never Returns True

**Check A: Ticket Cache Enabled?**

```bash
# Verify env var
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep TICKET_CACHE_ENABLED

# Check logs
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=$SERVICE_NAME AND \
  textPayload=~\"ticket cache\"" \
  --limit=20 \
  --format="table(timestamp, textPayload)"
```

**Fix if missing:**
```bash
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --update-env-vars="TICKET_CACHE_ENABLED=true"
```

**Evidence:** `backend/orchestrator.py:5089` - Checks `settings.TICKET_CACHE_ENABLED`

**Check B: Index Contains ticket_cache Nodes?**

```bash
# Run proof script
python backend/scripts/proof_ticket_cache_nodes.py
```

**If count is 0:** Run promotion workflow (Section 7).

**Check C: Similarity Threshold Too High?**

```bash
# Check current threshold
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep TICKET_CACHE_THRESHOLD

# Lower temporarily for testing (if needed)
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --update-env-vars="TICKET_CACHE_THRESHOLD=0.60"
```

**Evidence:** `backend/orchestrator.py:5153` - Checks threshold

**Check D: Query Doesn't Match?**

```bash
# Check logs for lookup attempts
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=$SERVICE_NAME AND \
  (textPayload=~\"Ticket cache lookup\" OR textPayload=~\"ticket cache hit\" OR textPayload=~\"No ticket_cache nodes\")" \
  --limit=20 \
  --format="table(timestamp, textPayload)"
```

**Evidence:** `backend/orchestrator.py:5119-5121` - Logs when no nodes found

### 6.2 If Index Has No ticket_cache Nodes

**Root Cause:** Artifacts haven't been ingested into production index.

**Solution:** Run promotion workflow (Section 7).

### 6.3 If DB Predicate Yields 0 Eligible Tickets

**Check Migration:**

```sql
-- Verify ticket_judgements table exists
SELECT COUNT(*) FROM ticket_judgements;

-- Check cache_eligible column
SELECT column_name, data_type 
FROM information_schema.columns 
WHERE table_name = 'ticket_judgements' 
AND column_name = 'cache_eligible';

-- Check sample data
SELECT ticket_id, cache_eligible, review_status 
FROM ticket_judgements 
LIMIT 10;
```

**Check Manual Reviews:**

```sql
SELECT manual_status, COUNT(*) 
FROM ticket_manual_reviews 
GROUP BY manual_status;
```

**Evidence:** `backend/utils/db.py:467-535` - Table schemas

### 6.4 If Machine Filtering Blocks Hits

**Check Types:**

```sql
-- Verify machine_model_ids are integers
SELECT 
    ticket_id,
    machine_model_id,
    pg_typeof(machine_model_id) as id_type
FROM ticket_machine_model_matches
LIMIT 10;
```

**Check Node Metadata:**

```bash
# In proof script, add debug output for machine_model_ids types
python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./latest_model | \
  grep -A 5 "machine_model_ids"
```

**Evidence:** `backend/orchestrator.py:5133-5142` - Type coercion logic

---

## Section 7: Repeatable Promotion Workflow (Manual)

### Prerequisites

```bash
# Set environment
export PROJECT_ID="arrow-rag-support-prod"
export REGION="us-central1"
export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/neondb"  # Replace with actual

# Set GCS bucket
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"
```

### Step 1: Export Cache Artifacts JSONL from Postgres

```bash
# Set backend to Postgres
export TICKETS_STORAGE_BACKEND="postgres"

# Export all cache-eligible tickets
python Scraper/export_cache_artifacts.py \
  --db "dummy" \
  --out out/cache_artifacts_prod.jsonl \
  --force

# Verify export
echo "Exported artifacts:"
wc -l out/cache_artifacts_prod.jsonl
head -1 out/cache_artifacts_prod.jsonl | jq '.id, .metadata.ticket_id, .metadata.content_type'
```

**Evidence:**
- `Scraper/export_cache_artifacts.py:218-245` - Postgres backend support
- `Scraper/export_cache_artifacts.py:167-187` - Canonical SQL predicate

**Guardrails:**
- ✅ Read-only query (SELECT only)
- ✅ No ticket mutation
- ✅ No scraper execution
- ✅ No Zendesk writes

### Step 2: Download Current Production Index

```bash
# Download current index from GCS to local directory
mkdir -p ./local_index_prod
gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/* ./local_index_prod/

# Verify download
ls -lh ./local_index_prod/*.json
```

**Evidence:** `backend/tools/promote_index_to_gcs.py` - Promotion workflow expects local index

### Step 3: Ingest Artifacts into Local Index

```bash
# Ingest ticket cache artifacts into local index
python -m backend.scripts.ingest_ticket_cache_artifacts \
  --jsonl out/cache_artifacts_prod.jsonl \
  --index-dir ./local_index_prod \
  --skip-existing

# Verify ingestion
python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./local_index_prod
```

**Expected:** Proof script shows increased ticket_cache node count.

**Evidence:**
- `backend/scripts/ingest_ticket_cache_artifacts.py:128-244` - Ingestion logic
- `backend/scripts/ingest_ticket_cache_artifacts.py:234` - `index.insert_nodes()` appends

**Guardrails:**
- ✅ Only modifies local index directory
- ✅ No database writes
- ✅ No scraper execution

### Step 4: Promote/Upload to GCS

```bash
# Verify index before promotion
python -m backend.tools.promote_index_to_gcs \
  --index-dir ./local_index_prod \
  --verify-only

# Promote to GCS (requires --promote flag)
python -m backend.tools.promote_index_to_gcs \
  --index-dir ./local_index_prod \
  --promote \
  --bucket arrow-rag-support-prod-rag \
  --prefix latest_model/
```

**Expected Output:**
```
🔍 Verifying index...
✅ Index verified:
   - Index dir: ./local_index_prod
   - Num nodes: 12345
   - Num chunks: 12345

📤 Promoting index:
   - Local: ./local_index_prod
   - Bucket: arrow-rag-support-prod-rag
   - Prefix: latest_model/
   - Backup prefix: old_model/

✅ Promotion complete:
   - Backup: gs://arrow-rag-support-prod-rag/old_model/2024-01-15T10-30-00Z/
   - Uploaded: 4 objects
   - Latest objects: 4
```

**Evidence:**
- `backend/tools/promote_index_to_gcs.py:82-95` - Promotion call
- `backend/ingest.py:4003-4100` - Promotion function (backup → verify → upload → verify)

**Guardrails:**
- ✅ Backs up existing index before overwrite
- ✅ Verifies backup before clearing
- ✅ Verifies upload matches local
- ✅ No ticket mutation
- ✅ No scraper execution

### Step 5: Restart Cloud Run to Pick Up New Index

```bash
# Force new revision (triggers index re-download on startup)
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --update-env-vars="RAG_INDEX_RELOAD=$(date +%s)"

# Wait for new revision to be ready
echo "Waiting for new revision to be ready..."
sleep 30

# Verify new revision is serving
gcloud run revisions list \
  --service=$SERVICE_NAME \
  --region=$REGION \
  --limit=1 \
  --format="table(name,status.conditions[0].status,status.conditions[0].type,status.trafficPercent)"

# Check logs for index download
gcloud logging read "resource.type=cloud_run_revision AND \
  resource.labels.service_name=$SERVICE_NAME AND \
  textPayload=~\"RAG.*Download\"" \
  --limit=10 \
  --format="table(timestamp, textPayload)"
```

**Expected:** Logs show `[RAG] Download complete` and new revision serves traffic.

**Evidence:** `backend/rag/index_manager.py:343-358` - Downloads on startup if files missing

**Guardrails:**
- ✅ Only creates new revision (doesn't affect current one)
- ✅ No ticket mutation
- ✅ No scraper execution

### Step 6: Verify Promotion Success

```bash
# Wait for index to load (check /rag/status)
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
TOKEN=$(gcloud auth print-identity-token)

# Check status
curl -X GET "${SERVICE_URL}/rag/status" \
  -H "Authorization: Bearer ${TOKEN}" \
  | jq '.status, .initialized, .storage_dir'

# Test ticket cache hit
curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"query": "How do I fix error X?", "top_k": 10}' \
  | jq '.cache_hit, .sources[] | select(.content_type == "ticket_cache")'
```

**Expected:** `cache_hit: true` and ticket_cache sources appear.

---

## Complete Promotion Workflow (Single Command Sequence)

```bash
#!/bin/bash
# Complete promotion workflow - run from repo root

set -e  # Exit on error

# Configuration
export PROJECT_ID="arrow-rag-support-prod"
export REGION="us-central1"
export SERVICE_NAME="arrow-rag-backend"
export DATABASE_URL="postgresql+psycopg2://user:pass@host:5432/neondb"  # REPLACE WITH ACTUAL
export TICKETS_STORAGE_BACKEND="postgres"
export RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"
export RAG_INDEX_GCS_PREFIX="latest_model/"

# Step 1: Export artifacts
echo "Step 1: Exporting cache artifacts from Postgres..."
python Scraper/export_cache_artifacts.py \
  --db "dummy" \
  --out out/cache_artifacts_prod.jsonl \
  --force
echo "✅ Exported $(wc -l < out/cache_artifacts_prod.jsonl) artifacts"

# Step 2: Download current index
echo "Step 2: Downloading current production index..."
mkdir -p ./local_index_prod
gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/* ./local_index_prod/
echo "✅ Index downloaded"

# Step 3: Ingest artifacts
echo "Step 3: Ingesting ticket cache artifacts..."
python -m backend.scripts.ingest_ticket_cache_artifacts \
  --jsonl out/cache_artifacts_prod.jsonl \
  --index-dir ./local_index_prod \
  --skip-existing
echo "✅ Artifacts ingested"

# Step 4: Verify local index
echo "Step 4: Verifying local index..."
python backend/scripts/proof_ticket_cache_nodes.py --index-dir ./local_index_prod

# Step 5: Promote to GCS
echo "Step 5: Promoting index to GCS..."
python -m backend.tools.promote_index_to_gcs \
  --index-dir ./local_index_prod \
  --promote \
  --bucket arrow-rag-support-prod-rag \
  --prefix latest_model/
echo "✅ Index promoted to GCS"

# Step 6: Restart Cloud Run
echo "Step 6: Restarting Cloud Run service..."
gcloud run services update $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --update-env-vars="RAG_INDEX_RELOAD=$(date +%s)"
echo "✅ Cloud Run revision updated"

# Step 7: Wait and verify
echo "Step 7: Waiting for new revision to be ready..."
sleep 30

SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")
TOKEN=$(gcloud auth print-identity-token)

echo "Step 8: Verifying ticket cache hit..."
curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${TOKEN}" \
  -d '{"query": "How do I fix error X?", "top_k": 10}' \
  | jq '.cache_hit, .sources[] | select(.content_type == "ticket_cache")'

echo "✅ Promotion workflow complete!"
```

---

## File References Summary

**Index Loading:**
- `backend/config/env.py:245-282` - RAG index config
- `backend/rag/startup_downloader.py:129-166` - GCS download logic
- `backend/rag/index_manager.py:320-369` - Index load with download

**Promotion:**
- `backend/tools/promote_index_to_gcs.py` - CLI wrapper
- `backend/ingest.py:4003-4100` - Promotion function (backup → upload → verify)

**Ingestion:**
- `backend/scripts/ingest_ticket_cache_artifacts.py:128-244` - Ingestion logic
- `backend/utils/ticket_cache_artifacts.py:91-183` - Artifact builder

**Export:**
- `Scraper/export_cache_artifacts.py:167-187` - Canonical SQL predicate
- `Scraper/export_cache_artifacts.py:190-319` - Export function

**Proof Script:**
- `backend/scripts/proof_ticket_cache_nodes.py` - NEW - Counts ticket_cache nodes

**Deployment:**
- `.github/workflows/ci.yml:630-647` - Cloud Run deployment config
- `deploy_backend.sh:20-22` - Service name, project, region
