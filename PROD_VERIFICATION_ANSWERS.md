# Production Verification Answers

## Q1) Is the data in the PRODUCTION database?

### 1) What to Run

**Step 1: Get Production DATABASE_URL from Cloud Run**

```bash
# Set variables
export PROJECT_ID="arrow-rag-support-prod"
export REGION="us-central1"
export SERVICE_NAME="arrow-rag-backend"

# Extract DATABASE_URL (masked for security)
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep "DATABASE_URL" | cut -d'=' -f2- | head -c 50
```

**Step 2: Connect to Cloud SQL**

**Option A: Cloud SQL Auth Proxy (Windows Git Bash)**

```bash
# Download proxy if needed (already in repo: tools/cloud-sql-proxy.exe)
# Start proxy in background
./tools/cloud-sql-proxy.exe arrow-rag-support-prod:us-central1:rag-postgres --port=5432 &

# Wait for connection
sleep 3

# Connect via psql (replace USER and PASSWORD from DATABASE_URL)
psql "host=127.0.0.1 port=5432 dbname=neondb user=rag_user sslmode=disable"
# OR if dbname is rag_app:
# psql "host=127.0.0.1 port=5432 dbname=rag_app user=rag_user sslmode=disable"
```

**Option B: gcloud sql connect**

```bash
gcloud sql connect rag-postgres \
  --user=rag_user \
  --database=neondb \
  --project=$PROJECT_ID
# OR if database is rag_app:
# gcloud sql connect rag-postgres --user=rag_user --database=rag_app --project=$PROJECT_ID
```

**Step 3: Run Verification SQL**

```sql
-- Single comprehensive query
WITH migration_check AS (
  SELECT version_num as current_version FROM alembic_version
),
table_counts AS (
  SELECT 'tickets_index' as table_name, COUNT(*) as row_count FROM tickets_index
  UNION ALL SELECT 'tickets_detail', COUNT(*) FROM tickets_detail
  UNION ALL SELECT 'ticket_judgements', COUNT(*) FROM ticket_judgements
  UNION ALL SELECT 'ticket_manual_reviews', COUNT(*) FROM ticket_manual_reviews
  UNION ALL SELECT 'ticket_machine_model_matches', COUNT(*) FROM ticket_machine_model_matches
),
orphan_checks AS (
  SELECT 
    'ticket_judgements missing tickets_detail' as check_name,
    COUNT(*) as orphan_count
  FROM ticket_judgements j
  LEFT JOIN tickets_detail t ON j.ticket_id = t.ticket_id
  WHERE t.ticket_id IS NULL
  UNION ALL
  SELECT 
    'ticket_manual_reviews missing ticket_judgements',
    COUNT(*)
  FROM ticket_manual_reviews m
  LEFT JOIN ticket_judgements j ON m.ticket_id = j.ticket_id
  WHERE j.ticket_id IS NULL
),
cache_eligible_count AS (
  SELECT COUNT(*) as eligible_count
  FROM ticket_judgements j
  LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
  WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
    OR (m.manual_status = 'approved')
  )
  AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
  AND j.cache_eligible = true
)
SELECT 
  'MIGRATION' as check_type,
  current_version as result_value,
  NULL::bigint as numeric_value
FROM migration_check
UNION ALL
SELECT 
  'TABLE_COUNT',
  table_name,
  row_count
FROM table_counts
UNION ALL
SELECT 
  'ORPHAN_CHECK',
  check_name,
  orphan_count
FROM orphan_checks
UNION ALL
SELECT 
  'CACHE_ELIGIBLE',
  'cache_eligible_tickets',
  eligible_count
FROM cache_eligible_count
ORDER BY check_type, result_value;
```

### 2) What Success Looks Like

**Expected Output:**

```
check_type      | result_value                                    | numeric_value
----------------+-------------------------------------------------+---------------
CACHE_ELIGIBLE  | cache_eligible_tickets                          | 234
MIGRATION       | 011_ticket_tables_postgres                      | NULL
ORPHAN_CHECK    | ticket_judgements missing tickets_detail       | 0
ORPHAN_CHECK    | ticket_manual_reviews missing ticket_judgements | 0
TABLE_COUNT     | ticket_judgements                                | 1234
TABLE_COUNT     | ticket_manual_reviews                            | 56
TABLE_COUNT     | ticket_machine_model_matches                      | 567
TABLE_COUNT     | tickets_detail                                   | 1234
TABLE_COUNT     | tickets_index                                    | 1234
```

**Success Criteria:**
- `MIGRATION` shows `011_ticket_tables_postgres` (or later migration)
- All `TABLE_COUNT` rows show `numeric_value > 0`
- Both `ORPHAN_CHECK` rows show `numeric_value = 0`
- `CACHE_ELIGIBLE` shows `numeric_value > 0` (tickets available for cache)

### 3) Where This is Configured in Code

**DATABASE_URL Source:**
- `.github/workflows/ci.yml:635` - Set via `secrets.DATABASE_URL` in Cloud Run deployment
- Format: `postgresql+psycopg2://user:pass@host:5432/dbname` (from `backend/utils/db.py:89-95`)

**Cloud SQL Instance:**
- `.github/workflows/ci.yml:636` - `--set-cloudsql-instances="arrow-rag-support-prod:us-central1:rag-postgres"`
- `.github/workflows/ci.yml:565` - Instance name: `arrow-rag-support-prod:us-central1:rag-postgres`

**Canonical Eligibility Predicate:**
- `Scraper/export_cache_artifacts.py:167-187` - `EFFECTIVE_CACHE_ELIGIBLE_SQL` query
- Exact SQL logic matches `backend/orchestrator.py:5224-5275` (`_is_ticket_cache_eligible`)

**Migration Check:**
- `backend/utils/migration_runner.py:267-295` - `check_migration_status()` function
- `backend/migrations/versions/011_ticket_tables_postgres.py` - Expected migration file

### 4) If It Fails, Top 3 Likely Causes

1. **Migration Not Applied:** Alembic version is not `011_ticket_tables_postgres` or later
   - **Check:** `SELECT version_num FROM alembic_version;`
   - **Fix:** Run migrations: `python scripts/db_migrate.py upgrade head` (requires DATABASE_URL)

2. **Wrong Database Name:** Connection uses `neondb` but data is in `rag_app` (or vice versa)
   - **Check:** Extract `dbname=` from DATABASE_URL: `gcloud run services describe ... | grep DATABASE_URL | grep -o 'dbname=[^&]*'`
   - **Fix:** Connect to correct database name

3. **Data Not Migrated:** Tables exist but are empty (migration ran but data wasn't copied)
   - **Check:** All `TABLE_COUNT` values are `0`
   - **Fix:** Run ticket migration script: `python scripts/migrate_tickets_sqlite_to_postgres.py` (if source SQLite exists)

---

## Q2) Is the PRODUCTION index actually being served containing ticket_cache nodes?

### 1) What to Run

**Step 1: Verify GCS Bucket/Prefix Configuration**

```bash
# Show RAG index env vars from Cloud Run
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --project=arrow-rag-support-prod \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep -E "(RAG_INDEX_GCS_BUCKET|RAG_INDEX_GCS_PREFIX|RAG_INDEX_LOCAL_DIR)"
```

**Step 2: List GCS Index Objects**

```bash
# List files in latest_model prefix
gsutil ls -lh gs://arrow-rag-support-prod-rag/latest_model/*.json | \
  awk '{print $5, $NF}'

# Verify required files exist
REQUIRED_FILES="docstore.json index_store.json default__vector_store.json index_manifest.json"
for file in $REQUIRED_FILES; do
  if gsutil ls "gs://arrow-rag-support-prod-rag/latest_model/$file" > /dev/null 2>&1; then
    SIZE=$(gsutil ls -lh "gs://arrow-rag-support-prod-rag/latest_model/$file" | awk '{print $5}')
    echo "✅ $file exists (size: $SIZE)"
  else
    echo "❌ $file missing"
  fi
done
```

**Step 3: Download and Verify Index Contains ticket_cache Nodes**

```bash
# Download index to local temp directory
mkdir -p /tmp/prod_index_check
gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/* /tmp/prod_index_check/

# Run proof script against downloaded index
python backend/scripts/proof_ticket_cache_nodes.py --index-dir /tmp/prod_index_check
```

**Alternative: One-Liner Python Check (if proof script unavailable)**

```bash
python -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('.').resolve()))
from llama_index.core import load_index_from_storage, StorageContext

storage_context = StorageContext.from_defaults(persist_dir='/tmp/prod_index_check')
index = load_index_from_storage(storage_context)
docstore = index.storage_context.docstore

total = 0
ticket_cache = 0
samples = []

for node_id in docstore.docs.keys():
    try:
        node = docstore.get_document(node_id)
        total += 1
        metadata = getattr(node, 'metadata', {}) or (getattr(node, 'node', None) and getattr(node.node, 'metadata', {})) or {}
        if metadata.get('content_type') == 'ticket_cache':
            ticket_cache += 1
            if len(samples) < 5:
                samples.append((node_id, metadata.get('ticket_id', 'unknown')))
    except:
        continue

print(f'Total nodes: {total}')
print(f'Ticket cache nodes: {ticket_cache}')
print(f'Sample IDs: {samples[:5]}')
"
```

### 2) What Success Looks Like

**Expected Output from Proof Script:**

```
📁 Using local index directory: /tmp/prod_index_check
📖 Loading index from /tmp/prod_index_check...
✅ Index loaded successfully
🔍 Analyzing nodes...
📊 Results:
   Total nodes: 12,345
   Ticket cache nodes: 234
   Percentage: 1.90%
📋 Sample ticket_cache node IDs (first 5):
   - node_id: ticket:12345, ticket_id: 12345
   - node_id: ticket:12346, ticket_id: 12346
   - node_id: ticket:12347, ticket_id: 12347
   - node_id: ticket:12348, ticket_id: 12348
   - node_id: ticket:12349, ticket_id: 12349
✅ Proof complete: Index contains 234 ticket_cache nodes
```

**Success Criteria:**
- `Ticket cache nodes: > 0` (e.g., `234`)
- Sample IDs match pattern `ticket:{ticket_id}` (e.g., `ticket:12345`)
- All required GCS files exist with non-zero sizes

### 3) Where This is Configured in Code

**GCS Bucket/Prefix Configuration:**
- `.github/workflows/ci.yml:635` - `RAG_INDEX_GCS_BUCKET=arrow-rag-support-prod-rag,RAG_INDEX_GCS_PREFIX=latest_model/`
- `backend/config/env.py:261-270` - Defaults: `RAG_INDEX_GCS_BUCKET="arrow-rag-support-prod-rag"`, `RAG_INDEX_GCS_PREFIX="latest_model/"`

**Local Directory:**
- `.github/workflows/ci.yml:635` - `RAG_INDEX_LOCAL_DIR=/tmp/latest_model`
- `backend/config/env.py:272-275` - Default: `"/tmp/latest_model"` in prod/Cloud Run

**Index Download Path:**
- `backend/rag/startup_downloader.py:129-166` - `download_index_from_gcs()` function
- `backend/rag/index_manager.py:343-358` - Downloads on startup if files missing
- `backend/rag/startup_downloader.py:166` - Logs: `"[RAG] Starting GCS index download from gs://{bucket}/{prefix}..."`

**Proof Script:**
- `backend/scripts/proof_ticket_cache_nodes.py` - Counts nodes with `metadata.content_type == "ticket_cache"`

### 4) If It Fails, Top 3 Likely Causes

1. **Ticket Cache Artifacts Not Ingested:** Index exists but has 0 ticket_cache nodes
   - **Check:** `Ticket cache nodes: 0` in proof script output
   - **Fix:** Run promotion workflow (Section 7 in `PROD_TICKET_CACHE_VALIDATION_RUNBOOK.md`)

2. **Wrong GCS Prefix:** Cloud Run downloads from different prefix than expected
   - **Check:** Compare `RAG_INDEX_GCS_PREFIX` env var with actual GCS path: `gsutil ls gs://arrow-rag-support-prod-rag/`
   - **Fix:** Update Cloud Run env var or verify correct prefix contains index

3. **Index Download Failed:** Cloud Run couldn't download index from GCS (permissions or network)
   - **Check:** Cloud Run logs: `gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-backend AND textPayload=~\"RAG.*Download\"" --limit=10`
   - **Fix:** Verify service account has `storage.objects.get` permission on bucket

---

## Q3) Is the runtime path returning a cache hit in PRODUCTION?

### 1) What to Run

**Step 1: Get Service URL and Verify Environment**

```bash
# Set variables
export PROJECT_ID="arrow-rag-support-prod"
export REGION="us-central1"
export SERVICE_NAME="arrow-rag-backend"

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --project=$PROJECT_ID \
  --format="value(status.url)")

echo "Service URL: $SERVICE_URL"

# Verify we're hitting prod (check CORS origin)
gcloud run services describe $SERVICE_NAME \
  --region=$REGION \
  --format="value(spec.template.spec.containers[0].env)" | \
  tr ',' '\n' | grep "CORS_ALLOWED_ORIGINS"
# Expected: CORS_ALLOWED_ORIGINS=https://support.arrsys.com
```

**Step 2: Check RAG Status (No Auth Required)**

```bash
# GET /rag/status (public endpoint, no auth)
curl -X GET "${SERVICE_URL}/rag/status" | jq '{
  status: .status,
  initialized: .initialized,
  rag_enabled: .rag_enabled,
  storage_dir: .storage_dir,
  missing_files: .missing_files,
  download_status: .download_status
}'
```

**Step 3: Get Auth Token (if /query requires it)**

**Note:** `/query` endpoint reads `X-User-Token` header but doesn't require it (`backend/api.py:3171-3187`). However, for full functionality, get a token:

**Option A: Use gcloud identity token (for testing)**

```bash
# Get identity token (works if your account has Cloud Run Invoker role)
TOKEN=$(gcloud auth print-identity-token --audiences="${SERVICE_URL}")
```

**Option B: Extract from browser (if logged into frontend)**

```bash
# If you have browser DevTools open on https://support.arrsys.com:
# 1. Open Network tab
# 2. Find any API request to backend
# 3. Copy "X-User-Token" header value
# Then set:
# TOKEN="<paste_token_here>"
```

**Step 4: Find a Query That Should Match (from DB)**

```bash
# Connect to Cloud SQL (see Q1) and run:
# psql "host=127.0.0.1 port=5432 dbname=neondb user=rag_user sslmode=disable"
```

```sql
-- Get a sample ticket problem text (first 50 words) for testing
SELECT 
    j.ticket_id,
    LEFT(j.raw_response_json->>'problem', 200) as problem_preview
FROM ticket_judgements j
LEFT JOIN ticket_manual_reviews m ON j.ticket_id = m.ticket_id
WHERE (
    (j.review_status = 'approved' OR (j.review_status IS NULL AND j.cache_eligible = true))
    OR (m.manual_status = 'approved')
)
AND (m.manual_status IS NULL OR m.manual_status != 'rejected')
AND j.cache_eligible = true
LIMIT 1;
```

**Step 5: Test /query Endpoint**

```bash
# Replace QUERY_TEXT with problem_preview from Step 4 (first 20-30 words)
QUERY_TEXT="How do I fix error X in my machine?"

# Test with token (if available)
curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -H "X-User-Token: ${TOKEN}" \
  -d "{
    \"query\": \"${QUERY_TEXT}\",
    \"top_k\": 10
  }" | jq '{
    cache_hit: .cache_hit,
    ticket_cache_sources: [.sources[] | select(.content_type == "ticket_cache")],
    answer_preview: .answer[0:200],
    confidence: .confidence,
    response_time_ms: .response_time_ms
  }'

# Test without token (should still work, but may have limited functionality)
curl -X POST "${SERVICE_URL}/query" \
  -H "Content-Type: application/json" \
  -d "{
    \"query\": \"${QUERY_TEXT}\",
    \"top_k\": 10
  }" | jq '.cache_hit, .sources[] | select(.content_type == "ticket_cache")'
```

**Step 6: Check Cloud Run Logs for Ticket Cache Lookup**

```bash
# Check logs for ticket cache hit markers
gcloud logging read \
  "resource.type=cloud_run_revision AND \
   resource.labels.service_name=$SERVICE_NAME AND \
   (textPayload=~\"🎫 Served from ticket cache\" OR textPayload=~\"ticket cache\" OR textPayload=~\"Ticket cache lookup\")" \
  --limit=20 \
  --format="table(timestamp, textPayload)" \
  --project=$PROJECT_ID

# Check logs for index download/ready
gcloud logging read \
  "resource.type=cloud_run_revision AND \
   resource.labels.service_name=$SERVICE_NAME AND \
   (textPayload=~\"RAG.*Download\" OR textPayload=~\"Index download\" OR textPayload=~\"RAG.*ready\")" \
  --limit=10 \
  --format="table(timestamp, textPayload)" \
  --project=$PROJECT_ID
```

### 2) What Success Looks Like

**Expected Output from /rag/status:**

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

**Expected Output from /query:**

```json
{
  "cache_hit": true,
  "ticket_cache_sources": [
    {
      "id": "ticket:12345",
      "name": "Ticket #12345",
      "content_type": "ticket_cache"
    }
  ],
  "answer_preview": "Based on a similar resolved ticket (#12345):\n\nProblem: ...",
  "confidence": 0.85,
  "response_time_ms": 234
}
```

**Expected Log Output:**

```
timestamp: 2024-01-15T10:30:00Z
textPayload: 🎫 Served from ticket cache! ticket_id=12345, score=0.852
```

**Success Criteria:**
- `/rag/status` shows `"status": "ready"` and `"initialized": true`
- `/query` returns `"cache_hit": true`
- `ticket_cache_sources` array contains at least one entry with `"content_type": "ticket_cache"` and `"id"` starting with `"ticket:"`
- Logs show `"🎫 Served from ticket cache!"` message
- `response_time_ms` is typically < 1000ms for cache hits (vs 3000-5000ms for full RAG)

### 3) Where This is Configured in Code

**Service URL:**
- `.github/workflows/ci.yml:630-647` - Cloud Run deployment config
- Service name: `arrow-rag-backend` (line 630)

**/rag/status Endpoint:**
- `backend/api.py:2042-2121` - `rag_status_public()` function
- **No auth required** (line 2042: `@app.get("/rag/status", include_in_schema=False)`)
- Returns `storage_dir`, `missing_files`, `download_status`

**/query Endpoint:**
- `backend/api.py:2999-3532` - `query_knowledge_base()` function
- **Auth is optional** (`backend/api.py:3171-3187` - reads `X-User-Token` but doesn't fail if missing)
- Returns `cache_hit` flag (`backend/api.py:3523`)

**Ticket Cache Lookup:**
- `backend/orchestrator.py:4646-4699` - Ticket cache lookup in `orchestrate_query()`
- `backend/orchestrator.py:4682` - Logs: `"🎫 Served from ticket cache! ticket_id={ticket_id}, score={score}"`
- `backend/orchestrator.py:4696` - Sets `cache_hit=True` in `StructuredResponse`

**Index Download Logging:**
- `backend/rag/startup_downloader.py:166` - Logs: `"[RAG] Starting GCS index download from gs://{bucket}/{prefix}..."`
- `backend/rag/index_manager.py:365-367` - Logs: `"Index download completed successfully in {duration:.2f}s"`

**CORS/Environment Verification:**
- `.github/workflows/ci.yml:635` - `CORS_ALLOWED_ORIGINS=https://support.arrsys.com` (prod indicator)

### 4) If It Fails, Top 3 Likely Causes

1. **Index Not Loaded:** `/rag/status` shows `"status": "error"` or `"initialized": false`
   - **Check:** `/rag/status` response and Cloud Run logs for download errors
   - **Fix:** Verify GCS bucket permissions, check `RAG_INDEX_GCS_BUCKET` env var, restart Cloud Run revision

2. **No ticket_cache Nodes in Index:** Index loaded but cache hits never occur
   - **Check:** Run Q2 verification (proof script shows 0 ticket_cache nodes)
   - **Fix:** Run promotion workflow to ingest ticket cache artifacts

3. **Similarity Threshold Too High:** Query matches ticket_cache nodes but score below threshold
   - **Check:** Logs show `"No ticket_cache nodes match"` or `"score below threshold"`
   - **Fix:** Lower `TICKET_CACHE_THRESHOLD` env var (default 0.75, try 0.60): `gcloud run services update arrow-rag-backend --region=us-central1 --update-env-vars="TICKET_CACHE_THRESHOLD=0.60"`

---

## Summary: One-Time Verification Checklist

Run these commands in order to verify all three aspects:

```bash
# Q1: Database verification
gcloud sql connect rag-postgres --user=rag_user --database=neondb --project=arrow-rag-support-prod
# Then run SQL from Q1 Step 3

# Q2: Index verification
gsutil ls -lh gs://arrow-rag-support-prod-rag/latest_model/*.json
mkdir -p /tmp/prod_index_check && gsutil -m cp -r gs://arrow-rag-support-prod-rag/latest_model/* /tmp/prod_index_check/
python backend/scripts/proof_ticket_cache_nodes.py --index-dir /tmp/prod_index_check

# Q3: Runtime verification
SERVICE_URL=$(gcloud run services describe arrow-rag-backend --region=us-central1 --project=arrow-rag-support-prod --format="value(status.url)")
curl "${SERVICE_URL}/rag/status" | jq '.status, .initialized'
# Then test /query with a known ticket problem text
```

**All checks pass = Ticket cache lookup is working in production.**
