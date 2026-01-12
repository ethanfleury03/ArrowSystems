# Scraper Pipeline Workflow

## Full Pipeline: From Scraping to RAG Export

### Stage 1: Data Ingestion (Re-scraping)

**1. Index All Requests** (Cheap operation)
```bash
python index_requests.py
```
- Logs into Zendesk via Selenium
- Fetches all requests via search API
- Stores in `tickets_index` table
- **File:** `index_requests.py` ✅

**2. Build Detailed Conversations** (Expensive operation)
```bash
python build_solved_tickets.py
```
- Reads solved ticket IDs from `tickets_index`
- Fetches detailed conversations via API
- Stores normalized JSON in `tickets_detail`
- **File:** `build_solved_tickets.py` ✅

**3. Optional: Triage Stage** (Cost optimization)
```bash
python ticket_triage.py
```
- Uses Claude Haiku to triage tickets
- Filters out definitely non-cache-eligible tickets
- **File:** `ticket_triage.py` ✅

### Stage 2: Cache Eligibility Judgment

**4. Judge Cache Eligibility**
```bash
python judge_ticket_cache_eligibility.py
```
- Main judgment pipeline using Claude Sonnet
- Stores judgments in `ticket_judgements` table
- **File:** `judge_ticket_cache_eligibility.py` ✅

**5. Validate Pipeline**
```bash
python validate_cache_pipeline.py
```
- Validates judgment consistency
- Checks for errors/issues
- **File:** `validate_cache_pipeline.py` ✅

**6. Manual Review** (Interactive)
```bash
python manual_review.py
```
- Interactive review tool for manual approval/rejection
- Updates `ticket_manual_reviews` table
- **File:** `manual_review.py` ✅

**7. Coverage Audit**
```bash
python coverage_audit.py
```
- Audits coverage of cache-eligible tickets
- **File:** `coverage_audit.py` ✅

**8. Schema Verification**
```bash
python verify_raw_response_schema.py
```
- Verifies `raw_response_json` schema consistency
- **File:** `verify_raw_response_schema.py` ✅

### Stage 3: RAG Export

**9. Export Cache Artifacts**
```bash
python export_cache_artifacts.py --db data/tickets.db --out out/cache_artifacts.jsonl
```
- Exports cache-eligible tickets to JSONL
- **File:** `export_cache_artifacts.py` ✅

## Supporting Files

- **`db.py`** - Database layer (used by all scripts) ✅
- **`config.py`** - Configuration loader ✅
- **`requirements.txt`** - Dependencies ✅

## Documentation

- **`CACHE_ARTIFACTS_USAGE.md`** - Export/ingestion usage ✅
- **`PIPELINE_VERIFICATION_REPORT.md`** - Verification report ✅
- **`PRE_REVIEW_CHECKLIST_RESULTS.md`** - Review checklist ✅

## Summary

**All core pipeline files are present and functional:**
- ✅ Data ingestion: `index_requests.py`, `build_solved_tickets.py`, `ticket_triage.py`
- ✅ Cache eligibility: `judge_ticket_cache_eligibility.py`, `validate_cache_pipeline.py`, `manual_review.py`, `coverage_audit.py`, `verify_raw_response_schema.py`
- ✅ RAG export: `export_cache_artifacts.py`
- ✅ Supporting: `db.py`, `config.py`, `requirements.txt`

**Deleted files were:**
- Old scraper infrastructure (replaced by modular scripts)
- Test/utility scripts (not needed for production)
- One-time scripts (already run)
- Old deprecated pipeline scripts (replaced by new versions)
