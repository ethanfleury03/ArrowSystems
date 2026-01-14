# Ticket Integration Context Analysis

**Generated:** 2025-01-XX  
**Purpose:** Comprehensive context gathering for integrating migrated tickets + cache-eligible pipeline into live ticket system

---

## Findings Summary

This repo is a **FastAPI + Next.js RAG system** for DuraFlex technical documentation. It already has:
- ✅ Complete ticket scraper pipeline (`Scraper/`) that processes Zendesk tickets
- ✅ Ticket tables migrated to PostgreSQL (`backend/utils/db.py`, migration `011_ticket_tables_postgres`)
- ✅ Cache eligibility judgment pipeline (`Scraper/judge_ticket_cache_eligibility.py`)
- ✅ Admin UI for ticket management (`frontend/app/admin/tickets/`)
- ✅ Ticket cache artifacts transformer (`backend/utils/ticket_cache_artifacts.py`)
- ❌ **NOT FOUND:** Live ticket cache lookup in RAG query pipeline (no cache hit implementation)
- ❌ **NOT FOUND:** Auto-send/draft workflow for ticket replies
- ❌ **NOT FOUND:** Outbound Zendesk API integration (only scraping via Selenium)

---

## 0) Quick Repo Map

### Q0.1: Top-level Services/Modules

| Directory | Purpose |
|-----------|---------|
| `backend/` | FastAPI backend (Python) - RAG orchestrator, API endpoints, database models |
| `frontend/` | Next.js frontend - Admin UI, chat interface, query insights dashboard |
| `Scraper/` | Zendesk ticket scraping pipeline - indexing, detail building, cache eligibility judgment |
| `deployment/` | Cloud Run deployment configs (YAML, scripts) |
| `scripts/` | Operational scripts - migrations, validation, smoke tests |
| `config/` | YAML configuration files (app config, user config) |
| `data/` | PDF documents for RAG ingestion |
| `latest_model/` | Vector index storage (RAG artifacts) |

**Key Files:**
- `backend/api.py` - FastAPI entrypoint
- `backend/orchestrator.py` - RAG orchestration engine
- `backend/utils/db.py` - SQLAlchemy models (including ticket tables)
- `Scraper/judge_ticket_cache_eligibility.py` - Cache eligibility judgment
- `backend/utils/ticket_cache_artifacts.py` - Ticket → RAG artifact transformer

### Q0.2: FastAPI Entrypoint

**File:** `backend/api.py`

**Entrypoint Function:** `main()` (lines 7651-7823)

**Dev Command:**
```bash
python -m backend.api --dev --reload
# OR
python -m backend.api --host 0.0.0.0 --port 8000 --dev --reload
```

**Prod Command (Gunicorn):**
```bash
gunicorn backend.api:app \
    --workers 3 \
    --worker-class uvicorn.workers.UvicornWorker \
    --bind 0.0.0.0:8080 \
    --timeout 300
```

**Startup Script:** `scripts/start_api.sh` (handles dev/prod detection)

**Docker:** `backend/Dockerfile.backend` (entrypoint: `python -m backend.api`)

### Q0.3: Docker/Cloud Run Deployment Configs

**Dockerfiles:**
- `backend/Dockerfile.backend` - Backend container
- `frontend/Dockerfile` - Frontend container

**Docker Compose:**
- `docker-compose.yml` - Main compose file
- `docker-compose.dev.yml` - Development overrides
- `docker-compose.prod.yml` - Production overrides

**Cloud Run:**
- `.github/workflows/ci.yml` - CI/CD pipeline (lines 616-647: backend deployment)
- `deployment/` directory - Additional deployment configs (if any)

**Key Deployment Env Vars (from CI):**
- `DATABASE_URL` - PostgreSQL connection string
- `JWT_SECRET_KEY` - JWT signing secret
- `ANTHROPIC_API_KEY` - Claude API key (optional)
- `RAG_INDEX_GCS_BUCKET` - GCS bucket for RAG index
- `DOCS_GCS_BUCKET` - GCS bucket for documents
- `CORS_ALLOWED_ORIGINS` - CORS whitelist

---

## 1) Ticket System Integration: What Exists Today

### Q1.1: Zendesk Integration

**Files Found:**
- `backend/utils/scraper_delta.py` - Delta check logic for identifying new solved tickets from Zendesk
- `Scraper/index_requests.py` - Indexes all Zendesk requests via search API
- `Scraper/build_solved_tickets.py` - Builds detailed conversations for solved tickets
- `Scraper/ticket_store_postgres.py` - PostgreSQL ticket store implementation
- `Scraper/ticket_store_sqlite.py` - SQLite ticket store (legacy)

**Integration Method:** **Selenium-based scraping** (NOT direct API integration)

**Zendesk Credentials:**
- `Scraper/.env` - Contains `ZENDESK_EMAIL` and `ZENDESK_PASSWORD`
- Loaded by `backend/utils/scraper_delta.py::load_credentials()`

**Webhook Integration:** **NOT FOUND** - No inbound webhook endpoints found

**Search Terms Checked:**
- ✅ "zendesk" - Found in scraper modules
- ✅ "ticket" - Found extensively (tables, admin UI, scripts)
- ❌ "webhook" - NOT FOUND
- ❌ "comment" - NOT FOUND (only in ticket conversation JSON)
- ❌ "reply" - NOT FOUND
- ❌ "support" - NOT FOUND (only in bucket names)
- ❌ "agent" - NOT FOUND
- ❌ "trigger" - NOT FOUND

### Q1.2: Inbound Endpoints for Ticket Ingestion

**NOT FOUND** - No webhook endpoints found for ticket ingestion.

**Existing Admin Endpoints:**
- `GET /admin/tickets` - List tickets (pagination, search, sorting)
- `GET /admin/tickets/{ticketId}` - Get ticket details
- `PATCH /admin/tickets/{ticketId}` - Update ticket (cache_eligible, confidence, review_status, machine_model_names)

**File:** `backend/routes/admin_routes.py` (lines 1400-1600+)

**Auth:** JWT token required (via `X-User-Token` header or cookie)

**Request Schema (PATCH):**
```python
# From frontend/app/api/admin/tickets/[ticketId]/route.ts
{
  cache_eligible?: boolean,
  confidence?: number,
  review_status?: string,
  outcome?: string,
  machine_model_names?: string[]
}
```

### Q1.3: Outbound Calls to Ticket System API

**NOT FOUND** - No outbound Zendesk API calls found.

**Only Scraping:** The system only **scrapes** tickets via Selenium, it does not:
- Post comments to tickets
- Update ticket status
- Tag tickets
- Create tickets

**Scraper Flow:**
1. `index_requests.py` - Lists all requests via Zendesk search API (read-only)
2. `build_solved_tickets.py` - Fetches detailed conversations (read-only)
3. `judge_ticket_cache_eligibility.py` - Judges cache eligibility (local processing)

---

## 2) Data Model + DB Access for Tickets (Postgres)

### Q2.1: SQLAlchemy Models

**File:** `backend/utils/db.py` (lines 398-600)

| Table | Model Class | Primary Key | Foreign Keys |
|-------|-------------|-------------|--------------|
| `tickets_index` | `TicketIndex` | `ticket_id` (String) | None |
| `tickets_detail` | `TicketDetail` | `ticket_id` (String) | `tickets_index.ticket_id` (CASCADE) |
| `ticket_summaries` | `TicketSummary` | `ticket_id` (String) | `tickets_detail.ticket_id` (CASCADE) |
| `ticket_judgements` | `TicketJudgement` | `ticket_id` (String) | `tickets_detail.ticket_id` (CASCADE) |
| `ticket_triage` | `TicketTriage` | `ticket_id` (String) | `tickets_detail.ticket_id` (CASCADE) |
| `ticket_manual_reviews` | `TicketManualReview` | `ticket_id` (String) | `ticket_judgements.ticket_id` (CASCADE) |
| `ticket_machine_model_matches` | `TicketMachineModelMatch` | Composite: `(ticket_id, machine_model_id, match_source)` | None (ticket_id is not FK) |
| `ticket_machine_model_assignment` | `TicketMachineModelAssignment` | `ticket_id` (String) | None |
| `scrape_runs` | `ScrapeRun` | `run_id` (String) | None |

**Migration File:** `backend/migrations/versions/011_ticket_tables_postgres.py`

### Q2.2: Primary Keys and Foreign Keys

**Primary Keys:**
- `tickets_index`: `ticket_id` (String(255))
- `tickets_detail`: `ticket_id` (String(255), FK to `tickets_index.ticket_id`)
- `ticket_summaries`: `ticket_id` (String(255), FK to `tickets_detail.ticket_id`)
- `ticket_judgements`: `ticket_id` (String(255), FK to `tickets_detail.ticket_id`)
- `ticket_triage`: `ticket_id` (String(255), FK to `tickets_detail.ticket_id`)
- `ticket_manual_reviews`: `ticket_id` (String(255), FK to `ticket_judgements.ticket_id`)
- `ticket_machine_model_matches`: Composite PK `(ticket_id, machine_model_id, match_source)`
- `ticket_machine_model_assignment`: `ticket_id` (String(255))
- `scrape_runs`: `run_id` (String(255))

**Foreign Key Relationships:**
```
tickets_index (root)
  └─ tickets_detail (CASCADE delete)
      ├─ ticket_summaries (CASCADE delete)
      ├─ ticket_judgements (CASCADE delete)
      │   └─ ticket_manual_reviews (CASCADE delete)
      └─ ticket_triage (CASCADE delete)

ticket_machine_model_matches (no FK, ticket_id is just a string)
ticket_machine_model_assignment (no FK, ticket_id is just a string)
```

### Q2.3: DB Engine/Session Creation

**File:** `backend/utils/db.py` (lines 43-146)

**DATABASE_URL Usage:**
- Function: `_get_database_url()` (line 43)
- Source: `settings.DATABASE_URL` (from `backend/config/env.py`) OR `os.environ["DATABASE_URL"]`
- Validation: Rejects SQLite URLs (raises RuntimeError)
- Required: Yes (except if `DEV_SKIP_DB=true`)

**Connection Pooling:**
```python
engine = create_engine(
    database_url,
    pool_pre_ping=True,      # Verify connections before using
    pool_recycle=3600,        # Recycle after 1 hour
    future=True,
)
```

**Session Factory:**
```python
SessionLocal = scoped_session(
    sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        future=True
    )
)
```

**SQLite Fallback:** **NOT SUPPORTED** - System explicitly rejects SQLite URLs (line 77-82)

**Read/Write Separation:** **NOT FOUND** - Single engine/session factory, no read replicas

### Q2.4: Alembic Migrations

**Config File:** `backend/migrations/alembic.ini`
- `script_location = backend/migrations`
- `prepend_sys_path = .`

**Env File:** `backend/migrations/env.py`
- Loads `DATABASE_URL` from `backend.utils.db`
- Sets SQLAlchemy URL dynamically: `config.set_main_option("sqlalchemy.url", DATABASE_URL)`

**Current Head:** `011_ticket_tables_postgres` (revision ID)

**Migration Chain:**
```
001_initial_schema
  → 002_schema_fixes
  → 003_ingestion_phase1
  → 004_documents_and_glossary
  → 005_add_conversation_id
  → 006_add_language_fields
  → 007_add_auth_tokens
  → 008_add_machine_kind
  → 009_add_printer_machine_kind
  → 010_document_machine_models_m2m
  → 011_ticket_tables_postgres (HEAD)
```

**Running Migrations:**
- Dev: Automatic on startup (`backend/api.py` calls `run_migrations()`)
- Prod: Manual via Cloud Run Job (see `.github/workflows/ci.yml` lines 557-604)

**Helper Scripts:**
- `scripts/db_migrate.py` - Wrapper for Alembic commands
- `backend/scripts/run_alembic.py` - Direct Alembic runner
- `backend/utils/migration_runner.py` - Migration utilities

---

## 3) "Cache Eligible" Pipeline: Current Behavior

### Q3.1: Cache Eligibility Computation

**File:** `Scraper/judge_ticket_cache_eligibility.py`

**Main Function:** `judge_ticket()` (line 2018)

**Pipeline Stages:**
1. **Hard-block check** (`hard_block()` function) - Deterministic rules (line 2303)
2. **Hard-allow check** (`hard_allow()` function) - Deterministic rules
3. **Sonnet judgment** - LLM-based classification (Claude Sonnet)

**Decision Logic:** `determine_review_status()` (line 1917)

**Cache Eligible Criteria:**
- `outcome == "resolved_remotely_actionable"`
- `confirmation.confirmed == True` (with evidence grounding)
- `len(resolution.steps) >= 1`
- `confidence >= approve_min_confidence` (default: 0.90)
- Not hard-blocked

**Confidence Thresholds:**
- `approve_min_confidence`: 0.90 (default)
- `reject_min_confidence`: 0.60 (default)
- `unclear_reject_min_confidence`: 0.80 (default)

**Blockers List:**
- Stored in `ticket_judgements.blockers_json` (JSONB array)
- Examples: `["missing_confirmation", "missing_resolution_steps", "outcome_needs_onsite"]`

**Review Status Values:**
- `"approved"` - Auto-approved (meets all criteria)
- `"rejected"` - Auto-rejected (hard-block or non-cacheable outcome)
- `"needs_review"` - Borderline case (requires manual review)

### Q3.2: Cache Eligibility Storage

**Table:** `ticket_judgements`

**Authoritative Columns:**
- `cache_eligible` (Boolean) - **PRIMARY** cache eligibility flag
- `confidence` (Float) - Model confidence (0.0-1.0)
- `review_status` (String) - `"approved"`, `"rejected"`, or `"needs_review"`
- `raw_response_json` (JSONB) - Full LLM response (source of truth)

**Model:** `backend/utils/db.py::TicketJudgement` (line 467)

**Indexes:**
- `idx_ticket_judgements_cache_eligible` - On `cache_eligible`
- `idx_ticket_judgements_review_status` - On `review_status`

### Q3.3: Cache Hit Definition

**NOT IMPLEMENTED** - No cache hit lookup found in RAG query pipeline.

**User-Validated Cache (Different):**
- `backend/orchestrator.py` lines 4589-4607 - User-validated query cache (exact match + semantic match)
- This is for **query caching**, not **ticket cache lookup**

**Ticket Cache Artifacts:**
- `backend/utils/ticket_cache_artifacts.py` - Transforms tickets → RAG artifacts
- `Scraper/export_cache_artifacts.py` - Exports cache-eligible tickets to JSONL
- **BUT:** No code found that **looks up** ticket cache during RAG queries

**Integration Point Needed:** Insert ticket cache lookup in `backend/orchestrator.py::orchestrate_query()` BEFORE RAG retrieval (around line 4608)

### Q3.4: Resolution Steps Storage

**Table:** `ticket_judgements.resolution_steps_json` (JSONB column)

**Schema:** Array of strings
```json
["Step 1: Check power connection", "Step 2: Verify firmware version", ...]
```

**Extraction Function:** `backend/utils/ticket_cache_artifacts.py::extract_steps()` (line 58)

**Source:** `raw_response_json.resolution.steps` (from LLM response)

**Fallback Paths:**
- `raw_response_json.resolution.steps` (primary)
- `raw_response_json.resolution_steps` (backward compatibility)
- `raw_response_json.steps` (backward compatibility)

**Usage:** Used in `build_ticket_cache_artifact()` to construct ticket cache artifact text

---

## 4) Machine Identification & Assignment

### Q4.1: Canonical Machine Identifier

**Table:** `machine_models` (not ticket-specific)

**Model:** `backend/utils/db.py::MachineModel` (line 291)

**Primary Key:** `id` (Integer, auto-increment)

**Canonical Identifier:** `machine_model_id` (Integer) - Used in foreign keys and assignments

**Name Field:** `name` (String(255), unique) - Human-readable name (e.g., "EZCut 330", "DuraFlex 500")

**Machine Kinds:** Enum `MachineKind` (line 283)
- `PRINT_ENGINE` = "Print Engine"
- `BLADE_CUTTER` = "Blade Cutter"
- `LASER_CUTTER` = "Laser Cutter"
- `PRINTER` = "Printer"

**Ticket Assignment Table:** `ticket_machine_model_assignment`
- `machine_model_ids` (JSONB) - Array of integer IDs
- `status` (String) - `"assigned"`, `"ambiguous"`, or `"unassigned"`
- `confidence` (Float) - Assignment confidence (0.0-1.0)

### Q4.2: Machine Model Assignment to Tickets

**Matching Algorithm:**
- **File:** `Scraper/utils/machine_model_matcher.py`
- **File:** `Scraper/scripts/backfill_ticket_machine_models.py` (line 131)

**Process:**
1. Extract machine mentions from ticket text (regex + keyword matching)
2. Score matches against `machine_models` table
3. Store matches in `ticket_machine_model_matches` (one row per match)
4. Determine final assignment in `ticket_machine_model_assignment` (one row per ticket)

**Assignment Logic:** `Scraper/utils/machine_model_matcher.py::determine_assignment()` (line 189)
- Single match → `status="assigned"`, `confidence=1.0`
- Multiple matches → `status="ambiguous"`, `confidence=0.8`
- No matches → `status="unassigned"`, `confidence=0.0`

**Match Sources:** `match_source` field values:
- `"subject"` - Found in ticket subject
- `"body"` - Found in ticket body
- `"manual"` - Manually assigned

**Manual Override:**
- `backend/utils/tickets_admin.py::update_ticket()` (line 626)
- Accepts `machine_model_names` parameter
- Updates `ticket_machine_model_assignment` table
- Sets `method="manual_edit"`

### Q4.3: No Machine Match Fallback

**Status:** `"unassigned"` in `ticket_machine_model_assignment.status`

**Fallback Behavior:** **NOT FOUND** - No explicit fallback logic found

**In RAG Pipeline:**
- `backend/orchestrator.py::orchestrate_query()` accepts `user_machine_models` parameter
- If no machine selected, uses all available machines or `GENERAL` (see `MACHINE_CONFIRMATION_REMOVAL.md`)

**Recommendation:** Consider defaulting to `GENERAL` or all machines when `status="unassigned"`

---

## 5) Runtime Integration Point: Where AI Answers Come From

### Q5.1: Frontend → Backend Query Endpoint

**Frontend Route:** `frontend/app/api/query/route.ts` (line 47)

**Backend Endpoint:** `POST /query`

**Handler:** `backend/api.py::query_knowledge_base()` (line 2999)

**Request Schema:**
```typescript
{
  query: string,
  top_k?: number,
  alpha?: number,
  dynamic_windowing?: boolean,
  machine_confirmation?: boolean,
  selected_machine?: string,
  conversation_id?: string
}
```

**Response Schema:** `QueryResponse` (defined in `backend/api.py`)
```python
{
  answer: str,
  reasoning: str,
  sources: List[Source],
  confidence: float,
  intent: Intent,
  conversation_id: str
}
```

### Q5.2: Ticket Cache Lookup Insertion Point

**File:** `backend/orchestrator.py::orchestrate_query()` (line 4535)

**Current Flow:**
1. Intent classification (line ~4560)
2. Query rewriting (line ~4570)
3. Machine matching (line ~4581)
4. **User-validated cache lookup** (line 4589-4607) ← **INSERT TICKET CACHE HERE**
5. Hybrid retrieval (dense + BM25) (line ~4610)
6. Re-ranking (line ~4650)
7. Response generation (line ~4976)

**Recommended Insertion:** After machine matching (line ~4587), before user-validated cache lookup

**Lookup Logic Needed:**
1. Query ticket cache artifacts (filter by `cache_eligible=true`, `review_status="approved"`)
2. Semantic similarity match (using query embedding)
3. If match found above threshold → return cached answer
4. Otherwise → continue to RAG pipeline

### Q5.3: No RAG / Models Disabled Mode

**Env Flag:** `DISABLE_RAG=true`

**File:** `backend/config/env.py::_load_fast_dev_config()` (line 108)

**Enforcement:**
- `backend/api.py::require_rag_enabled()` (line 159) - Dependency function
- `backend/api.py::query_knowledge_base()` (line 3028) - Checks `settings.DISABLE_RAG`
- Returns HTTP 503 if RAG disabled

**Other Dev Flags:**
- `DEV_SKIP_DB=true` - Skip database initialization
- `DEV_SKIP_GCS=true` - Skip GCS smoke checks
- `DEV_FAST=true` - Enables all skip flags

**RAG Status Endpoint:** `GET /rag/status` - Returns RAG initialization status

---

## 6) Safety / Guardrails for Auto-Actions

### Q6.1: Draft vs Auto-Send

**NOT FOUND** - No draft/auto-send workflow found.

**Search Terms Checked:**
- ❌ "draft" - NOT FOUND
- ❌ "autopost" - NOT FOUND
- ❌ "auto_reply" - NOT FOUND
- ❌ "auto-send" - NOT FOUND
- ✅ "send" - Found in email utils (`backend/utils/email_utils.py`) - Only for invite emails
- ✅ "post" - Found in HTTP POST endpoints (not ticket posting)

**Conclusion:** System currently has **no auto-reply or ticket posting functionality**

### Q6.2: Policy Checks

**Confidence Thresholds:**
- **File:** `Scraper/judge_ticket_cache_eligibility.py::determine_review_status()` (line 1917)
- `approve_min_confidence`: 0.90 (default)
- `reject_min_confidence`: 0.60 (default)
- `unclear_reject_min_confidence`: 0.80 (default)

**Blockers List:**
- Stored in `ticket_judgements.blockers_json`
- Examples: `["missing_confirmation", "missing_resolution_steps", "onsite_required"]`

**Review Status Logic:**
- `"approved"` → All criteria met (high confidence + confirmed + steps)
- `"rejected"` → Hard-blocked or non-cacheable outcome
- `"needs_review"` → Borderline (requires manual review)

**Manual Review Table:** `ticket_manual_reviews`
- `manual_status`: `"approved"` or `"rejected"`
- `reviewer`: Reviewer email/name
- `reviewed_at`: Timestamp

**Required Confirmations:** **NOT FOUND** - No confirmation workflow for auto-actions

### Q6.3: Human Review Workflow

**Table:** `ticket_manual_reviews`

**Model:** `backend/utils/db.py::TicketManualReview` (line 518)

**Fields:**
- `manual_status` - `"approved"` or `"rejected"` (CHECK constraint)
- `manual_reason` - Text reason
- `manual_confirmation_quote` - Quote from ticket confirming resolution
- `reviewer` - Reviewer identifier
- `reviewed_at` - Timestamp

**UI:** `frontend/app/admin/tickets/page.tsx` - Admin ticket management page
- Shows `review_status` column
- Allows editing `cache_eligible`, `confidence`, `review_status`
- **BUT:** No explicit "Review" workflow UI found

**Manual Review Script:** `Scraper/manual_review.py` - Interactive CLI tool for reviewing tickets

**API Endpoint:** `PATCH /admin/tickets/{ticketId}` - Allows updating `review_status` manually

---

## 7) Auth, Permissions, and Secrets (Production Readiness)

### Q7.1: API Authentication

**Method:** **JWT tokens** (HS256 algorithm)

**Files:**
- `backend/security.py` - JWT creation/validation
- `backend/config/auth.py` - Auth configuration

**Token Sources (Priority Order):**
1. `X-User-Token` header (preferred, used by frontend API routes)
2. Cookie (`access_token` by default, configurable via `AUTH_COOKIE_NAME`)
3. `Authorization: Bearer <token>` header (fallback)

**Token Creation:** `backend/security.py::create_access_token()` (line 20)
- Payload: `{"email": str, "role": str, "exp": datetime}`
- Secret: `settings.JWT_SECRET_KEY`
- Expiration: `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (default: 60 minutes)

**Token Validation:** `backend/security.py::get_current_user_from_token()` (line 63)

**Enforcement:**
- `backend/api.py::query_knowledge_base()` - Reads token from `X-User-Token` header (line 3171)
- `backend/routes/admin_routes.py` - Uses `Depends(get_current_user_from_token)` on admin endpoints

**Frontend Auth:** Next.js middleware (`frontend/middleware.ts`) - Validates JWT, sets `X-User-Token` header

### Q7.2: Required Secrets

**Production Secrets (from `backend/config/env.py`):**

| Env Var | Required | Purpose | Source |
|---------|----------|---------|--------|
| `DATABASE_URL` | ✅ Yes | PostgreSQL connection string | Google Secret Manager |
| `JWT_SECRET_KEY` | ✅ Yes | JWT signing secret (min 32 chars) | Google Secret Manager |
| `FRONTEND_SESSION_SECRET` | ✅ Yes | Next.js session secret | Google Secret Manager |
| `ANTHROPIC_API_KEY` | ⚠️ Optional | Claude API key (for LLM synthesis) | Google Secret Manager |
| `CORS_ALLOWED_ORIGINS` | ✅ Yes (prod) | CORS whitelist (comma-separated) | Env var |
| `DOCS_GCS_BUCKET` | ✅ Yes (prod) | GCS bucket for documents | Env var |
| `RAG_INDEX_GCS_BUCKET` | ⚠️ Optional | GCS bucket for RAG index | Env var (default: `arrow-rag-support-prod-rag`) |
| `SMTP_HOST` | ⚠️ Optional | SMTP server for invite emails | Google Secret Manager |
| `SMTP_PORT` | ⚠️ Optional | SMTP port | Google Secret Manager |
| `SMTP_USERNAME` | ⚠️ Optional | SMTP username | Google Secret Manager |
| `SMTP_PASSWORD` | ⚠️ Optional | SMTP password | Google Secret Manager |

**Zendesk Credentials (Scraper only):**
- `ZENDESK_EMAIL` - Zendesk account email (`Scraper/.env`)
- `ZENDESK_PASSWORD` - Zendesk account password (`Scraper/.env`)

### Q7.3: IP Allowlists, Rate Limiting, CORS

**Rate Limiting:**
- **File:** `backend/api.py` (lines 722-727)
- **Library:** `slowapi` (in-memory storage)
- **Config:** `backend/config/env.py::_load_rate_limit_config()` (line 408)

**Rate Limits:**
- `RATE_LIMIT_GLOBAL`: `100/minute` (default)
- `RATE_LIMIT_LOGIN`: `5/minute` (default)
- `RATE_LIMIT_QUERY`: `10/minute` (default)
- `/health` endpoint: **NOT rate limited**

**Rate Limit Key:** Client IP address (uses `X-Forwarded-For` header if available)

**CORS:**
- **File:** `backend/api.py` (lines 714-720)
- **Config:** `backend/config/env.py::_load_cors_origins()` (line 370)
- **Prod:** Requires `CORS_ALLOWED_ORIGINS` (comma-separated list, no wildcards)
- **Dev:** Defaults to `["http://localhost:3000", "http://127.0.0.1:3000"]`

**IP Allowlists:** **NOT FOUND** - No IP allowlist configuration found

**Request Size Limits:** **NOT FOUND** - No explicit request size limits found (FastAPI default applies)

---

## 8) Observability & Debugging

### Q8.1: Logging Configuration

**File:** `backend/logging_config.py`

**Format:**
- **Prod:** JSON logs (structlog with `JSONRenderer`)
- **Dev:** Pretty console output with colors (`ConsoleRenderer`)

**Level:** `INFO` (default)

**Structured Fields:**
- `request_id` - Request correlation ID
- `user_id` - User email (from JWT)
- `role` - User role (from JWT)
- `timestamp` - ISO format

**Middleware:** `backend/middleware/logging_middleware.py::LoggingMiddleware`
- Captures all HTTP requests/responses
- Logs request path, method, status code, duration

**Debug Logs:**
- Set log level via `logging.basicConfig(level=logging.DEBUG)`
- Or use `structlog.configure()` with custom processors

**File Logging:** Uvicorn logs to file (configurable via `API_LOG_FILE_PATH` env var, default: `api.log`)

### Q8.2: Metrics/Tracing

**NOT FOUND** - No OpenTelemetry, Prometheus, or other metrics/tracing found.

**Audit Logs:**
- **Table:** `audit_logs`
- **Model:** `backend/utils/db.py::AuditLog` (line 266)
- **Fields:** `event`, `level`, `user_id`, `role`, `ip_address`, `event_metadata`, `request_id`
- **Function:** `backend/utils/audit_log.py::audit_log()` (line 22)

**Query History:**
- **Table:** `query_history`
- **Model:** `backend/utils/db.py::QueryHistory` (line 196)
- **Fields:** `query_text`, `answer_text`, `response_time_ms`, `token_input`, `token_output`, `cost_usd`

### Q8.3: Request IDs

**Request ID Generation:**
- **File:** `backend/middleware/logging_middleware.py` (line 24)
- **Format:** UUID v4
- **Header:** `X-Request-ID` (if not present, generates new)

**Correlation:**
- Stored in `logging_context` (contextvars)
- Included in all structlog logs via `merge_contextvars` processor
- Stored in `audit_logs.request_id` column

**Query Correlation:**
- `query_history` table does NOT have `request_id` column
- Uses `conversation_id` for grouping queries in conversations

---

## 9) Integration Design Decisions

### Q9.1: Answer Output Format

**Response Model:** `QueryResponse` (defined in `backend/api.py`)

**Fields:**
```python
{
  "answer": str,                    # Main answer text
  "reasoning": str,                 # Reasoning summary
  "sources": List[Source],          # Source citations
  "confidence": float,              # Confidence score (0.0-1.0)
  "intent": Intent,                 # Intent classification
  "conversation_id": str            # Conversation grouping ID
}
```

**Source Format:**
```python
{
  "id": str,                        # Source ID (e.g., "doc_123_chunk_456")
  "name": str,                      # Document name
  "pages": List[int],               # Page numbers
  "content_type": str,              # "text", "table", "image", etc.
  "relevance_score": float          # Relevance score
}
```

**Intent Format:**
```python
{
  "intent_type": str,               # "definition", "lookup", "troubleshooting", etc.
  "confidence": float,              # Intent confidence
  "keywords": List[str]             # Extracted keywords
}
```

**File:** `backend/api.py` (lines 2800-2900, QueryResponse model definition)

### Q9.2: Citations/Evidence Assembly

**File:** `backend/orchestrator.py::ResponseGenerator.generate_structured_response()` (line ~4800)

**Citation Format:**
- Inline citations: `[1]`, `[2]`, etc. in answer text
- Source summary: List of sources with page numbers at end

**Ticket Citations (Not Implemented):**
- **NOT FOUND** - No code found that cites prior tickets
- **Recommendation:** Add ticket citation format similar to document citations
- **Format Suggestion:** `[Ticket #12345]` or `[ticket:12345]`

**Evidence Assembly:**
- Sources come from hybrid retrieval (dense + BM25)
- Re-ranked by cross-encoder
- Filtered by machine models (if user has machines assigned)

### Q9.3: Known Failure Modes

**DB Down:**
- **File:** `backend/api.py::ensure_db_manager_initialized()` (line ~900)
- **Behavior:** Retries with exponential backoff (max 3 attempts)
- **Fallback:** Returns HTTP 503 if DB unavailable
- **Non-RAG Endpoints:** Still work (RAG endpoints require DB for user context)

**Model Disabled:**
- **Env:** `DISABLE_RAG=true`
- **Behavior:** Returns HTTP 503 with `{"detail": "RAG disabled (DISABLE_RAG=true)"}`
- **File:** `backend/api.py::get_rag_disabled_response()` (line 125)

**Missing Machine:**
- **Behavior:** Uses all available machines or `GENERAL` (see `MACHINE_CONFIRMATION_REMOVAL.md`)
- **File:** `backend/orchestrator.py::orchestrate_query()` (line 4544)
- **Fallback:** No machine filtering applied (searches all documents)

**Index Not Found:**
- **File:** `backend/rag/index_manager.py`
- **Behavior:** Downloads from GCS on startup (Cloud Run)
- **Fallback:** Returns HTTP 503 if index unavailable

**Anthropic API Failure:**
- **Behavior:** Falls back to chunk-based answer (no LLM synthesis)
- **File:** `backend/orchestrator.py::ResponseGenerator._build_answer()` (line ~2096)
- **Fallback:** `_build_chunk_based_answer()` - Returns raw chunks with "According to..." format

---

## 10) CI / Scripts / Operational Workflow

### Q10.1: CI Scripts

**File:** `.github/workflows/ci.yml`

**Jobs:**
1. `validate-secrets` - Validates required secrets are set
2. `frontend-build` - Builds Next.js frontend
3. `docker-build` - Builds and pushes Docker images
4. `gcp-auth-check` - Validates GCP authentication
5. `deploy-backend` - Deploys backend to Cloud Run (includes DB migrations)
6. `deploy-frontend` - Deploys frontend to Cloud Run
7. `post-deploy-verification` - Verifies cost control settings

**Database Migrations:**
- **File:** `.github/workflows/ci.yml` (lines 557-604)
- **Method:** Cloud Run Job (`arrow-rag-backend-migrate`)
- **Command:** `python -m alembic -c backend/migrations/alembic.ini upgrade head`

**Tests:**
- **NOT FOUND** - No pytest/lint steps in CI workflow
- **Test Directory:** `backend/tests/` exists but not run in CI

### Q10.2: Validation Scripts

**Found Scripts:**
- ✅ `scripts/validate_tickets_pipeline.py` - Validates ticket migrations + parity + smoke tests
- ✅ `backend/scripts/smoke_ticket_reads.py` - Smoke test for ticket reads from Postgres
- ✅ `backend/scripts/verify_tickets_parity.py` - Verifies parity between SQLite and Postgres
- ✅ `backend/scripts/migrate_tickets_sqlite_to_postgres.py` - Migration script

**Other Validation Scripts:**
- `Scraper/validate_cache_pipeline.py` - Validates cache eligibility judgments
- `Scraper/verify_raw_response_schema.py` - Verifies raw_response_json schema consistency
- `backend/scripts/verify_tickets_parity.py` - Parity verification

### Q10.3: Staging "Draft-Only" Mode

**NOT FOUND** - No draft-only mode configuration found.

**Recommendation:** Add env flag:
- `TICKET_CACHE_MODE` - `"draft"` (return answers but don't auto-post) or `"live"` (auto-post if confidence high)

**Current State:** System has no auto-posting, so "draft-only" is the default

---

## Gaps / Unknowns

### Critical Gaps

1. **❌ Ticket Cache Lookup Not Implemented**
   - No code found that looks up ticket cache during RAG queries
   - Need to implement semantic similarity matching against cache-eligible tickets
   - Insertion point: `backend/orchestrator.py::orchestrate_query()` before RAG retrieval

2. **❌ No Outbound Zendesk API Integration**
   - System only scrapes tickets (read-only)
   - No code found for posting comments, updating status, or tagging tickets
   - Need to implement Zendesk API client for auto-replies

3. **❌ No Draft/Auto-Send Workflow**
   - No draft mode or confirmation workflow found
   - Need to implement draft storage and approval workflow

4. **❌ No Ticket Citation Format**
   - Citations only reference PDF documents
   - Need to add ticket citation format (e.g., `[Ticket #12345]`)

### Medium Priority Gaps

5. **⚠️ No Metrics/Tracing**
   - No OpenTelemetry or Prometheus integration
   - Only audit logs and query history for observability

6. **⚠️ No IP Allowlists**
   - Only rate limiting and CORS for security
   - No IP-based access control

7. **⚠️ No Request Size Limits**
   - Relies on FastAPI defaults
   - May need explicit limits for ticket ingestion endpoints

### Low Priority / Nice-to-Have

8. **📝 No Test Coverage in CI**
   - Tests exist but not run in CI pipeline
   - Should add pytest step to CI

9. **📝 No Read Replica Support**
   - Single database connection
   - May need read replicas for scaling

10. **📝 No Explicit Fallback for Unassigned Machines**
    - System handles it gracefully but no explicit fallback logic documented

---

## Next Steps for Integration

1. **Implement Ticket Cache Lookup**
   - Add semantic similarity matching in `backend/orchestrator.py`
   - Query `ticket_judgements` filtered by `cache_eligible=true`, `review_status="approved"`
   - Return cached answer if similarity > threshold

2. **Add Ticket Citation Format**
   - Extend `Source` model to include `source_type` ("document" vs "ticket")
   - Format ticket citations as `[Ticket #12345]` in answer text

3. **Implement Draft/Auto-Send Workflow**
   - Add `draft_responses` table for storing draft answers
   - Add confirmation endpoint for approving drafts
   - Add auto-send logic (if confidence > threshold and approved)

4. **Add Outbound Zendesk API Integration**
   - Implement Zendesk API client (using `zendesk` Python library)
   - Add endpoints for posting comments and updating ticket status
   - Add webhook handler for ticket events (optional)

5. **Add Integration Tests**
   - Test ticket cache lookup end-to-end
   - Test draft workflow
   - Test Zendesk API integration (with mock)

---

**End of Document**
