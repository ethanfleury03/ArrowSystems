# Ticket Scraper Migration: SQLite → Postgres

This document describes the migration of ticket scraper data and pipeline from local SQLite (`Scraper/data/tickets.db`) to GCP Cloud SQL Postgres (the same database used by the backend via `DATABASE_URL`).

## Overview

The migration preserves existing ticket pipeline stages and behavior while moving all ticket data to Postgres for better scalability and integration with the backend application.

### Tables Migrated

- `tickets_index` - Stage 1: Cheap indexing of all tickets
- `tickets_detail` - Stage 2: Detailed conversation JSON for solved tickets
- `ticket_summaries` - Stage 3: Structured problem/solution extraction
- `ticket_judgements` - LLM-based cache eligibility classification
- `ticket_triage` - Cheap model triage stage
- `ticket_manual_reviews` - Manual override layer
- `ticket_machine_model_matches` - Machine model matches (from backfill script)
- `ticket_machine_model_assignment` - Machine model assignments (from backfill script)
- `scrape_runs` - Background scrape job tracking

## Prerequisites

1. **Postgres database** accessible via `DATABASE_URL` environment variable
2. **SQLite database** at `Scraper/data/tickets.db` (or path specified via `TICKETS_DB_PATH`)
3. **Alembic migrations** run (ticket tables must exist in Postgres)
4. **Python dependencies** installed (SQLAlchemy, psycopg2, etc.)

## Migration Steps

**⚠️ IMPORTANT: All commands must be run from the repository root (`C:\Users\ethan\ArrowSystems`), NOT from `backend/`.**

### Step 1: Run Alembic Migrations

Ensure ticket tables exist in Postgres:

**Windows Git Bash (from repo root):**

```bash
# Check current migration status
python scripts/db_migrate.py current

# Run migrations
python scripts/db_migrate.py upgrade head

# Verify ticket tables were created
python scripts/tickets_migration_check.py
```

This will create all ticket tables via migration `011_ticket_tables_postgres`.

**What the helper script does:**
- Loads `backend/.env` if present (or uses `DATABASE_URL` env var)
- Validates `DATABASE_URL` is set
- Shows target database info (database name, user)
- Runs Alembic with correct config file (`backend/migrations/alembic.ini`)

### Step 2: Dry-Run Backfill

Test the migration without committing changes:

**Windows Git Bash (from repo root):**

```bash
python -m backend.scripts.migrate_tickets_sqlite_to_postgres \
    --dry-run \
    --sqlite-path Scraper/data/tickets.db
```

Review the output to ensure no errors. The script will show:
- Number of rows found per table
- Number of rows that would be migrated
- Any errors encountered

### Step 3: Run Real Backfill

Once dry-run passes, run the actual migration:

**Windows Git Bash (from repo root):**

```bash
python -m backend.scripts.migrate_tickets_sqlite_to_postgres \
    --sqlite-path Scraper/data/tickets.db
```

**Note:** The script is idempotent - safe to rerun. It uses `ON CONFLICT` UPSERT logic.

### Step 4: Verify Parity

Compare SQLite and Postgres to ensure data matches:

**Windows Git Bash (from repo root):**

```bash
python -m backend.scripts.verify_tickets_parity \
    --sqlite-path Scraper/data/tickets.db \
    --sample 50
```

The script will:
- Compare row counts per table
- Sample random tickets and compare key fields
- Compare JSON fields by hash
- Output `PASS` or `FAIL` with details

### Step 5: Cutover to Postgres

Once verification passes, switch the application to use Postgres:

#### In Cloud Run / Production

Set environment variable:

```bash
TICKETS_STORAGE_BACKEND=postgres
```

The application will automatically:
- Use Postgres for all ticket operations
- Fail fast if `TICKETS_STORAGE_BACKEND=sqlite` is detected (SQLite not available in containers)

#### In Local Development

For local testing with Postgres:

```bash
export TICKETS_STORAGE_BACKEND=postgres
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
```

For local testing with SQLite (default):

```bash
# No env var needed, defaults to sqlite
# Ensure Scraper/data/tickets.db exists
```

### Step 6: Verify Application Works

1. **Start backend** with `TICKETS_STORAGE_BACKEND=postgres`
2. **Visit Admin Tickets page** (`/admin/tickets`)
3. **Verify ticket counts** match expectations
4. **Test scrape functionality** (should write to Postgres)
5. **Verify ticket updates** work correctly

## Configuration

### Environment Variables

- `TICKETS_STORAGE_BACKEND`: `postgres` or `sqlite` (default: `sqlite`)
- `DATABASE_URL`: Postgres connection string (required when using `postgres`)
- `TICKETS_DB_PATH`: Path to SQLite database (optional, for `sqlite` backend)

### Backend Selection Logic

1. **Cloud Run / Production**: Always uses `postgres` (enforced)
2. **Local Development**: Defaults to `sqlite`, can override with env var

## Rollback Plan

If issues are discovered after cutover:

### Option 1: Revert Environment Variable

```bash
# In Cloud Run, remove or change:
TICKETS_STORAGE_BACKEND=sqlite  # or remove entirely
```

The application will fall back to SQLite (if available locally).

### Option 2: Restore from Backup

If Postgres data is corrupted:

1. Restore Postgres from backup (if available)
2. Or re-run backfill script from SQLite

### Option 3: Point Back to SQLite

1. Ensure `Scraper/data/tickets.db` is available
2. Set `TICKETS_STORAGE_BACKEND=sqlite` (or remove env var)
3. Restart application

## Cloud Run Guardrails

The application includes safety checks:

1. **Enforced Postgres in Cloud Run**: If `K_SERVICE` or `GAE_ENV` is set, the application requires `TICKETS_STORAGE_BACKEND=postgres`
2. **No SQLite in Containers**: Attempts to use SQLite in Cloud Run will fail fast with a clear error message
3. **Connection Validation**: Postgres connection is validated at startup

## Windows Git Bash Quick Reference

All commands below assume you are in the repository root (`C:\Users\ethan\ArrowSystems`).

### Verify DATABASE_URL is Loaded

```bash
python scripts/db_migrate.py current
```

If this fails with "DATABASE_URL not found", ensure `backend/.env` exists with:
```
DATABASE_URL=postgresql://user:pass@host:port/dbname
```

### Check Current Migration Status

```bash
python scripts/db_migrate.py current
```

Expected output if up-to-date: `011_ticket_tables_postgres (head)`

### View Migration History

```bash
python scripts/db_migrate.py history
```

Or show last 5 migrations:
```bash
python scripts/db_migrate.py history --last 5
```

### Run Migrations

```bash
python scripts/db_migrate.py upgrade head
```

### Verify Ticket Tables Exist

```bash
python scripts/tickets_migration_check.py
```

This will list all ticket-related tables and confirm all 9 expected tables are present.

## Troubleshooting

### Migration Script Errors

**Error: "Table not found in Postgres"**
- Solution: Run Alembic migrations first: `python scripts/db_migrate.py upgrade head`

**Error: "No config file 'alembic.ini' found"**
- Solution: You're running from the wrong directory. Always run from repo root (`C:\Users\ethan\ArrowSystems`), not from `backend/`.

**Error: "JSON parsing failed"**
- Solution: Check SQLite data integrity. Some JSON fields may be malformed.

**Error: "Connection refused"**
- Solution: Verify `DATABASE_URL` is correct and Postgres is accessible
- Check if Cloud SQL Proxy is running (if using Cloud SQL): `cloud-sql-proxy.exe ...`

### Verification Script Errors

**Error: "Row count mismatch"**
- Solution: Re-run backfill script. Check for errors during migration.

**Error: "JSON hash mismatch"**
- Solution: Review the specific ticket differences. May be due to JSON normalization differences.

### Application Errors

**Error: "TICKETS_STORAGE_BACKEND must be 'postgres' in Cloud Run"**
- Solution: Set `TICKETS_STORAGE_BACKEND=postgres` in Cloud Run environment variables

**Error: "DATABASE_URL environment variable is required"**
- Solution: Set `DATABASE_URL` when using Postgres backend

## Post-Migration Cleanup

**⚠️ DO NOT DELETE SQLite DATA UNTIL VERIFIED**

After successful migration and verification:

1. **Wait 1-2 weeks** of production use with Postgres
2. **Monitor for issues** (missing tickets, data corruption, etc.)
3. **Verify new scrapes** write to Postgres correctly
4. **Only then** consider archiving/deleting `Scraper/data/tickets.db`

## Manual Cleanup Checklist

Before deleting SQLite data:

- [ ] Migration completed successfully
- [ ] Verification script reports PASS
- [ ] Admin Tickets UI shows correct counts
- [ ] New scrape runs write to Postgres
- [ ] Ticket updates work correctly
- [ ] No production issues for 1-2 weeks
- [ ] Backup of SQLite database created (optional)

## Support

For issues or questions:
1. Check logs: `backend/logs/` and Cloud Run logs
2. Review migration script output
3. Run verification script to identify specific mismatches
4. Check Alembic migration status: `python scripts/db_migrate.py current`
5. Verify ticket tables: `python scripts/tickets_migration_check.py`
