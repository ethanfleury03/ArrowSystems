# Tickets Pipeline Validation

This document describes how to validate the SQLite→Postgres ticket migration pipeline.

## Quick Start

Run the complete validation pipeline:

```bash
# Windows Git Bash
python scripts/validate_tickets_pipeline.py

# With custom sample sizes
python scripts/validate_tickets_pipeline.py --parity-sample 50 --smoke-sample 20
```

## Individual Steps

### 1. Check Migration Status

```bash
python scripts/db_migrate.py current
```

### 2. Ticket Migration (Dry-Run)

```bash
python -m backend.scripts.migrate_tickets_sqlite_to_postgres \
  --dry-run \
  --sqlite-path Scraper/data/tickets.db \
  --orphan-policy skip
```

### 3. Parity Verification

```bash
python -m backend.scripts.verify_tickets_parity \
  --sqlite-path Scraper/data/tickets.db \
  --sample 20
```

### 4. Smoke Test (Ticket Reads)

```bash
python -m backend.scripts.smoke_ticket_reads --sample 10
```

## Runtime Validation

The backend automatically logs database connection info on startup:

```
[DB_INIT] dialect=postgresql host=localhost port=5432 database=mydb user=postgres sqlite_fallback=False
```

This confirms:
- ✅ Postgres is being used (not SQLite)
- ✅ Connection details (host, port, database)
- ✅ No SQLite fallback mode

## Logging

### Migration Script

By default, the migration script uses compact logging (no huge SQL dumps):

```bash
# Compact logging (default)
python -m backend.scripts.migrate_tickets_sqlite_to_postgres --dry-run

# Verbose SQL logging (for debugging)
python -m backend.scripts.migrate_tickets_sqlite_to_postgres --dry-run --debug-sql
```

### Parity Script

The parity script uses tolerance-based timestamp comparison:

```bash
# Default tolerance (1.0s)
python -m backend.scripts.verify_tickets_parity --sqlite-path Scraper/data/tickets.db

# Custom tolerance
python -m backend.scripts.verify_tickets_parity \
  --sqlite-path Scraper/data/tickets.db \
  --timestamp-tolerance-seconds 2.0

# Ignore timestamps completely
python -m backend.scripts.verify_tickets_parity \
  --sqlite-path Scraper/data/tickets.db \
  --ignore-timestamps
```

## CI/CD Integration

Add to your CI pipeline:

```yaml
# Example GitHub Actions
- name: Validate Tickets Pipeline
  run: |
    python scripts/validate_tickets_pipeline.py \
      --parity-sample 20 \
      --smoke-sample 10
```

Or run individual steps:

```bash
# Check migrations
python scripts/db_migrate.py current

# Validate migration (dry-run)
python -m backend.scripts.migrate_tickets_sqlite_to_postgres \
  --dry-run --sqlite-path Scraper/data/tickets.db --orphan-policy skip

# Verify parity
python -m backend.scripts.verify_tickets_parity \
  --sqlite-path Scraper/data/tickets.db --sample 20

# Smoke test
python -m backend.scripts.smoke_ticket_reads --sample 10
```

## Troubleshooting

### Migration Fails

1. Check DATABASE_URL is set correctly:
   ```bash
   echo $DATABASE_URL
   ```

2. Verify SQLite database exists:
   ```bash
   ls -la Scraper/data/tickets.db
   ```

3. Run with verbose logging:
   ```bash
   python -m backend.scripts.migrate_tickets_sqlite_to_postgres \
     --dry-run --debug-sql --verbose
   ```

### Parity Check Fails

1. Check timestamp tolerance:
   ```bash
   python -m backend.scripts.verify_tickets_parity \
     --sqlite-path Scraper/data/tickets.db \
     --timestamp-tolerance-seconds 5.0
   ```

2. Ignore timestamps to check other fields:
   ```bash
   python -m backend.scripts.verify_tickets_parity \
     --sqlite-path Scraper/data/tickets.db \
     --ignore-timestamps
   ```

### Smoke Test Fails

1. Verify Postgres connection:
   ```bash
   python -m backend.scripts.smoke_ticket_reads --sample 1
   ```

2. Check table counts manually:
   ```bash
   psql $DATABASE_URL -c "SELECT COUNT(*) FROM tickets_index;"
   ```

## Files Modified/Created

- `backend/utils/db.py` - Added startup DB logging
- `backend/scripts/smoke_ticket_reads.py` - New smoke test script
- `scripts/validate_tickets_pipeline.py` - New validation pipeline script
- `docs/TICKETS_VALIDATION.md` - This documentation

## Local Cleanup

After successful migration to Cloud SQL, you can clean up local artifacts:

### Quick Cleanup

```bash
# Preview what will be cleaned (dry-run)
python scripts/cleanup_local_tickets_artifacts.py --dry-run

# Apply cleanup (requires clean git repo)
python scripts/cleanup_local_tickets_artifacts.py --apply

# Force apply (skip git status check)
python scripts/cleanup_local_tickets_artifacts.py --apply --force
```

### What Gets Cleaned

**Category A: Deleted (logs, temp outputs)**
- `out/migrate_dryrun.log` and other `out/*.log` files
- `scripts/__pycache__/` (optional)

**Category B: Archived (SQLite databases)**
- `Scraper/data/tickets.db` → `.archive/tickets_migration/YYYYMMDD_HHMMSS/tickets.db`
- Other `.db` files in `Scraper/data/`
- Postgres dumps (if found)

**Category C: Protected (never touched)**
- `backend/scripts/*.py` (migration/verification scripts)
- `scripts/validate_tickets_pipeline.py`
- `docs/TICKETS_VALIDATION.md`
- `backend/migrations/`
- `backend/.env` and other config files

### Archive Location

Archived files are moved to:
```
.archive/tickets_migration/YYYYMMDD_HHMMSS/
```

This preserves SQLite databases as backups while cleaning up the working directory.

### Safety Features

- **Dry-run by default**: Shows plan without making changes
- **Git status check**: Requires clean repo (or `--force`) before applying
- **Idempotent**: Safe to re-run
- **Windows-safe**: Uses `pathlib` for cross-platform paths

## Related Scripts

- `backend/scripts/migrate_tickets_sqlite_to_postgres.py` - Migration script
- `backend/scripts/verify_tickets_parity.py` - Parity verification
- `backend/scripts/smoke_ticket_reads.py` - Smoke test
- `scripts/validate_tickets_pipeline.py` - Validation pipeline
- `scripts/cleanup_local_tickets_artifacts.py` - Cleanup script
- `scripts/db_migrate.py` - Alembic migration wrapper
