# Alembic Migration Commands for Ticket Migration

## Section A: What I Found

### Alembic Configuration Files

1. **`backend/migrations/alembic.ini`** - Alembic configuration file
   - Location: `c:\Users\ethan\ArrowSystems\backend\migrations\alembic.ini`
   - `script_location = backend/migrations` (relative path, expects to be run from repo root)
   - `prepend_sys_path = .` (adds repo root to Python path)
   - `sqlalchemy.url = ` (empty - set dynamically in env.py)

2. **`backend/migrations/env.py`** - Alembic environment setup
   - Location: `c:\Users\ethan\ArrowSystems\backend\migrations\env.py`
   - Loads `DATABASE_URL` from `backend.utils.db` → `backend.config.env.settings`
   - Sets SQLAlchemy URL dynamically: `config.set_main_option("sqlalchemy.url", DATABASE_URL)`

3. **Migration Files**
   - Location: `c:\Users\ethan\ArrowSystems\backend\migrations\versions\`
   - Migration `011_ticket_tables_postgres.py` exists and is discoverable
   - Revision chain verified: `010_document_machine_models_m2m` → `011_ticket_tables_postgres` (head)

### Environment Variable Loading

- **DATABASE_URL** is required and loaded from:
  - Environment variable `DATABASE_URL` (takes precedence)
  - Or from `.env` file in `backend/.env` (via python-dotenv)
- **Settings module** (`backend/config/env.py`) loads DATABASE_URL in `_load_secrets()`
- **No automatic .env loading** in Alembic - must be loaded manually or set as env var

### Migration Chain Verification

✅ Migration `011_ticket_tables_postgres` is in the chain:
```
010_document_machine_models_m2m -> 011_ticket_tables_postgres (head)
009_add_printer_machine_kind -> 010_document_machine_models_m2m
...
```

## Section B: Exact Commands to Run

### Prerequisites: Set DATABASE_URL

**Option 1: Load from .env file (recommended for local dev)**

```bash
# From repo root (C:\Users\ethan\ArrowSystems)
# Load .env file before running Alembic
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini upgrade head')"
```

**Option 2: Set DATABASE_URL as environment variable (Windows Git Bash)**

```bash
# From repo root
export DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
python -m alembic -c backend/migrations/alembic.ini upgrade head
```

**Option 3: Set DATABASE_URL inline (Windows PowerShell)**

```powershell
# From repo root
$env:DATABASE_URL="postgresql://user:pass@localhost:5432/dbname"
python -m alembic -c backend/migrations/alembic.ini upgrade head
```

### Step 1: Verify Current Migration Status

```bash
# From repo root (C:\Users\ethan\ArrowSystems)
# Load .env and check current migration
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini current')"
```

**Expected output if migrations are up-to-date:**
```
011_ticket_tables_postgres (head)
```

**Expected output if migration needed:**
```
010_document_machine_models_m2m
```

### Step 2: Run Migrations

```bash
# From repo root (C:\Users\ethan\ArrowSystems)
# Load .env and run migrations
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini upgrade head')"
```

**Alternative (if DATABASE_URL is already set):**

```bash
# From repo root
python -m alembic -c backend/migrations/alembic.ini upgrade head
```

### Step 3: Verify Migration Applied

```bash
# From repo root
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini current')"
```

**Expected output:**
```
011_ticket_tables_postgres (head)
```

### Step 4: View Migration History

```bash
# From repo root
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini history')"
```

**Expected output (last 5):**
```
010_document_machine_models_m2m -> 011_ticket_tables_postgres (head)
009_add_printer_machine_kind -> 010_document_machine_models_m2m
008_add_machine_kind -> 009_add_printer_machine_kind
007_add_auth_tokens -> 008_add_machine_kind
006_add_language_fields -> 007_add_auth_tokens
```

## Section C: Guardrails & Verification

### Verify DATABASE_URL Before Running

```bash
# From repo root
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); db_url = os.getenv('DATABASE_URL', ''); print('DATABASE_URL:', db_url[:50] + '...' if len(db_url) > 50 else db_url) if db_url else print('ERROR: DATABASE_URL not set')"
```

**Expected:** Should show your Postgres connection string (first 50 chars)

### Verify You're Pointing at Correct Database

```bash
# From repo root
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); from sqlalchemy import create_engine, text; engine = create_engine(os.getenv('DATABASE_URL')); conn = engine.connect(); result = conn.execute(text('SELECT current_database(), current_user')); row = result.fetchone(); print(f'Database: {row[0]}, User: {row[1]}'); conn.close()"
```

**Expected:** Should show your intended database name and user

### Check if Ticket Tables Already Exist

```bash
# From repo root
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); from sqlalchemy import create_engine, text, inspect; engine = create_engine(os.getenv('DATABASE_URL')); inspector = inspect(engine); tables = inspector.get_table_names(); ticket_tables = [t for t in tables if 'ticket' in t.lower()]; print('Ticket tables found:', ticket_tables if ticket_tables else 'None (migration needed)')"
```

**Expected:** Should show `['tickets_index', 'tickets_detail', ...]` if migration already ran, or `None (migration needed)` if not

## Section D: End-to-End Checklist (No Destructive Actions)

### Pre-Migration Checklist

- [ ] **Verify DATABASE_URL is set**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); print('DATABASE_URL:', 'SET' if os.getenv('DATABASE_URL') else 'NOT SET')"
  ```

- [ ] **Verify database connection works**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); from sqlalchemy import create_engine, text; engine = create_engine(os.getenv('DATABASE_URL')); conn = engine.connect(); print('Connection OK'); conn.close()"
  ```

- [ ] **Check current migration status**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini current')"
  ```

- [ ] **Verify migration 011 exists and is discoverable**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini history')"
  ```
  Should show `011_ticket_tables_postgres` in the chain

### Migration Execution

- [ ] **Run migrations**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini upgrade head')"
  ```

- [ ] **Verify migration applied**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini current')"
  ```
  Should show `011_ticket_tables_postgres (head)`

- [ ] **Verify tables created**
  ```bash
  python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); from sqlalchemy import create_engine, inspect; engine = create_engine(os.getenv('DATABASE_URL')); inspector = inspect(engine); tables = [t for t in inspector.get_table_names() if 'ticket' in t.lower()]; print('Created tables:', sorted(tables))"
  ```
  Should show all 9 ticket tables

### Post-Migration: Backfill & Verify (Next Steps)

After migrations succeed, you will run:

1. **Dry-run backfill** (no changes):
   ```bash
   python -m backend.scripts.migrate_tickets_sqlite_to_postgres --dry-run --sqlite-path Scraper/data/tickets.db
   ```

2. **Real backfill** (after dry-run passes):
   ```bash
   python -m backend.scripts.migrate_tickets_sqlite_to_postgres --sqlite-path Scraper/data/tickets.db
   ```

3. **Verify parity**:
   ```bash
   python -m backend.scripts.verify_tickets_parity --sqlite-path Scraper/data/tickets.db --sample 50
   ```

## Section E: If Something is Missing

### Issue: DATABASE_URL not found

**Fix:** Ensure `backend/.env` exists and contains:
```
DATABASE_URL=postgresql://user:pass@host:port/dbname
```

**Or set as environment variable:**
```bash
export DATABASE_URL="postgresql://user:pass@host:port/dbname"
```

### Issue: Alembic can't find migration files

**Fix:** The `script_location` in `alembic.ini` is `backend/migrations`, which is correct when running from repo root with `-c backend/migrations/alembic.ini`.

**Verify working directory:**
```bash
# Should be in repo root
pwd
# Should show: /c/Users/ethan/ArrowSystems
```

### Issue: Import errors when running Alembic

**Fix:** The `prepend_sys_path = .` in `alembic.ini` adds repo root to Python path, which should allow imports. If issues persist:

```bash
# Set PYTHONPATH explicitly
export PYTHONPATH="C:/Users/ethan/ArrowSystems"
python -m alembic -c backend/migrations/alembic.ini upgrade head
```

### Issue: Migration 011 not found

**Verify file exists:**
```bash
ls backend/migrations/versions/011_ticket_tables_postgres.py
```

**Check revision ID matches:**
```bash
grep "revision = " backend/migrations/versions/011_ticket_tables_postgres.py
# Should show: revision = "011_ticket_tables_postgres"
```

**Check down_revision matches:**
```bash
grep "down_revision = " backend/migrations/versions/011_ticket_tables_postgres.py
# Should show: down_revision = "010_document_machine_models_m2m"
```

## Quick Reference: One-Liner Commands

**Check status:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini current')"
```

**Run migrations:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini upgrade head')"
```

**View history:**
```bash
python -c "from dotenv import load_dotenv; import os; load_dotenv('backend/.env'); os.system('python -m alembic -c backend/migrations/alembic.ini history')"
```
