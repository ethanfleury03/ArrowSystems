# Database Migrations Guide

This guide explains how to work with database migrations in the ArrowSystems RAG application.

## Overview

The application uses **Alembic** for database schema migrations. Migrations are versioned, tracked, and can be safely applied to both SQLite (development) and PostgreSQL (production) databases.

## Migration System Architecture

### Components

1. **Alembic** - Migration framework
2. **Migration Runner** (`backend/utils/migration_runner.py`) - Dev/prod logic wrapper
3. **Migration Scripts** (`backend/migrations/versions/`) - Versioned schema changes
4. **Startup Integration** - Automatic migration execution in dev mode

### Directory Structure

```
backend/
├── migrations/
│   ├── README.md              # Migration workflow guide
│   ├── alembic.ini            # Alembic configuration
│   ├── env.py                 # Alembic environment (uses get_engine())
│   └── versions/              # Migration scripts
│       ├── 001_initial_schema.py
│       └── 002_schema_fixes.py
└── utils/
    └── migration_runner.py    # Migration execution utility
```

## Development Mode Behavior

**Migrations run automatically on application startup.**

- No manual intervention required
- Migrations are applied before database initialization
- If migration fails, application fails to start (fail-fast)
- All migration activity is logged

## Production Mode Behavior

**Migrations DO NOT run automatically in production.**

- Application checks for pending migrations on startup
- If pending migrations exist → **application fails to start** with clear error message
- Migrations must be run manually before deployment
- This prevents accidental schema changes in production

### Why No Auto-Run in Production?

1. **Safety** - Schema changes can be destructive
2. **Control** - Operations team needs to review and approve migrations
3. **Rollback** - Manual execution allows for backup/rollback procedures
4. **Audit** - Manual execution provides audit trail
5. **Zero-downtime** - Allows for staged migrations in multi-instance deployments

## Creating a New Migration

### Step 1: Make Schema Changes

Update the SQLAlchemy models in `backend/utils/db.py`:

```python
class MyTable(Base):
    __tablename__ = "my_table"
    
    id = Column(Integer, primary_key=True)
    new_column = Column(String(255), nullable=True)  # Add new column
```

### Step 2: Generate Migration Script

```bash
# From project root
cd backend
alembic revision --autogenerate -m "add new_column to my_table"
```

This creates a new file in `backend/migrations/versions/` with a name like:
`003_add_new_column_to_my_table.py`

### Step 3: Review Generated Migration

**Always review the generated migration script!** Alembic's autogenerate is not perfect:

- Check that it correctly detects your changes
- Verify SQL operations are safe
- Ensure it handles existing data correctly
- Test on a copy of production data if possible

### Step 4: Test in Development

1. Start the application in dev mode
2. Migrations run automatically
3. Verify the schema changes are applied correctly
4. Test application functionality

### Step 5: Commit Migration

```bash
git add backend/migrations/versions/003_*.py
git commit -m "Add migration: add new_column to my_table"
```

## Running Migrations

### Development

**Automatic** - No action needed. Migrations run on startup.

### Production

**Manual execution required:**

```bash
# Option 1: Use migration runner
python -m backend.utils.migration_runner upgrade

# Option 2: Use Alembic directly
alembic upgrade head

# Option 3: Check status first
python -m backend.utils.migration_runner status
```

### Migration Commands

```bash
# Check migration status
python -m backend.utils.migration_runner status

# Run pending migrations
python -m backend.utils.migration_runner upgrade

# Check if migrations are pending
python -m backend.utils.migration_runner check
```

## Migration Scripts Included

### 001_initial_schema.py

Creates the initial database schema:
- `users` table
- `query_history` table
- `feedback` table
- `saved_responses` table
- `audit_logs` table
- All primary keys, foreign keys, and basic indexes

### 002_schema_fixes.py

Applies production-readiness fixes:
- Adds `updated_at` columns to `Feedback` and `QueryHistory`
- Adds NOT NULL constraints to `User.name` and `User.password_hash`
- Adds indexes on `query_text` and `answer_text`
- Adds composite indexes for common query patterns
- Sets default values for existing NULL data

## Docker & Cloud Run Integration

### Local Docker Development

Migrations run automatically when containers start (dev mode).

### Production Deployment

**Before deploying to Cloud Run:**

1. **Check migration status:**
   ```bash
   python -m backend.utils.migration_runner status
   ```

2. **Run migrations manually:**
   ```bash
   # Set production DATABASE_URL
   export DATABASE_URL="postgresql://..."
   export ENV=prod
   
   # Run migrations
   python -m backend.utils.migration_runner upgrade
   ```

3. **Deploy application:**
   ```bash
   # Application will verify migrations are up to date
   # If not, deployment will fail with clear error
   ```

### CI/CD Pipeline

Include migration step in your deployment pipeline:

```yaml
# Example GitHub Actions / Cloud Build step
- name: Run Database Migrations
  run: |
    export DATABASE_URL="${{ secrets.DATABASE_URL }}"
    export ENV=prod
    python -m backend.utils.migration_runner upgrade
  env:
    DATABASE_URL: ${{ secrets.DATABASE_URL }}
```

## Migration Safety Guidelines

### DO:

✅ Always test migrations on a copy of production data  
✅ Review generated migration scripts before committing  
✅ Use Alembic `op` functions (not raw SQL) when possible  
✅ Make migrations idempotent where possible  
✅ Include data migration steps if schema changes require it  
✅ Test rollback procedures (`alembic downgrade`)  

### DON'T:

❌ Don't modify existing migration scripts after they've been applied  
❌ Don't skip migration steps  
❌ Don't run migrations directly on production without testing  
❌ Don't use raw SQL that's database-specific (use `op` functions)  
❌ Don't delete migration files from version control  

## Troubleshooting

### Migration Fails on Startup

**Error:** `Database migration failed: ...`

**Solution:**
1. Check the error message for details
2. Verify database connection
3. Check migration script syntax
4. Review Alembic version table: `SELECT * FROM alembic_version;`

### Pending Migrations in Production

**Error:** `Database schema is outdated. Pending migrations detected.`

**Solution:**
```bash
# Run migrations manually
python -m backend.utils.migration_runner upgrade
```

### Schema Drift

**Symptom:** Application models don't match database schema

**Solution:**
1. Check current migration: `alembic current`
2. Check expected migration: `alembic heads`
3. Generate new migration to sync: `alembic revision --autogenerate -m "sync schema"`

### Migration History Issues

**View migration history:**
```bash
alembic history
```

**View current revision:**
```bash
alembic current
```

## SQLite vs PostgreSQL Compatibility

Migrations are designed to work with both databases:

- **SQLite:** Uses batch mode for ALTER TABLE operations
- **PostgreSQL:** Uses standard ALTER TABLE operations
- **CHECK constraints:** Enforced at application level for SQLite (SQLite limitation)
- **JSON columns:** TEXT in SQLite, JSONB in PostgreSQL (handled automatically)

## Migration Version Tracking

Alembic creates an `alembic_version` table to track applied migrations:

```sql
SELECT * FROM alembic_version;
```

This table stores the current revision ID. Never modify this table manually.

## Rollback Procedures

To rollback a migration:

```bash
# Rollback one step
alembic downgrade -1

# Rollback to specific revision
alembic downgrade <revision_id>

# Rollback all migrations (DANGEROUS!)
alembic downgrade base
```

**Warning:** Always backup your database before rolling back migrations.

## Best Practices

1. **One logical change per migration** - Don't bundle unrelated changes
2. **Descriptive migration names** - Clear, concise descriptions
3. **Test migrations** - Always test on dev/staging before production
4. **Backup before migration** - Always backup production database
5. **Monitor migration execution** - Watch logs during migration
6. **Document breaking changes** - Note any API or data format changes

## Additional Resources

- [Alembic Documentation](https://alembic.sqlalchemy.org/)
- [SQLAlchemy Migrations Guide](https://docs.sqlalchemy.org/en/20/core/metadata.html)
- Migration scripts: `backend/migrations/versions/`
- Migration runner: `backend/utils/migration_runner.py`

