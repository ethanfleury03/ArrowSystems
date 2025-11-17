# Database Migrations

This directory contains Alembic database migration scripts.

## Migration Workflow

### Development Mode

Migrations run automatically on application startup. No manual intervention needed.

### Production Mode

**Migrations DO NOT run automatically in production.** You must run them manually:

```bash
# Check current migration status
python -m backend.utils.migration_runner status

# Run pending migrations
python -m backend.utils.migration_runner upgrade

# Or use Alembic directly
alembic upgrade head
```

### Creating a New Migration

1. **Make schema changes** in `backend/utils/db.py` (update SQLAlchemy models)

2. **Generate migration script:**
   ```bash
   alembic revision --autogenerate -m "description of changes"
   ```

3. **Review the generated migration** in `backend/migrations/versions/`

4. **Test in dev mode** - migrations run automatically on startup

5. **Commit migration file** to version control

### Running Migrations

**Development:**
- Automatic on startup (no action needed)

**Production:**
- **Before deployment:** Run migrations manually
- **During deployment:** Include migration step in deployment script
- **After deployment:** Application will fail to start if migrations are pending

### Migration Safety

- All migrations use Alembic `op` functions (not raw SQL where possible)
- SQLite and PostgreSQL compatible migrations
- Migrations are idempotent where possible
- Always test migrations on a copy of production data first

### Troubleshooting

**Migration fails:**
- Check database connection
- Verify migration script syntax
- Review Alembic version table: `SELECT * FROM alembic_version;`

**Schema drift detected:**
- Application will fail to start in production
- Run `alembic upgrade head` to apply pending migrations
- Check migration history: `alembic history`

## Directory Structure

```
migrations/
├── README.md          # This file
├── alembic.ini        # Alembic configuration
├── env.py             # Alembic environment setup
└── versions/          # Migration scripts (auto-generated)
    └── *.py           # Individual migration files
```

