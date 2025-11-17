# Stamping Existing Databases

If you have an existing database with tables but no Alembic version tracking, you need to "stamp" it with the current migration revision.

## When to Stamp

- You have an existing database with tables
- Alembic version table doesn't exist or is empty
- You want to start using migrations going forward

## How to Stamp

```bash
# Stamp database with the latest migration (head)
alembic stamp head

# Or stamp with a specific revision
alembic stamp 002_schema_fixes
```

## After Stamping

- Alembic will track the database as being at the stamped revision
- Future migrations will only apply new changes
- Existing schema will be preserved

## Important Notes

- **Only stamp if your database schema matches the migration you're stamping**
- If schema doesn't match, you may need to create a custom migration to sync
- Always backup your database before stamping

