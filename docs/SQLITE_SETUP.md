# SQLite Setup Guide

## Overview

SQLite is **automatically used** by default when PostgreSQL is not configured. This makes local development **zero-configuration**!

## How It Works

The application automatically:
1. ✅ Tries PostgreSQL first (if credentials are set)
2. ✅ Falls back to SQLite if PostgreSQL is unavailable
3. ✅ Creates the database file automatically (`rag_app.db`)
4. ✅ Creates all required tables on first use

## No Setup Required!

Just start your app - SQLite will work automatically:

```bash
python app.py
# or
streamlit run app.py
```

The database file `rag_app.db` will be created automatically in your project root.

## Customizing SQLite Location

If you want to store the database in a different location:

```bash
# Set environment variable
export SQLITE_DB_PATH=/path/to/custom/rag_app.db

# Or on Windows PowerShell
$env:SQLITE_DB_PATH = "C:\path\to\custom\rag_app.db"
```

## Database Features

All features work exactly the same with SQLite:
- ✅ Session management
- ✅ Query history
- ✅ User feedback (thumbs up/down)
- ✅ Validated Q&A cache
- ✅ Analytics and metrics

## Database File

- **Location**: `rag_app.db` (default, in project root)
- **Format**: SQLite 3 database file
- **Size**: Starts small, grows with usage
- **Backup**: Just copy the file!
- **Reset**: Delete the file and restart

## Migration to PostgreSQL (When Ready)

When you get access to GCP and want to use PostgreSQL:

1. **Export data from SQLite** (optional):
   ```python
   # You can write a migration script if needed
   # Or just start fresh in PostgreSQL
   ```

2. **Set PostgreSQL environment variables**:
   ```bash
   export POSTGRES_HOST=your-gcp-host
   export POSTGRES_PORT=5432
   export POSTGRES_DB=rag_app
   export POSTGRES_USER=your-user
   export POSTGRES_PASSWORD=your-password
   ```

3. **Restart the app** - it will automatically switch to PostgreSQL!

The app uses the **same API** for both databases, so no code changes needed.

## Performance

- **Local Development**: SQLite is perfect - fast, simple, no server needed
- **Production**: PostgreSQL recommended for multi-user, concurrent access
- **SQLite Limitations**: 
  - Single writer at a time (fine for local dev)
  - No network access (file-based only)
  - Perfect for development, testing, and small deployments

## Troubleshooting

### Database file not created?

Check file permissions in the project directory.

### Want to reset the database?

```bash
# Delete the database file
rm rag_app.db  # Linux/Mac
del rag_app.db  # Windows

# Restart app - new database will be created
```

### Database locked errors?

This usually means:
- Another process is using the database
- Close other instances of the app
- Make sure only one instance is writing at a time

## Benefits

✅ **Zero configuration** - Just works out of the box  
✅ **No server required** - Single file database  
✅ **Easy backup** - Just copy the file  
✅ **Perfect for development** - Fast and simple  
✅ **Easy migration** - Switch to PostgreSQL anytime  

## Next Steps

1. ✅ Start using SQLite now (it's automatic!)
2. ✅ When ready for production, set PostgreSQL env vars
3. ✅ App automatically switches to PostgreSQL
4. ✅ Same code, same features, seamless transition!

