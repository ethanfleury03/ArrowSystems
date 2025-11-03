# Database Setup Guide

This guide explains how to set up and configure **PostgreSQL** (for production) or **SQLite** (for local development) for the ArrowSystems RAG application.

The application **automatically detects and uses the best available database**:
- ✅ **SQLite** - Used automatically if PostgreSQL is not configured (perfect for local dev)
- ✅ **PostgreSQL** - Used if credentials are provided (recommended for production)

## Quick Start

### Option 1: SQLite (Zero Configuration - Recommended for Local Dev)

**No setup required!** Just start the app and SQLite will be used automatically.

```bash
# That's it! The app will create rag_app.db automatically
python app.py
```

### Option 2: PostgreSQL (For Production)

1. **Install PostgreSQL** (if not already installed)
2. **Set environment variables**
3. **Create database** (if needed)
4. **Run setup script** to create tables
5. **Verify connection**

## Quick Start

1. **Install PostgreSQL** (if not already installed)
2. **Set environment variables**
3. **Create database** (if needed)
4. **Run setup script** to create tables
5. **Verify connection**

## SQLite Setup (Automatic - Zero Configuration)

SQLite is **automatically used** when PostgreSQL is not available. No setup needed!

### Features:
- ✅ **No installation required** - Built into Python
- ✅ **No server needed** - Single file database
- ✅ **Perfect for local development** - Fast and simple
- ✅ **Auto-creates database** - Just start the app!

### Customizing SQLite Location:

```bash
# Optional: Set custom path
export SQLITE_DB_PATH=/path/to/custom/rag_app.db
```

### SQLite Database File:

- **Default location**: `rag_app.db` (in project root)
- **Can be checked into git** (for testing) or added to `.gitignore`
- **Easy to backup** - Just copy the file
- **Easy to reset** - Delete the file and restart

---

## PostgreSQL Setup (For Production)

When you're ready for production or need PostgreSQL features:

### Windows

Download and install from [PostgreSQL Downloads](https://www.postgresql.org/download/windows/)

Or use Chocolatey:
```powershell
choco install postgresql
```

### macOS

```bash
brew install postgresql@15
brew services start postgresql@15
```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install postgresql postgresql-contrib
sudo systemctl start postgresql
sudo systemctl enable postgresql
```

## Environment Variables

The application supports **two naming conventions** for compatibility:

### Option 1: POSTGRES_* (Recommended)
```bash
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=rag_app
POSTGRES_USER=postgres
POSTGRES_PASSWORD=your-secure-password
```

### Option 2: DB_* (Legacy/Docker)
```bash
DB_HOST=localhost
DB_PORT=5432
DB_NAME=rag_app
DB_USER=postgres
DB_PASSWORD=your-secure-password
```

**Note:** If both are set, `POSTGRES_*` takes precedence.

### Setting Environment Variables

#### PowerShell (Windows)
```powershell
# One-time (current session)
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "rag_app"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "your-password"
```

#### Bash (Linux/Mac)
```bash
# One-time (current session)
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=rag_app
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your-password
```

#### Using .env File (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your PostgreSQL credentials

3. Load the file (if your shell supports it):
   ```bash
   # For bash/zsh (Linux/Mac)
   export $(cat .env | xargs)
   
   # For PowerShell (Windows)
   Get-Content .env | ForEach-Object {
       if ($_ -match '^([^#][^=]*)=(.*)$') {
           [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
       }
   }
   ```

## Creating the Database

If your database doesn't exist yet:

### Using psql Command Line

```bash
# Connect to PostgreSQL as superuser
psql -U postgres

# Create database
CREATE DATABASE rag_app;

# Create user (optional, if not using 'postgres' user)
CREATE USER rag_user WITH PASSWORD 'your-password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE rag_app TO rag_user;

# Exit psql
\q
```

### Using Createdb Command

```bash
createdb -U postgres rag_app
```

## Running the Setup Script

Once PostgreSQL is running and environment variables are set:

```bash
python scripts/setup_postgres.py
```

This will create all required tables:
- `sessions` - User session management
- `queries` - Query history and responses
- `feedback` - User feedback (thumbs up/down)
- `validated_qna` - Validated Q&A cache

## Verifying Connection

Use the connection checker script:

```bash
python scripts/check_postgres.py
```

This will:
- ✅ Check environment variables
- ✅ Test database connection
- ✅ Verify tables exist
- ✅ Show connection details

## Troubleshooting

### "password authentication failed"

**Problem:** Wrong username or password

**Solution:**
1. Verify your `POSTGRES_USER` and `POSTGRES_PASSWORD` are correct
2. Reset password if needed:
   ```sql
   ALTER USER postgres WITH PASSWORD 'new-password';
   ```

### "could not connect to server"

**Problem:** PostgreSQL is not running or wrong host/port

**Solution:**
1. Check if PostgreSQL is running:
   ```bash
   # Windows
   Get-Service postgresql*
   
   # Linux/Mac
   sudo systemctl status postgresql
   ```

2. Start PostgreSQL:
   ```bash
   # Windows
   Start-Service postgresql-x64-15  # Version may vary
   
   # Linux/Mac
   sudo systemctl start postgresql
   ```

3. Verify host and port in environment variables

### "database does not exist"

**Problem:** The database hasn't been created

**Solution:**
```sql
CREATE DATABASE rag_app;
```

### "psycopg2 not installed"

**Problem:** Python package missing

**Solution:**
```bash
pip install psycopg2-binary
```

## Google Cloud SQL Setup

If using Google Cloud SQL, you'll need the Cloud SQL Proxy:

1. **Install Cloud SQL Proxy:**
   ```bash
   # Download from: https://cloud.google.com/sql/docs/postgres/sql-proxy
   ```

2. **Set connection name:**
   ```bash
   export CLOUD_SQL_CONNECTION_NAME=project:region:instance-name
   ```

3. **Start the proxy:**
   ```bash
   ./cloud-sql-proxy $CLOUD_SQL_CONNECTION_NAME
   ```

4. **Connect through localhost:**
   ```bash
   export POSTGRES_HOST=127.0.0.1
   export POSTGRES_PORT=5432
   ```

## Docker Setup

If running PostgreSQL in Docker:

```yaml
# Add to docker-compose.yml
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_DB: rag_app
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: your-password
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data

volumes:
  postgres-data:
```

Then set:
```bash
export POSTGRES_HOST=localhost  # or 'postgres' if in same network
export POSTGRES_PORT=5432
export POSTGRES_DB=rag_app
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your-password
```

## Security Best Practices

1. **Use strong passwords** - Don't use default "password"
2. **Limit database user privileges** - Create a dedicated user with only needed permissions
3. **Use environment variables** - Never commit passwords to git
4. **Use SSL connections** - In production, enable SSL for PostgreSQL connections
5. **Firewall rules** - Restrict database access to trusted IPs only

## Next Steps

After PostgreSQL is set up:
1. ✅ Verify connection: `python scripts/check_postgres.py`
2. ✅ Create tables: `python scripts/setup_postgres.py`
3. ✅ Start the application: `./start.sh` or `python app.py`

## Need Help?

- Check logs for detailed error messages
- Verify environment variables are set correctly
- Ensure PostgreSQL service is running
- Test connection with: `psql -U postgres -d rag_app -h localhost`

