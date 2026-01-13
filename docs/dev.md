# Local Development Setup

This guide covers setting up a local development environment for Windows without Docker.

## Prerequisites

- Python 3.12+
- Node.js and npm
- Google Cloud SDK (for Cloud SQL Proxy authentication)
- VS Code or Cursor IDE

## Setup Steps

### 1. Install Cloud SQL Auth Proxy

**Option A: Automatic Download (Recommended)**

Run this PowerShell command from the repo root:

```powershell
.\tools\download-cloud-sql-proxy.ps1
```

**Option B: Manual Download**

1. Visit: https://cloud.google.com/sql/docs/postgres/sql-proxy#install
2. Download the Windows 64-bit executable
3. Rename it to `cloud-sql-proxy.exe`
4. Place it in the `tools/` directory at the repo root:
   ```
   tools/cloud-sql-proxy.exe
   ```

### 2. Configure Environment Variables

Create or update `backend/.env` (this file is gitignored). See `backend/.env.example` for a template.

**Required for Cloud SQL Proxy:**

```env
# Database connection via Cloud SQL Proxy (TCP on Windows)
# IMPORTANT: Use 127.0.0.1:5433 (TCP) on Windows, NOT /cloudsql/... (Unix socket)
# IMPORTANT: Do NOT include "DATABASE_URL=" in the value - only the connection string
DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@127.0.0.1:5433/rag_app

# Development flags
DISABLE_RAG=true
ENVIRONMENT=dev
```

**Common .env Format Errors:**
- ❌ Wrong: `DATABASE_URL=DATABASE_URL=postgresql+psycopg2://...` (duplicate key)
- ❌ Wrong: `DATABASE_URL="postgresql+psycopg2://..."` (quotes not needed, but OK)
- ✅ Correct: `DATABASE_URL=postgresql+psycopg2://USER:PASSWORD@127.0.0.1:5433/rag_app`

**Important Notes:**
- Replace `USER` and `PASSWORD` with your actual Cloud SQL credentials
- Replace `rag_app` with your actual database name if different
- Use `127.0.0.1:5433` (TCP) instead of `/cloudsql/...` (Unix socket) on Windows
- The proxy runs on port 5433 to avoid conflicts with local PostgreSQL (default 5432)
- On Linux/Mac or Cloud Run, you can use Unix socket format: `postgresql+psycopg2://USER:PASSWORD@/rag_app?host=/cloudsql/arrow-rag-support-prod:us-central1:rag-postgres`

### 3. Authenticate with Google Cloud

Before starting the Cloud SQL Proxy, authenticate with Google Cloud:

```powershell
gcloud auth application-default login
```

Or if using a service account:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="path/to/service-account-key.json"
```

## Running the Development Environment

### Option 1: Run Everything Together (Recommended)

Use the compound task **"Dev: All (Proxy + API + Web)"** which starts:
1. Cloud SQL Proxy (port 5433)
2. FastAPI Backend (port 8000)
3. Next.js Frontend (port 3000)

**In VS Code/Cursor:**
- Press `Ctrl+Shift+P` (or `Cmd+Shift+P` on Mac)
- Type "Tasks: Run Task"
- Select "Dev: All (Proxy + API + Web)"

### Option 2: Run Tasks Individually

You can also run each component separately:

1. **Dev: Cloud SQL Proxy** - Starts the Cloud SQL Auth Proxy
2. **Dev: Backend (FastAPI)** - Starts the FastAPI server with auto-reload
3. **Dev: Frontend (Next.js)** - Starts the Next.js development server

### Option 3: Run Without Cloud SQL Proxy

If you have a local PostgreSQL database, use **"Dev: All (No Docker)"** which runs only the backend and frontend.

## VS Code Tasks

The following tasks are available in `.vscode/tasks.json`:

- **Dev: Cloud SQL Proxy** - Runs Cloud SQL Auth Proxy v2 on port 5433
- **Dev: Backend (FastAPI)** - Runs FastAPI with `--reload` on port 8000
- **Dev: Frontend (Next.js)** - Runs Next.js dev server on port 3000
- **Dev: All (No Docker)** - Runs backend + frontend (no proxy)
- **Dev: All (Proxy + API + Web)** - Runs proxy + backend + frontend

## Troubleshooting

### Cloud SQL Proxy Connection Issues

- **Error: "dial tcp: lookup ..."** - Ensure you're authenticated: `gcloud auth application-default login`
- **Error: "connection refused"** - Check that the proxy is running and listening on port 5433
- **Error: "permission denied"** - Verify your Google Cloud account has Cloud SQL Client role

### Backend Import Errors

- Ensure `backend/__init__.py` exists (makes `backend` a proper Python package)
- Run tasks from the repo root (not from `backend/` directory)
- Check that `PYTHONPATH` includes the project root

### Database Connection Errors

- Verify `DATABASE_URL` in `backend/.env` uses `127.0.0.1:5433` (not `/cloudsql/...`)
- Ensure Cloud SQL Proxy is running before starting the backend
- Check that the database name, user, and password are correct

## Development Workflow

1. Start Cloud SQL Proxy (if using Cloud SQL)
2. Start Backend - FastAPI will auto-reload on code changes
3. Start Frontend - Next.js will hot-reload on code changes
4. Access:
   - API: http://127.0.0.1:8000
   - API Docs: http://127.0.0.1:8000/docs
   - Frontend: http://localhost:3000

## Environment Variables

Key environment variables for local development:

- `DISABLE_RAG=true` - Skips RAG model loading for faster startup
- `ENVIRONMENT=dev` - Sets development mode
- `DATABASE_URL` - Database connection string (TCP via proxy on Windows)

See `backend/.env.example` for a complete list of available variables.
