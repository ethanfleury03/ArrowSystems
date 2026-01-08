# Setting Up Database Connection on Windows

## Option 1: Install Cloud SQL Proxy (Recommended)

### Download Cloud SQL Proxy for Windows:

1. **Download the Windows binary:**
   ```powershell
   # Using PowerShell
   Invoke-WebRequest -Uri "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe" -OutFile "cloud-sql-proxy.exe"
   ```

   Or download manually from:
   https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe

2. **Place it in your PATH or current directory:**
   - Option A: Move `cloud-sql-proxy.exe` to a folder in your PATH (e.g., `C:\Windows\System32`)
   - Option B: Keep it in your project directory and use `.\cloud-sql-proxy.exe`

3. **Authenticate with Google Cloud:**
   ```powershell
   gcloud auth application-default login
   ```

4. **Start the proxy:**
   ```powershell
   # In PowerShell (run in background)
   Start-Process -NoNewWindow .\cloud-sql-proxy.exe -ArgumentList "arrow-rag-support-prod:us-central1:rag-postgres"
   
   # Or in Git Bash
   ./cloud-sql-proxy.exe arrow-rag-support-prod:us-central1:rag-postgres &
   ```

5. **Set DATABASE_URL:**
   ```powershell
   # PowerShell
   $env:DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"
   
   # Git Bash
   export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@127.0.0.1:5432/rag_app"
   ```

## Option 2: Direct Connection (If External IP Enabled)

If your Cloud SQL instance has an external IP address enabled:

1. **Find your Cloud SQL instance IP:**
   - Go to Google Cloud Console → SQL → Your instance
   - Check "Public IP address" or "Private IP address"

2. **Set DATABASE_URL directly:**
   ```powershell
   # PowerShell
   $env:DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app"
   
   # Git Bash
   export DATABASE_URL="postgresql://rag_user:YOUR_PASSWORD@<CLOUD_SQL_IP>:5432/rag_app"
   ```

## Option 3: Use Python Script with Authentication

If you have Google Cloud credentials set up, you can use the Python Cloud SQL connector instead.

## Quick Test

After setting up, test the connection:
```bash
python check_db_connection.py
```

Then run your script:
```bash
python update_failed_to_complete.py
```

