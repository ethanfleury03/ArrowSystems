# Quick Reference: Setting Environment Variables

## Required Environment Variables

### 1. Anthropic Claude API Key

**PowerShell (one-time per session):**
```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-api03-your-key-here"
```

**Bash (Linux/Mac):**
```bash
export ANTHROPIC_API_KEY="sk-ant-api03-your-key-here"
```

### 2. PostgreSQL Database Configuration

**PowerShell (one-time per session):**
```powershell
$env:POSTGRES_HOST = "localhost"
$env:POSTGRES_PORT = "5432"
$env:POSTGRES_DB = "rag_app"
$env:POSTGRES_USER = "postgres"
$env:POSTGRES_PASSWORD = "your-secure-password"
```

**Bash (Linux/Mac):**
```bash
export POSTGRES_HOST=localhost
export POSTGRES_PORT=5432
export POSTGRES_DB=rag_app
export POSTGRES_USER=postgres
export POSTGRES_PASSWORD=your-secure-password
```

**Alternative: Using DB_* naming (also supported):**
```bash
export DB_HOST=localhost
export DB_PORT=5432
export DB_NAME=rag_app
export DB_USER=postgres
export DB_PASSWORD=your-secure-password
```

## Using .env File (Recommended)

1. Copy the example file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and fill in your values

3. Load the file:
   
   **PowerShell (Windows):**
   ```powershell
   Get-Content .env | ForEach-Object {
       if ($_ -match '^([^#][^=]*)=(.*)$') {
           [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
       }
   }
   ```
   
   **Bash (Linux/Mac):**
   ```bash
   export $(cat .env | grep -v '^#' | xargs)
   ```

## Making Environment Variables Permanent

### PowerShell Profile (Windows)

```powershell
# Check if profile exists
Test-Path $PROFILE

# If not, create it
New-Item -Path $PROFILE -Type File -Force

# Add your environment variables
Add-Content $PROFILE '$env:ANTHROPIC_API_KEY = "your-key"'
Add-Content $PROFILE '$env:POSTGRES_HOST = "localhost"'
Add-Content $PROFILE '$env:POSTGRES_PORT = "5432"'
Add-Content $PROFILE '$env:POSTGRES_DB = "rag_app"'
Add-Content $PROFILE '$env:POSTGRES_USER = "postgres"'
Add-Content $PROFILE '$env:POSTGRES_PASSWORD = "your-password"'
```

### Shell Profile (Linux/Mac - .bashrc or .zshrc)

```bash
echo 'export ANTHROPIC_API_KEY="your-key"' >> ~/.bashrc
echo 'export POSTGRES_HOST=localhost' >> ~/.bashrc
echo 'export POSTGRES_PORT=5432' >> ~/.bashrc
echo 'export POSTGRES_DB=rag_app' >> ~/.bashrc
echo 'export POSTGRES_USER=postgres' >> ~/.bashrc
echo 'export POSTGRES_PASSWORD="your-password"' >> ~/.bashrc

# Reload shell
source ~/.bashrc
```

## Verifying Environment Variables

### Check if variables are set:

**PowerShell:**
```powershell
echo $env:ANTHROPIC_API_KEY
echo $env:POSTGRES_HOST
```

**Bash:**
```bash
echo $ANTHROPIC_API_KEY
echo $POSTGRES_HOST
```

### Test PostgreSQL connection:

```bash
python scripts/check_postgres.py
```

## Order of Operations

1. **Set environment variables** (see above)
2. **Verify PostgreSQL connection**: `python scripts/check_postgres.py`
3. **Create database tables**: `python scripts/setup_postgres.py`
4. **Start the application**: `./start.sh` or `python app.py`

## Troubleshooting

### Variables not persisting?

- Make sure you're setting them in the correct shell/profile
- Restart your terminal after adding to profile
- Check if variables are set: `python scripts/check_postgres.py`

### PostgreSQL connection issues?

See `docs/POSTGRESQL_SETUP.md` for detailed troubleshooting.

## Security Notes

- ⚠️ Never commit `.env` files to git (it's in .gitignore)
- ⚠️ Use strong passwords for PostgreSQL
- ⚠️ Don't share API keys or passwords publicly
- ✅ Use environment variables instead of hardcoding secrets

