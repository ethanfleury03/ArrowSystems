# Database restore script for Windows PowerShell
# Supports both SQLite and PostgreSQL

param(
    [Parameter(Mandatory=$true)]
    [string]$BackupFile
)

$DATABASE_URL = $env:DATABASE_URL

if ([string]::IsNullOrEmpty($DATABASE_URL)) {
    $DATABASE_URL = "file:./dev.db"
}

Write-Host "🔄 Starting database restore..." -ForegroundColor Cyan
Write-Host "📅 Date: $(Get-Date)" -ForegroundColor Gray
Write-Host "📁 Backup file: $BackupFile" -ForegroundColor Gray
Write-Host "📍 Database URL: $DATABASE_URL" -ForegroundColor Gray

# Check if backup file exists
if (-not (Test-Path $BackupFile)) {
    Write-Host "❌ Error: Backup file not found: $BackupFile" -ForegroundColor Red
    exit 1
}

# Check if DATABASE_URL is SQLite or PostgreSQL
if ($DATABASE_URL -like "file:*") {
    # SQLite restore
    $DB_FILE = $DATABASE_URL -replace "file:", ""
    
    # Check if backup is compressed
    $TEMP_FILE = $BackupFile
    if ($BackupFile -like "*.gz") {
        Write-Host "📦 Decompressing backup..." -ForegroundColor Cyan
        $TEMP_FILE = $BackupFile -replace "\.gz$", ""
        
        # Decompress using .NET
        $compressed = [System.IO.File]::OpenRead($BackupFile)
        $gzip = [System.IO.Compression.GZipStream]::new($compressed, [System.IO.Compression.CompressionMode]::Decompress)
        $output = [System.IO.File]::Create($TEMP_FILE)
        $gzip.CopyTo($output)
        $output.Close()
        $gzip.Close()
        $compressed.Close()
    }
    
    # Create backup of current database before restore
    if (Test-Path $DB_FILE) {
        $CURRENT_BACKUP = "$DB_FILE.pre-restore-$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Copy-Item $DB_FILE $CURRENT_BACKUP
        Write-Host "💾 Current database backed up to: $CURRENT_BACKUP" -ForegroundColor Yellow
    }
    
    # Restore
    Copy-Item $TEMP_FILE $DB_FILE -Force
    Write-Host "✅ SQLite database restored successfully" -ForegroundColor Green
    
    # Clean up temporary file if it was decompressed
    if ($BackupFile -like "*.gz") {
        Remove-Item $TEMP_FILE
    }
    
} elseif ($DATABASE_URL -like "postgresql*") {
    # PostgreSQL restore
    Write-Host "⚠️  WARNING: This will overwrite your current PostgreSQL database!" -ForegroundColor Red
    $CONFIRM = Read-Host "Are you sure you want to continue? (yes/no)"
    
    if ($CONFIRM -ne "yes") {
        Write-Host "❌ Restore cancelled" -ForegroundColor Yellow
        exit 0
    }
    
    # Check if backup is compressed
    $TEMP_FILE = $BackupFile
    if ($BackupFile -like "*.gz") {
        Write-Host "📦 Decompressing backup..." -ForegroundColor Cyan
        $TEMP_FILE = $BackupFile -replace "\.gz$", ""
        
        # Decompress
        $compressed = [System.IO.File]::OpenRead($BackupFile)
        $gzip = [System.IO.Compression.GZipStream]::new($compressed, [System.IO.Compression.CompressionMode]::Decompress)
        $output = [System.IO.File]::Create($TEMP_FILE)
        $gzip.CopyTo($output)
        $output.Close()
        $gzip.Close()
        $compressed.Close()
    }
    
    # Restore using psql
    $psql = Get-Command psql -ErrorAction SilentlyContinue
    
    if ($psql) {
        try {
            Get-Content $TEMP_FILE | & psql $DATABASE_URL
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ PostgreSQL database restored successfully" -ForegroundColor Green
            } else {
                Write-Host "❌ Error: PostgreSQL restore failed" -ForegroundColor Red
                exit 1
            }
        } catch {
            Write-Host "❌ Error: PostgreSQL restore failed: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Error: psql not found. Install PostgreSQL client tools." -ForegroundColor Red
        Write-Host "   Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        exit 1
    }
    
    # Clean up temporary file if it was decompressed
    if ($BackupFile -like "*.gz") {
        Remove-Item $TEMP_FILE
    }
} else {
    Write-Host "❌ Error: Unsupported database type in DATABASE_URL" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Restore completed successfully!" -ForegroundColor Green

