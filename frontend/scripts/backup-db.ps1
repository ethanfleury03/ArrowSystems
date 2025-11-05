# Database backup script for Windows PowerShell
# Supports both SQLite and PostgreSQL

$BACKUP_DIR = ".\backups"
$DATE = Get-Date -Format "yyyyMMdd_HHmmss"
$DATABASE_URL = $env:DATABASE_URL

if ([string]::IsNullOrEmpty($DATABASE_URL)) {
    $DATABASE_URL = "file:./dev.db"
}

# Create backup directory if it doesn't exist
if (-not (Test-Path $BACKUP_DIR)) {
    New-Item -ItemType Directory -Path $BACKUP_DIR | Out-Null
}

Write-Host "🔄 Starting database backup..." -ForegroundColor Cyan
Write-Host "📅 Date: $(Get-Date)" -ForegroundColor Gray
Write-Host "📍 Database URL: $DATABASE_URL" -ForegroundColor Gray

# Check if DATABASE_URL is SQLite or PostgreSQL
if ($DATABASE_URL -like "file:*") {
    # SQLite backup
    $DB_FILE = $DATABASE_URL -replace "file:", ""
    
    if (Test-Path $DB_FILE) {
        $BACKUP_FILE = "$BACKUP_DIR\sqlite_backup_$DATE.db"
        Copy-Item $DB_FILE $BACKUP_FILE
        Write-Host "✅ SQLite backup created: $BACKUP_FILE" -ForegroundColor Green
        
        # Compress the backup using .NET compression
        $bytes = [System.IO.File]::ReadAllBytes($BACKUP_FILE)
        $compressed = [System.IO.Compression.GZipStream]::new(
            [System.IO.File]::Create("$BACKUP_FILE.gz"),
            [System.IO.Compression.CompressionLevel]::Optimal
        )
        $compressed.Write($bytes, 0, $bytes.Length)
        $compressed.Close()
        
        Write-Host "✅ Compressed backup created: $BACKUP_FILE.gz" -ForegroundColor Green
        
        # Remove uncompressed backup to save space
        Remove-Item $BACKUP_FILE
    } else {
        Write-Host "❌ Error: SQLite database file not found: $DB_FILE" -ForegroundColor Red
        exit 1
    }
} elseif ($DATABASE_URL -like "postgresql*") {
    # PostgreSQL backup
    $BACKUP_FILE = "$BACKUP_DIR\postgres_backup_$DATE.sql"
    
    # Check if pg_dump is available
    $pgDump = Get-Command pg_dump -ErrorAction SilentlyContinue
    
    if ($pgDump) {
        try {
            & pg_dump $DATABASE_URL | Out-File -FilePath $BACKUP_FILE -Encoding UTF8
            
            if ($LASTEXITCODE -eq 0) {
                Write-Host "✅ PostgreSQL backup created: $BACKUP_FILE" -ForegroundColor Green
                
                # Compress the backup
                $bytes = [System.IO.File]::ReadAllBytes($BACKUP_FILE)
                $compressed = [System.IO.Compression.GZipStream]::new(
                    [System.IO.File]::Create("$BACKUP_FILE.gz"),
                    [System.IO.Compression.CompressionLevel]::Optimal
                )
                $compressed.Write($bytes, 0, $bytes.Length)
                $compressed.Close()
                
                Write-Host "✅ Compressed backup created: $BACKUP_FILE.gz" -ForegroundColor Green
                
                # Remove uncompressed backup
                Remove-Item $BACKUP_FILE
            } else {
                Write-Host "❌ Error: PostgreSQL backup failed" -ForegroundColor Red
                exit 1
            }
        } catch {
            Write-Host "❌ Error: PostgreSQL backup failed: $_" -ForegroundColor Red
            exit 1
        }
    } else {
        Write-Host "❌ Error: pg_dump not found. Install PostgreSQL client tools." -ForegroundColor Red
        Write-Host "   Download from: https://www.postgresql.org/download/windows/" -ForegroundColor Yellow
        exit 1
    }
} else {
    Write-Host "❌ Error: Unsupported database type in DATABASE_URL" -ForegroundColor Red
    exit 1
}

# Clean up old backups (keep last 30 days)
Write-Host "🧹 Cleaning up old backups (keeping last 30 days)..." -ForegroundColor Cyan
Get-ChildItem -Path $BACKUP_DIR -Filter "*.db.gz" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | Remove-Item
Get-ChildItem -Path $BACKUP_DIR -Filter "*.sql.gz" | Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) } | Remove-Item

Write-Host "✅ Backup completed successfully!" -ForegroundColor Green
Write-Host "📁 Backup location: $BACKUP_DIR" -ForegroundColor Gray

