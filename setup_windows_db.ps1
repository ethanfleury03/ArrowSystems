# PowerShell script to set up Cloud SQL Proxy and DATABASE_URL on Windows

param(
    [string]$Password,
    [string]$ConnectionMethod = "proxy"  # "proxy" or "direct"
)

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Windows Database Connection Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Cloud SQL Proxy exists
$proxyPath = "cloud-sql-proxy.exe"
$proxyInPath = Get-Command cloud-sql-proxy.exe -ErrorAction SilentlyContinue

if (-not $proxyInPath -and -not (Test-Path $proxyPath)) {
    Write-Host "⚠️  Cloud SQL Proxy not found!" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Downloading Cloud SQL Proxy..." -ForegroundColor Yellow
    $downloadUrl = "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe"
    try {
        Invoke-WebRequest -Uri $downloadUrl -OutFile $proxyPath
        Write-Host "✅ Downloaded cloud-sql-proxy.exe" -ForegroundColor Green
    } catch {
        Write-Host "❌ Failed to download Cloud SQL Proxy" -ForegroundColor Red
        Write-Host "Please download manually from: $downloadUrl" -ForegroundColor Yellow
        exit 1
    }
}

if ($ConnectionMethod -eq "proxy") {
    Write-Host "Setting up Cloud SQL Proxy connection..." -ForegroundColor Cyan
    Write-Host ""
    
    # Check if proxy is already running
    $proxyProcess = Get-Process -Name "cloud-sql-proxy" -ErrorAction SilentlyContinue
    if ($proxyProcess) {
        Write-Host "⚠️  Cloud SQL Proxy is already running (PID: $($proxyProcess.Id))" -ForegroundColor Yellow
        Write-Host "   If you need to restart it, stop it first:" -ForegroundColor Yellow
        Write-Host "   Stop-Process -Name cloud-sql-proxy" -ForegroundColor Gray
    } else {
        Write-Host "Starting Cloud SQL Proxy..." -ForegroundColor Cyan
        $proxyExe = if ($proxyInPath) { "cloud-sql-proxy.exe" } else { ".\cloud-sql-proxy.exe" }
        Start-Process -NoNewWindow $proxyExe -ArgumentList "arrow-rag-support-prod:us-central1:rag-postgres"
        Start-Sleep -Seconds 3
        
        $proxyProcess = Get-Process -Name "cloud-sql-proxy" -ErrorAction SilentlyContinue
        if ($proxyProcess) {
            Write-Host "✅ Cloud SQL Proxy started (PID: $($proxyProcess.Id))" -ForegroundColor Green
        } else {
            Write-Host "❌ Failed to start Cloud SQL Proxy" -ForegroundColor Red
            Write-Host "   Make sure you're authenticated: gcloud auth application-default login" -ForegroundColor Yellow
            exit 1
        }
    }
    
    $hostAddress = "127.0.0.1"
} else {
    Write-Host "Setting up direct connection..." -ForegroundColor Cyan
    Write-Host ""
    $hostAddress = Read-Host "Enter Cloud SQL IP address"
}

if (-not $Password) {
    $Password = Read-Host "Enter database password" -AsSecureString
    $Password = [Runtime.InteropServices.Marshal]::PtrToStringAuto([Runtime.InteropServices.Marshal]::SecureStringToBSTR($Password))
}

$databaseUrl = "postgresql://rag_user:$Password@${hostAddress}:5432/rag_app"
$env:DATABASE_URL = $databaseUrl

Write-Host ""
Write-Host "✅ DATABASE_URL has been set!" -ForegroundColor Green
Write-Host "   Connection: postgresql://rag_user:***@${hostAddress}:5432/rag_app" -ForegroundColor Gray
Write-Host ""
Write-Host "To make this permanent in this session, run:" -ForegroundColor Yellow
Write-Host "   `$env:DATABASE_URL = '$databaseUrl'" -ForegroundColor Gray
Write-Host ""
Write-Host "Test the connection:" -ForegroundColor Cyan
Write-Host "   python check_db_connection.py" -ForegroundColor Gray
Write-Host ""
Write-Host "Run your script:" -ForegroundColor Cyan
Write-Host "   python update_failed_to_complete.py" -ForegroundColor Gray
Write-Host ""

