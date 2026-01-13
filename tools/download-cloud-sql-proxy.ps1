# Download Cloud SQL Auth Proxy v2 for Windows
# This script downloads the latest Windows 64-bit executable

$ErrorActionPreference = "Stop"

Write-Host "Downloading Cloud SQL Auth Proxy v2 for Windows..." -ForegroundColor Cyan

# Cloud SQL Proxy download URL (Windows 64-bit)
$downloadUrl = "https://storage.googleapis.com/cloud-sql-connectors/cloud-sql-proxy/v2.8.0/cloud-sql-proxy.x64.exe"
$outputPath = Join-Path $PSScriptRoot "cloud-sql-proxy.exe"

try {
    # Download the file
    Invoke-WebRequest -Uri $downloadUrl -OutFile $outputPath -UseBasicParsing
    
    Write-Host "✅ Successfully downloaded cloud-sql-proxy.exe to $outputPath" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "1. Ensure you're authenticated: gcloud auth application-default login" -ForegroundColor Yellow
    Write-Host "2. Run the VS Code task: Dev: Cloud SQL Proxy" -ForegroundColor Yellow
}
catch {
    Write-Host "❌ Error downloading Cloud SQL Proxy: $_" -ForegroundColor Red
    Write-Host ""
    Write-Host "Manual download:" -ForegroundColor Yellow
    Write-Host "1. Visit: https://cloud.google.com/sql/docs/postgres/sql-proxy#install" -ForegroundColor Yellow
    Write-Host "2. Download Windows 64-bit executable" -ForegroundColor Yellow
    Write-Host "3. Rename to cloud-sql-proxy.exe and place in tools/ directory" -ForegroundColor Yellow
    exit 1
}
