# PowerShell script to enable Docker BuildKit
# This script sets the required environment variables for BuildKit caching
# Run this before using docker-compose to ensure dependency caching works properly

Write-Host "Enabling Docker BuildKit for improved caching..." -ForegroundColor Cyan

$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

Write-Host "✅ BuildKit enabled!" -ForegroundColor Green
Write-Host ""
Write-Host "Environment variables set for this session:" -ForegroundColor Yellow
Write-Host "  DOCKER_BUILDKIT = $env:DOCKER_BUILDKIT" -ForegroundColor White
Write-Host "  COMPOSE_DOCKER_CLI_BUILD = $env:COMPOSE_DOCKER_CLI_BUILD" -ForegroundColor White
Write-Host ""
Write-Host "You can now run docker-compose commands and dependency caching will work." -ForegroundColor Cyan
Write-Host ""
Write-Host "Note: These variables are only set for the current PowerShell session." -ForegroundColor Yellow
Write-Host "To make them permanent, add them to your PowerShell profile or set them system-wide." -ForegroundColor Yellow





