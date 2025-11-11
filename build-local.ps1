# PowerShell script to build Docker image locally
# This helps test the build before deploying to GCP

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Building Docker Image Locally" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is running
Write-Host "Checking Docker status..." -ForegroundColor Yellow
docker ps 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "❌ Docker is not running!" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please start Docker Desktop:" -ForegroundColor Yellow
    Write-Host "  1. Open Docker Desktop from Start menu" -ForegroundColor White
    Write-Host "  2. Wait for 'Docker Desktop is running' message" -ForegroundColor White
    Write-Host "  3. Run this script again" -ForegroundColor White
    Write-Host ""
    exit 1
}
Write-Host "✅ Docker is running" -ForegroundColor Green

Write-Host ""
Write-Host "Building image with BuildKit enabled..." -ForegroundColor Yellow
Write-Host "Image name: rag-app:local" -ForegroundColor Cyan
Write-Host "Build environment: production" -ForegroundColor Cyan
Write-Host ""
Write-Host "This may take 10-20 minutes on first build..." -ForegroundColor Yellow
Write-Host ""

# Build with BuildKit enabled (required for heredoc syntax)
$env:DOCKER_BUILDKIT = "1"
docker build `
    --build-arg BUILD_ENV=production `
    -t rag-app:local `
    -f backend/Dockerfile.backend `
    .

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host "✅ Build successful!" -ForegroundColor Green
    Write-Host "==========================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "To run the container locally:" -ForegroundColor Cyan
    Write-Host "  docker run -p 8000:8000 rag-app:local" -ForegroundColor White
    Write-Host ""
    Write-Host "Or use docker-compose to run both backend and frontend:" -ForegroundColor Cyan
    Write-Host "  docker-compose up" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host ""
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host "❌ Build failed!" -ForegroundColor Red
    Write-Host "==========================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
    exit 1
}

