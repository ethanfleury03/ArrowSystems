# PowerShell script to run Docker container locally
# Run this after building with build-local.ps1

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Running Docker Container Locally" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if image exists
Write-Host "Checking if image exists..." -ForegroundColor Yellow
$imageExists = docker images rag-app:local --format "{{.Repository}}:{{.Tag}}" | Select-String "rag-app:local"

if (-not $imageExists) {
    Write-Host "❌ Image 'rag-app:local' not found!" -ForegroundColor Red
    Write-Host "Please run build-local.ps1 first to build the image." -ForegroundColor Yellow
    exit 1
}

Write-Host "✅ Image found" -ForegroundColor Green
Write-Host ""
Write-Host "Starting container..." -ForegroundColor Yellow
Write-Host "The app will be available at: http://localhost:8501" -ForegroundColor Cyan
Write-Host ""
Write-Host "Press Ctrl+C to stop the container" -ForegroundColor Yellow
Write-Host ""

# Check for ANTHROPIC_API_KEY and pass it if available
$envVars = @("-e", "PYTHONPATH=/app")
if ($env:ANTHROPIC_API_KEY) {
    Write-Host "✅ ANTHROPIC_API_KEY found in environment" -ForegroundColor Green
    $envVars += "-e", "ANTHROPIC_API_KEY=$env:ANTHROPIC_API_KEY"
} else {
    Write-Host "⚠️  ANTHROPIC_API_KEY not set in environment" -ForegroundColor Yellow
    Write-Host "   Claude features will be disabled" -ForegroundColor Yellow
    Write-Host "   Set it with: `$env:ANTHROPIC_API_KEY='your-key'" -ForegroundColor Cyan
}

# Run the container
docker run -it --rm `
    -p 8501:8501 `
    $envVars `
    rag-app:local

