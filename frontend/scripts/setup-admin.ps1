# Quick Admin Account Setup Script
# This script helps you create your admin account for login

Write-Host "🔐 Admin Account Setup" -ForegroundColor Cyan
Write-Host ""

# Check if .env.local exists
if (-not (Test-Path ".env.local")) {
    Write-Host "❌ Error: .env.local file not found!" -ForegroundColor Red
    Write-Host "Please create .env.local in the frontend directory with:" -ForegroundColor Yellow
    Write-Host "  DATABASE_URL=..." -ForegroundColor Gray
    Write-Host "  SESSION_SECRET=..." -ForegroundColor Gray
    Write-Host "  ADMIN_EMAIL=your-email@example.com" -ForegroundColor Gray
    Write-Host "  ADMIN_PASSWORD=your-password" -ForegroundColor Gray
    exit 1
}

# Load environment variables from .env.local
$envVars = Get-Content ".env.local" | Where-Object { $_ -match '^\s*[^#]' -and $_ -match '=' }
foreach ($line in $envVars) {
    if ($line -match '^\s*([^=]+)=(.*)$') {
        $key = $matches[1].Trim()
        $value = $matches[2].Trim() -replace '^["\']|["\']$', ''
        [Environment]::SetEnvironmentVariable($key, $value, "Process")
    }
}

Write-Host "📋 Checking database setup..." -ForegroundColor Yellow

# Check if migrations are run
if (-not (Test-Path "prisma/migrations")) {
    Write-Host "⚠️  No migrations found. Running migrations..." -ForegroundColor Yellow
    npm run prisma:migrate:dev -- --name init
    if ($LASTEXITCODE -ne 0) {
        Write-Host "❌ Migration failed!" -ForegroundColor Red
        exit 1
    }
}

# Check if Prisma client is generated
if (-not (Test-Path "node_modules/.prisma")) {
    Write-Host "📦 Generating Prisma client..." -ForegroundColor Yellow
    npm run prisma:generate
}

# Get admin credentials
$adminEmail = $env:ADMIN_EMAIL
$adminPassword = $env:ADMIN_PASSWORD

if (-not $adminEmail -or -not $adminPassword) {
    Write-Host "❌ Error: ADMIN_EMAIL and ADMIN_PASSWORD not found in .env.local" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please add to your .env.local file:" -ForegroundColor Yellow
    Write-Host "  ADMIN_EMAIL=your-email@example.com" -ForegroundColor Gray
    Write-Host "  ADMIN_PASSWORD=your-password" -ForegroundColor Gray
    exit 1
}

Write-Host "✅ Found admin credentials:" -ForegroundColor Green
Write-Host "   Email: $adminEmail" -ForegroundColor Gray
Write-Host ""

# Create admin account
Write-Host "🔨 Creating admin account..." -ForegroundColor Yellow
$env:ADMIN_EMAIL = $adminEmail
$env:ADMIN_PASSWORD = $adminPassword
npm run prisma:seed

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✅ Admin account created successfully!" -ForegroundColor Green
    Write-Host ""
    Write-Host "🚀 You can now:" -ForegroundColor Cyan
    Write-Host "   1. Start the dev server: npm run dev" -ForegroundColor White
    Write-Host "   2. Go to http://localhost:3000" -ForegroundColor White
    Write-Host "   3. Login with:" -ForegroundColor White
    Write-Host "      Email: $adminEmail" -ForegroundColor Gray
    Write-Host "      Password: [your password]" -ForegroundColor Gray
} else {
    Write-Host "❌ Failed to create admin account" -ForegroundColor Red
    exit 1
}

