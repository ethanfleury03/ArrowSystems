# Dockerfile Syntax Validator
# This script validates the Dockerfile without requiring Docker Desktop

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Dockerfile Syntax Validation" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Dockerfile exists
if (-not (Test-Path "Dockerfile")) {
    Write-Host "❌ Dockerfile not found!" -ForegroundColor Red
    exit 1
}

Write-Host "✅ Dockerfile found" -ForegroundColor Green
Write-Host ""

# Basic syntax checks
Write-Host "Running basic syntax checks..." -ForegroundColor Yellow
$errors = @()
$warnings = @()

$content = Get-Content "Dockerfile" -Raw
$lines = Get-Content "Dockerfile"

# Check 1: Verify BuildKit syntax declaration
if ($lines[0] -notmatch "^# syntax=docker/dockerfile:") {
    $warnings += "Line 1: BuildKit syntax declaration not found (recommended for heredoc support)"
}

# Check 2: Check for proper FROM statements
$fromCount = ($lines | Select-String "^FROM ").Count
if ($fromCount -eq 0) {
    $errors += "No FROM statement found"
}

# Check 3: Check heredoc syntax
$heredocStarts = ($lines | Select-String -Pattern "<<EOF").Matches.Count
$heredocEnds = ($lines | Select-String -Pattern "^EOF$").Matches.Count
if ($heredocStarts -ne $heredocEnds) {
    $errors += "Heredoc mismatch: $heredocStarts starts but $heredocEnds EOF markers found"
} else {
    Write-Host "✅ Heredoc syntax check passed ($heredocStarts heredocs with matching EOF)" -ForegroundColor Green
}

# Check 4: Check for unclosed quotes in heredoc sections
$inHeredoc = $false
$lineNum = 0
foreach ($line in $lines) {
    $lineNum++
    if ($line -match "COPY.*<<EOF") {
        $inHeredoc = $true
    }
    if ($line -match "^EOF") {
        $inHeredoc = $false
    }
    
    # Check for potential quote issues in heredocs
    if ($inHeredoc -and $line -match 'echo.*".*".*"') {
        # Count quotes - should be even
        $doubleQuotes = ([regex]::Matches($line, '"')).Count
        if ($doubleQuotes % 2 -ne 0) {
            $warnings += "Line $lineNum : Potential unclosed quote in heredoc"
        }
    }
}

# Check 5: Verify WORKDIR exists
$hasWorkdir = ($lines | Select-String "^WORKDIR ").Count -gt 0
if (-not $hasWorkdir) {
    $warnings += "No WORKDIR statement found (file paths may be relative to /)"
}

# Check 6: Check for EXPOSE statement
$hasExpose = ($lines | Select-String "^EXPOSE ").Count -gt 0
if (-not $hasExpose) {
    $warnings += "No EXPOSE statement found (recommended for documentation)"
}

# Check 7: Check for CMD or ENTRYPOINT
$hasCmd = ($lines | Select-String "^CMD |^ENTRYPOINT ").Count -gt 0
if (-not $hasCmd) {
    $errors += "No CMD or ENTRYPOINT statement found (container won't run)"
}

# Report results
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Validation Results" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

if ($errors.Count -eq 0 -and $warnings.Count -eq 0) {
    Write-Host "✅ Basic syntax checks passed!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Note: This is basic validation. Full validation requires Docker BuildKit." -ForegroundColor Yellow
    Write-Host "To fully test, start Docker Desktop and run: .\build-local.ps1" -ForegroundColor Cyan
} else {
    if ($errors.Count -gt 0) {
        Write-Host "❌ ERRORS FOUND:" -ForegroundColor Red
        foreach ($error in $errors) {
            Write-Host "  • $error" -ForegroundColor Red
        }
        Write-Host ""
    }
    
    if ($warnings.Count -gt 0) {
        Write-Host "⚠️  WARNINGS:" -ForegroundColor Yellow
        foreach ($warning in $warnings) {
            Write-Host "  • $warning" -ForegroundColor Yellow
        }
        Write-Host ""
    }
}

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Additional info
Write-Host "To fully validate with Docker:" -ForegroundColor Cyan
Write-Host "  1. Start Docker Desktop" -ForegroundColor White
Write-Host "  2. Run: .\build-local.ps1" -ForegroundColor White
Write-Host ""

if ($errors.Count -gt 0) {
    exit 1
} else {
    exit 0
}

