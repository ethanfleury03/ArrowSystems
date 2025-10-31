# Quick Reference: Setting Environment Variables

## For Claude API Key

**PowerShell (one-time per session):**
```powershell
$env:ANTHROPIC_API_KEY = "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"
```

**To make it permanent (add to PowerShell profile):**
```powershell
# Check if profile exists
Test-Path $PROFILE

# If not, create it
New-Item -Path $PROFILE -Type File -Force

# Add the line to your profile
Add-Content $PROFILE '$env:ANTHROPIC_API_KEY = "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"'
```

## Order of Operations

1. **Set API key** (if needed)
   ```powershell
   $env:ANTHROPIC_API_KEY = "your-key"
   ```

2. **Rebuild image** (after Dockerfile changes)
   ```powershell
   .\build-local.ps1
   ```

3. **Run container**
   ```powershell
   .\run-local.ps1
   ```

## Note

The errors you're seeing (`/root/.cache/huggingface/` permission denied) are from the **old container** that was built before the fixes. You **must rebuild** the image for the fixes to take effect!


