# Windows Line Ending Fix Applied

## Problem
The build failed with:
```
chmod: cannot access '/app/start.sh'$'\r': No such file or directory
```

This occurs because Windows line endings (CRLF) were being preserved in the heredoc, causing `chmod` to fail.

## Solution Applied

1. **Added `set -e`** at the start of RUN blocks to catch errors early
2. **Use `tr -d '\r'`** to strip carriage returns before `chmod`
3. **Fixed casing warnings** - Changed `as` to `AS` to match `FROM` casing

## Changes Made

### Before:
```dockerfile
RUN <<'EOFSCRIPT' bash
cat > /app/start.sh <<'EOF'
...
EOF
chmod +x /app/start.sh
EOFSCRIPT
```

### After:
```dockerfile
RUN <<'EOFSCRIPT' bash
set -e
cat > /app/start.sh <<'EOF'
...
EOF
# Ensure Unix line endings and set permissions
tr -d '\r' < /app/start.sh > /app/start.sh.tmp && mv /app/start.sh.tmp /app/start.sh
chmod +x /app/start.sh
EOFSCRIPT
```

## Why This Works

- `tr -d '\r'` removes Windows carriage returns (`\r`)
- Creates a temporary file, then replaces the original
- Ensures `chmod` operates on a file with Unix line endings (LF only)
- Applied to both `start.sh` and `healthcheck.sh`

## Additional Fixes

- Fixed casing: `FROM ... as` → `FROM ... AS` (matches Docker best practices)
- Added `set -e` for better error handling

The Dockerfile should now build successfully on Windows!


