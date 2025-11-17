# Docker Dependency Caching Verification

This document explains how to verify that Python dependencies are properly cached and only rebuild when `backend/requirements.txt` changes.

## Setup

1. **Enable BuildKit** (required for caching to work):
   ```powershell
   # PowerShell
   .\enable-buildkit.ps1
   
   # Or manually:
   $env:DOCKER_BUILDKIT = "1"
   $env:COMPOSE_DOCKER_CLI_BUILD = "1"
   ```

2. **Verify BuildKit is enabled**:
   ```powershell
   docker buildx version
   ```

## Verification Steps

### Step 1: Initial Build (Populate Cache)

Build the backend image to populate the cache:
```powershell
docker-compose -f docker-compose.dev.yml build backend
```

**Expected**: The pip install step will run and download all packages (takes ~5-10 minutes).

### Step 2: Verify Cache Works (Code Change)

1. Make a small code change (e.g., add a comment to any `.py` file in `backend/`):
   ```python
   # Test comment for cache verification
   ```

2. Rebuild the backend:
   ```powershell
   docker-compose -f docker-compose.dev.yml build backend
   ```

3. **Check the build output** - you should see:
   ```
   => [backend dependencies 4/5] RUN --mount=type=cache,target=/root/.cache/pip     pip install --upgrade pip
   => CACHED [backend dependencies 4/5] ...
   => CACHED [backend dependencies 5/5] RUN --mount=type=cache,target=/root/.cache/pip     pip install -r /tmp/requirements.txt
   => CACHED [backend dependencies 5/5] ...
   ```

   The `CACHED` indicator means the layer was reused from cache and pip install did NOT run.

### Step 3: Verify Cache Invalidation (Requirements Change)

1. Make a small change to `backend/requirements.txt` (e.g., add a comment):
   ```
   # Test comment
   ```

2. Rebuild:
   ```powershell
   docker-compose -f docker-compose.dev.yml build backend
   ```

3. **Check the build output** - you should see:
   ```
   => [backend dependencies 3/5] COPY backend/requirements.txt /tmp/requirements.txt
   => [backend dependencies 3/5] ...
   => [backend dependencies 4/5] RUN --mount=type=cache,target=/root/.cache/pip     pip install --upgrade pip
   => CACHED [backend dependencies 4/5] ...
   => [backend dependencies 5/5] RUN --mount=type=cache,target=/root/.cache/pip     pip install -r /tmp/requirements.txt
   => [backend dependencies 5/5] ...
   ```

   The pip install step should run again because `requirements.txt` changed.

## Troubleshooting

### Cache Not Working?

1. **Verify BuildKit is enabled**:
   ```powershell
   echo $env:DOCKER_BUILDKIT
   # Should output: 1
   ```

2. **Check Docker version** (BuildKit requires Docker 18.09+):
   ```powershell
   docker --version
   ```

3. **Clear cache and rebuild**:
   ```powershell
   docker builder prune
   docker-compose -f docker-compose.dev.yml build --no-cache backend
   ```

4. **Verify Dockerfile structure**:
   - `COPY backend/requirements.txt /tmp/requirements.txt` should be BEFORE pip install
   - `COPY . .` should be AFTER the dependencies stage
   - All pip install commands should use `--mount=type=cache,target=/root/.cache/pip`

### Still Slow?

- The first build will always be slow (downloading packages)
- Subsequent builds with code changes should be fast (cached dependencies)
- If you see `CACHED` in the build output, caching is working correctly

## Expected Behavior

| Change Type | Pip Install Runs? | Build Time |
|------------|-------------------|------------|
| Code file change (`.py`) | ❌ No (CACHED) | ~10-30 seconds |
| `requirements.txt` change | ✅ Yes | ~5-10 minutes |
| First build | ✅ Yes | ~5-10 minutes |
| No changes | ❌ No (CACHED) | ~5-10 seconds |

## Dockerfile Structure

The Dockerfile is structured to maximize cache hits:

1. **Base stage**: System dependencies (rarely changes)
2. **Dependencies stage**: 
   - Copies ONLY `requirements.txt` (line 51)
   - Upgrades pip (line 55-56) - cached separately
   - Installs requirements (line 60-61) - only rebuilds when requirements.txt changes
3. **Final stage**:
   - Copies dependencies from dependencies stage (line 93-94)
   - Copies application code LAST (line 96) - doesn't affect dependency cache

This ensures that code changes don't invalidate the dependency installation cache.

