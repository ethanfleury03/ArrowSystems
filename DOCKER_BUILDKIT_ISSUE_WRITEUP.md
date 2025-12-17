# Docker BuildKit Pull Issue - Situation Write-up

## Context

**Repository**: ArrowSystems (FastAPI backend + Next.js frontend on Cloud Run)
**CI/CD**: GitHub Actions workflow (`.github/workflows/ci.yml`)
**Issue**: Docker builds are failing because `docker/setup-buildx-action@v3` is trying to pull BuildKit image from Docker Hub, which is experiencing 500 errors.

## What Was Working Before

Previously, the workflow was using `docker/setup-buildx-action@v3` in a simple configuration:

```yaml
- name: Set up Docker Buildx
  uses: docker/setup-buildx-action@v3
```

This worked fine because:
1. GitHub Actions runners already have Docker and BuildKit pre-installed
2. The action would use the existing BuildKit on the runner
3. No external image pulls were required for BuildKit setup
4. Builds completed successfully

## What Changed

In an attempt to make the workflow more resilient to Docker Hub outages, the following changes were made:

1. **Added `buildkitd-config-inline`** to configure a docker.io mirror:
   ```yaml
   buildkitd-config-inline: |
     [registry."docker.io"]
       mirrors = ["mirror.gcr.io"]
   ```

2. **Added `driver-opts: network=host`** (attempted fix)

3. **Removed `# syntax=docker/dockerfile:1`** from Dockerfiles (this was correct - prevents Dockerfile frontend pulls)

4. **Added retry logic** for build steps (this was correct)

## The Problem

After these changes, the workflow started failing with:

```
Error: ERROR: Error response from daemon: Head "https://registry-1.docker.io/v2/moby/buildkit/manifests/buildx-stable-1": received unexpected HTTP status: 500 Internal Server Error
```

**Root Cause**: The `docker/setup-buildx-action@v3` action is now trying to pull the BuildKit image (`moby/buildkit:buildx-stable-1`) from Docker Hub, even though:
- GitHub Actions runners already have BuildKit installed
- This pull was NOT happening before
- The action should be able to use the existing BuildKit on the runner

## Why This Is Happening

The `docker/setup-buildx-action@v3` action has different behavior depending on configuration:
- **Default behavior**: Uses existing BuildKit on runner (what was working before)
- **With custom `buildkitd-config-inline`**: May trigger a BuildKit image pull from Docker Hub
- **With `driver-opts`**: Can change how BuildKit is initialized

The addition of `buildkitd-config-inline` or `driver-opts` may have changed the action's behavior to pull the BuildKit image instead of using the runner's existing BuildKit.

## What We Need

1. **Revert to the original simple configuration** that was working:
   ```yaml
   - name: Set up Docker Buildx
     uses: docker/setup-buildx-action@v3
   ```

2. **Keep the beneficial changes**:
   - Removed `# syntax=docker/dockerfile:1` from Dockerfiles (prevents Dockerfile frontend pulls)
   - Retry logic for build steps (handles transient failures)
   - Mirror configuration can stay, but only if it doesn't trigger BuildKit pulls

3. **Alternative approach for mirror** (if needed):
   - Configure the mirror at the Docker daemon level instead of in buildkitd-config
   - Or use the mirror only for base images, not for BuildKit

## Current State

**Backend Buildx Setup** (line ~218):
```yaml
- name: Set up Docker Buildx
  id: buildx
  uses: docker/setup-buildx-action@v3
  with:
    driver-opts: network=host
    buildkitd-config-inline: |
      [registry."docker.io"]
        mirrors = ["mirror.gcr.io"]
```

**Frontend Buildx Setup** (line ~1060):
```yaml
- name: Set up Docker Buildx
  id: buildx-frontend
  uses: docker/setup-buildx-action@v3
  with:
    driver-opts: network=host
    buildkitd-config-inline: |
      [registry."docker.io"]
        mirrors = ["mirror.gcr.io"]
```

## Desired Outcome

1. **Restore the original simple Buildx setup** that doesn't pull BuildKit from Docker Hub
2. **Keep the Dockerfile syntax directive removal** (this was correct)
3. **Keep retry logic** for build steps (this was correct)
4. **Optionally configure mirror for base images only** (if Docker Hub outages are a concern for base images)

## Key Insight

The original configuration was working because it used the BuildKit that's already installed on GitHub Actions runners. The addition of `buildkitd-config-inline` appears to have changed the action's behavior to pull a fresh BuildKit image from Docker Hub, which is unnecessary and causes failures when Docker Hub is down.

## Files to Fix

- `.github/workflows/ci.yml` - Two `docker/setup-buildx-action@v3` steps (backend and frontend)
- `backend/Dockerfile.backend` - Already fixed (syntax directive removed - keep this)

## Questions to Resolve

1. Can we use `buildkitd-config-inline` without triggering BuildKit image pulls?
2. Is there a way to configure the docker.io mirror without affecting BuildKit initialization?
3. Should we configure the mirror at a different level (Docker daemon config) instead?

## Summary

**Before**: Simple `docker/setup-buildx-action@v3` → Used existing BuildKit → Worked fine
**After**: Added `buildkitd-config-inline` → Action pulls BuildKit from Docker Hub → Fails when Docker Hub is down
**Solution**: Revert to simple configuration, keep Dockerfile and retry improvements, find alternative way to configure mirror if needed.

