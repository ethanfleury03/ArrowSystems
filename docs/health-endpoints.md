# Health Check Endpoints

## Overview

The backend provides two health check endpoints for different purposes:

- **`/healthz`** (and `/api/healthz`): Liveness check - indicates the service is running
- **`/readyz`** (and `/api/readyz`): Readiness check - indicates the service is ready to serve requests

## `/healthz` - Liveness Check

**Purpose**: Verify the FastAPI server is running and can respond to requests.

**Characteristics**:
- Zero dependencies (no database, no RAG, no models)
- Returns immediately (< 10ms typically)
- Always returns HTTP 200 when the server is running
- Unauthenticated (public endpoint)
- Not rate limited

**Response**:
```json
{
  "status": "ok",
  "service": "arrow-rag-backend",
  "timestamp": 1234567890.123,
  "pid": 12345,
  "revision": "revision-abc123"
}
```

**Use Cases**:
- Cloud Run health checks
- CI/CD pipeline verification
- Load balancer health checks
- Kubernetes liveness probes

**Important**: This endpoint will return 200 even if:
- RAG index is still loading
- Database is unavailable
- Models are not loaded
- Any other heavy dependencies are not ready

## `/readyz` - Readiness Check

**Purpose**: Verify the service is ready to serve queries (RAG index loaded).

**Characteristics**:
- Quick read-only state check (non-blocking)
- Returns HTTP 200 only when RAG index is ready
- Returns HTTP 503 when index is loading, failed, or not started
- Unauthenticated (public endpoint)
- Not rate limited

**Response (Ready)**:
```json
{
  "ready": true,
  "rag_state": "ready",
  "detail": "RAG index is ready to serve queries"
}
```

**Response (Not Ready)**:
```json
{
  "ready": false,
  "rag_state": "loading",
  "detail": "Index is currently loading"
}
```

**Use Cases**:
- Kubernetes readiness probes
- Monitoring dashboards
- Pre-flight checks before sending queries

## CI/CD Integration

The GitHub Actions workflow uses `/api/healthz` for deployment verification:

- **Timeout**: 30 seconds per attempt
- **Total wait**: Up to 10 minutes
- **Retry interval**: 2 seconds between attempts
- **Authentication**: Tries without auth first, falls back to identity token if needed

The health check includes comprehensive diagnostics on failure:
- DNS resolution test
- TCP connection test
- Last HTTP code and curl exit code
- Error messages

## Local Testing

Test the endpoints locally:

```bash
# Liveness check (should always return 200)
curl http://localhost:8000/api/healthz

# Readiness check (returns 200 when RAG is ready, 503 otherwise)
curl http://localhost:8000/api/readyz
```

## RAG Index Loading Configuration

The backend supports two modes for loading the RAG index at startup:

### Eager Load (Recommended for Production)

**Configuration**: Set `RAG_EAGER_LOAD_ON_STARTUP=1` and `RAG_BACKGROUND_LOAD_ON_STARTUP=0`

**Behavior**:
- Index is loaded synchronously during startup
- Startup does not complete until index is loaded or timeout is reached
- `/api/readyz` will return 200 once startup completes (if load succeeded)
- More reliable than background loading in Cloud Run environments
- Recommended for production deployments

**Timeout**: Configurable via `RAG_EAGER_STARTUP_TIMEOUT_SEC` (default: 600s, falls back to `RAG_MAX_LOAD_TIME_SEC`)

**Expected Log Sequence**:
```
[RESOURCE] rag_eager_load_start
rag_index_load_* checkpoints (download + load)
[RESOURCE] rag_index_loaded
[RESOURCE] rag_eager_load_success
```

### Background Load (Default)

**Configuration**: Set `RAG_BACKGROUND_LOAD_ON_STARTUP=1` (or leave unset, defaults to 1)

**Behavior**:
- Index loads in background after startup completes
- Startup completes quickly, but `/api/readyz` returns 503 until load completes
- `/query` returns 503 with Retry-After header while loading
- May be less reliable in Cloud Run due to background task throttling

### Lazy Load

**Configuration**: Set both `RAG_EAGER_LOAD_ON_STARTUP=0` and `RAG_BACKGROUND_LOAD_ON_STARTUP=0`

**Behavior**:
- Index loads on first `/query` request
- Startup completes immediately
- First query may be slow

## Deployment Configuration

To enable eager load in production:

```bash
gcloud run services update arrow-rag-backend \
  --set-env-vars="RAG_EAGER_LOAD_ON_STARTUP=1" \
  --set-env-vars="RAG_BACKGROUND_LOAD_ON_STARTUP=0" \
  --region=us-central1
```

## Verification Commands

After deployment, verify the configuration:

### 1. Confirm env vars are set in Cloud Run:

```bash
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="yaml(spec.template.spec.containers[0].env)" | \
  grep -E "RAG_EAGER_LOAD_ON_STARTUP|RAG_BACKGROUND_LOAD_ON_STARTUP"
```

Expected output should show:
```yaml
- name: RAG_EAGER_LOAD_ON_STARTUP
  value: "1"
- name: RAG_BACKGROUND_LOAD_ON_STARTUP
  value: "0"
```

### 2. Confirm backend reports correct mode:

```bash
curl -i -H "Authorization: Bearer $(gcloud auth print-identity-token --audiences=$BACKEND_URL)" \
  $BACKEND_URL/api/rag_mode
```

Expected response should show `"mode": "eager"` and matching env values.

### 3. Confirm eager semantics:

- During cold start, `/api/readyz` should NOT respond quickly if eager is enabled; it should be unreachable until startup completes.
- Once it responds, it should be 200 (not 503).

If `/readyz` responds quickly with 503 while "eager" is supposedly enabled, then startup is not blocking and there's a bug.

## Security Notes

- Both endpoints are intentionally unauthenticated for health check compatibility
- They do not expose sensitive information
- They are not rate limited to allow frequent health checks
- Other endpoints remain protected by authentication as configured

