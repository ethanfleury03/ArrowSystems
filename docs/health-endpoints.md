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

## Security Notes

- Both endpoints are intentionally unauthenticated for health check compatibility
- They do not expose sensitive information
- They are not rate limited to allow frequent health checks
- Other endpoints remain protected by authentication as configured

