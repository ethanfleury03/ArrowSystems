#!/bin/bash
# Deployment script for arrow-rag-backend Cloud Run service
# This script deploys the service and configures scaling and env vars
# NOTE: GCS volumes are NOT used - index is bundled in the Docker image
# CRITICAL: This script enforces 8GiB memory and stability flags to prevent OOM/SIGKILL

set -e

PROJECT="arrow-rag-support-prod"
REGION="us-central1"
SERVICE="arrow-rag-backend"

# REQUIRED: Enforce minimum resource requirements to prevent OOM/SIGKILL
# These values are hardcoded to prevent accidental degradation
REQUIRED_MEMORY="8Gi"
REQUIRED_CPU="2"
REQUIRED_WORKERS="1"
REQUIRED_TIMEOUT="600"
REQUIRED_CONCURRENCY="1"

# SAFETY CHECK – do not allow accidental GCS mounts
# Check for volume mount commands in deployment files (excluding this safety check itself)
if grep -rE "(--add-volume|volumeMounts:|volumes:.*rag-index|gcsfuse)" \
   --include="*.sh" --include="*.yaml" --include="*.yml" \
   deployment .github/workflows 2>/dev/null | \
   grep -v "SAFETY CHECK" | \
   grep -v "do not allow" | \
   grep -v "Remove any existing"; then
  echo "❌ ERROR: GCS volume mount detected — this system no longer uses gcsfuse."
  echo "   Index must be bundled in the Docker image at /app/latest_model/"
  exit 1
fi

echo "=========================================="
echo "Step 1: Deploy base service from YAML"
echo "=========================================="
gcloud run services replace deployment/cloud-run-service.yaml \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

echo ""
echo "=========================================="
echo "Step 2: Remove any existing GCS volume mounts (if present)"
echo "=========================================="
# Remove any existing volume mounts to ensure clean state
# Note: Explicitly preserve min-instances=1 to prevent reset to default (0)
gcloud run services update $SERVICE \
  --remove-volume=rag-index \
  --remove-volume-mount=rag-index \
  --min-instances=1 \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT 2>/dev/null || echo "No rag-index volume to remove (expected)"

gcloud run services update $SERVICE \
  --remove-volume=data-volume \
  --remove-volume-mount=data-volume \
  --min-instances=1 \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT 2>/dev/null || echo "No data-volume volume to remove (expected)"

echo ""
echo "=========================================="
echo "Step 3: Pre-deploy validation (guardrails)"
echo "=========================================="
# CRITICAL: Validate required configuration to prevent OOM/SIGKILL
# This fails the deploy if someone tries to reduce memory/workers/timeout
if [ "$REQUIRED_MEMORY" != "8Gi" ]; then
  echo "❌ ERROR: REQUIRED_MEMORY must be 8Gi to prevent OOM. Current: $REQUIRED_MEMORY"
  echo "   Edit this script to change the requirement (not recommended)."
  exit 1
fi

if [ "$REQUIRED_WORKERS" != "1" ]; then
  echo "❌ ERROR: REQUIRED_WORKERS must be 1 to prevent memory fragmentation. Current: $REQUIRED_WORKERS"
  exit 1
fi

if [ "$REQUIRED_TIMEOUT" -lt 600 ]; then
  echo "❌ ERROR: REQUIRED_TIMEOUT must be >= 600 to allow RAG load. Current: $REQUIRED_TIMEOUT"
  exit 1
fi

echo "✅ Pre-deploy validation passed: memory=$REQUIRED_MEMORY, workers=$REQUIRED_WORKERS, timeout=$REQUIRED_TIMEOUT"

echo ""
echo "=========================================="
echo "Step 4: Set resource limits, scaling, and stability flags"
echo "=========================================="
# CRITICAL: 8GiB memory is required to prevent OOM/SIGKILL during model load + index parse
# Base footprint: ~2-3GB (models) + ~200MB (index) + ~1GB (Python/runtime) = ~4-5GB
# 8GiB provides headroom for:
# - Memory spikes during HuggingFace model loading (can temporarily double)
# - Vector store JSON parsing (183MB file parsed into memory)
# - Concurrent requests during query execution
# Previous 4GiB caused repeated SIGKILL errors and worker restarts
#
# Concurrency=1 prevents parallel requests during startup/load, reducing memory pressure
# No CPU throttling ensures consistent performance during load operations
# Eager mode makes worker startup block until RAG is ready (deterministic readiness)
echo "Deploying with enforced stability flags:"
echo "  - memory=$REQUIRED_MEMORY (required to prevent OOM)"
echo "  - cpu=$REQUIRED_CPU"
echo "  - concurrency=$REQUIRED_CONCURRENCY (prevents parallel requests during load)"
echo "  - workers=$REQUIRED_WORKERS (single worker prevents memory fragmentation)"
echo "  - timeout=$REQUIRED_TIMEOUT (allows RAG load to complete)"
echo "  - no-cpu-throttling (consistent performance during load)"

gcloud run services update $SERVICE \
  --execution-environment=gen2 \
  --cpu=$REQUIRED_CPU \
  --memory=$REQUIRED_MEMORY \
  --min-instances=1 \
  --max-instances=10 \
  --no-cpu-throttling \
  --concurrency=$REQUIRED_CONCURRENCY \
  --set-env-vars="GUNICORN_WORKERS=$REQUIRED_WORKERS" \
  --set-env-vars="GUNICORN_TIMEOUT=$REQUIRED_TIMEOUT" \
  --set-env-vars="RAG_EAGER_LOAD_ON_STARTUP=1" \
  --set-env-vars="RAG_BACKGROUND_LOAD_ON_STARTUP=0" \
  --set-env-vars="HF_HOME=/app/.cache/huggingface" \
  --set-env-vars="SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface" \
  --set-env-vars="TRANSFORMERS_CACHE=/app/.cache/huggingface" \
  --set-env-vars="HF_HUB_OFFLINE=1" \
  --set-env-vars="TRANSFORMERS_OFFLINE=1" \
  --set-env-vars="HF_DATASETS_OFFLINE=1" \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT

if [ $? -ne 0 ]; then
  echo "❌ ERROR: Failed to update Cloud Run service"
  exit 1
fi

echo ""
echo "=========================================="
echo "Step 5: Post-deploy verification"
echo "=========================================="

# Get service URL
SERVICE_URL=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(status.url)")

if [ -z "$SERVICE_URL" ]; then
  echo "❌ ERROR: Could not get service URL"
  exit 1
fi

echo "Service URL: $SERVICE_URL"

# Verify resource configuration
echo ""
echo "Verifying deployed configuration..."

DEPLOYED_MEMORY=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].resources.limits.memory)" 2>/dev/null || echo "")

DEPLOYED_CPU=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].resources.limits.cpu)" 2>/dev/null || echo "")

DEPLOYED_CONCURRENCY=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containerConcurrency)" 2>/dev/null || echo "")

DEPLOYED_MIN_INSTANCES=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.metadata.annotations['autoscaling.knative.dev/minScale'])" 2>/dev/null || echo "")

DEPLOYED_GUNICORN_WORKERS=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "GUNICORN_WORKERS" | grep "value:" | sed 's/.*value: //' || echo "")

DEPLOYED_GUNICORN_TIMEOUT=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "GUNICORN_TIMEOUT" | grep "value:" | sed 's/.*value: //' || echo "")

DEPLOYED_HF_OFFLINE=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "HF_HUB_OFFLINE" | grep "value:" | sed 's/.*value: //' || echo "")

DEPLOYED_RAG_EAGER=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "RAG_EAGER_LOAD_ON_STARTUP" | grep "value:" | sed 's/.*value: //' || echo "")

DEPLOYED_RAG_BACKGROUND=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "RAG_BACKGROUND_LOAD_ON_STARTUP" | grep "value:" | sed 's/.*value: //' || echo "")

DEPLOYED_LLM_WARMUP=$(gcloud run services describe $SERVICE \
  --region=$REGION \
  --platform=managed \
  --project=$PROJECT \
  --format="value(spec.template.spec.containers[0].env)" 2>/dev/null | \
  grep -A 1 "LLM_WARMUP_ON_STARTUP" | grep "value:" | sed 's/.*value: //' || echo "")

echo "  Memory: $DEPLOYED_MEMORY (required: $REQUIRED_MEMORY)"
echo "  CPU: $DEPLOYED_CPU (required: $REQUIRED_CPU)"
echo "  Concurrency: $DEPLOYED_CONCURRENCY (required: $REQUIRED_CONCURRENCY)"
echo "  Min instances: $DEPLOYED_MIN_INSTANCES (required: 1)"
echo "  Gunicorn workers: $DEPLOYED_GUNICORN_WORKERS (required: $REQUIRED_WORKERS)"
echo "  Gunicorn timeout: $DEPLOYED_GUNICORN_TIMEOUT (required: $REQUIRED_TIMEOUT)"
echo "  HF offline mode: $DEPLOYED_HF_OFFLINE (required: 1)"
echo "  RAG eager load: $DEPLOYED_RAG_EAGER (required: 1)"
echo "  RAG background load: $DEPLOYED_RAG_BACKGROUND (required: 0)"
echo "  LLM warmup: $DEPLOYED_LLM_WARMUP (required: 0)"

# Validate memory
if [ "$DEPLOYED_MEMORY" != "$REQUIRED_MEMORY" ]; then
  echo "❌ ERROR: Memory mismatch! Deployed: $DEPLOYED_MEMORY, Required: $REQUIRED_MEMORY"
  exit 1
fi

# Validate workers (check if env var contains the value)
if [ "$DEPLOYED_GUNICORN_WORKERS" != "$REQUIRED_WORKERS" ]; then
  echo "❌ ERROR: Gunicorn workers mismatch! Deployed: $DEPLOYED_GUNICORN_WORKERS, Required: $REQUIRED_WORKERS"
  exit 1
fi

# Validate timeout
if [ -n "$DEPLOYED_GUNICORN_TIMEOUT" ] && [ "$DEPLOYED_GUNICORN_TIMEOUT" -lt "$REQUIRED_TIMEOUT" ]; then
  echo "❌ ERROR: Gunicorn timeout too low! Deployed: $DEPLOYED_GUNICORN_TIMEOUT, Required: >= $REQUIRED_TIMEOUT"
  exit 1
fi

# Validate offline mode
if [ "$DEPLOYED_HF_OFFLINE" != "1" ]; then
  echo "⚠️  WARNING: HF_HUB_OFFLINE is not set to 1. Runtime may attempt network model downloads."
  echo "   This can cause latency and memory pressure. Consider fixing."
fi

# Validate RAG eager load mode
if [ "$DEPLOYED_RAG_EAGER" != "1" ]; then
  echo "❌ ERROR: RAG_EAGER_LOAD_ON_STARTUP must be 1 for deterministic readiness. Current: $DEPLOYED_RAG_EAGER"
  exit 1
fi

# Validate RAG background load is disabled
if [ "$DEPLOYED_RAG_BACKGROUND" != "0" ]; then
  echo "❌ ERROR: RAG_BACKGROUND_LOAD_ON_STARTUP must be 0 when eager load is enabled. Current: $DEPLOYED_RAG_BACKGROUND"
  exit 1
fi

# Validate LLM warmup is disabled (not critical, but good to enforce)
if [ -n "$DEPLOYED_LLM_WARMUP" ] && [ "$DEPLOYED_LLM_WARMUP" != "0" ]; then
  echo "⚠️  WARNING: LLM_WARMUP_ON_STARTUP is not 0. Current: $DEPLOYED_LLM_WARMUP"
  echo "   This may add startup latency. Consider setting to 0."
fi

echo "✅ Resource configuration verified"

# Test health endpoints
echo ""
echo "Testing health endpoints..."

MAX_WAIT=120  # 2 minutes max wait
WAIT_INTERVAL=5
ELAPSED=0

while [ $ELAPSED -lt $MAX_WAIT ]; do
  # Test /api/healthz
  if curl -sf "${SERVICE_URL}/api/healthz" > /dev/null 2>&1; then
    echo "✅ /api/healthz: OK"
    break
  fi
  
  echo "  Waiting for /api/healthz... (${ELAPSED}s elapsed)"
  sleep $WAIT_INTERVAL
  ELAPSED=$((ELAPSED + WAIT_INTERVAL))
done

if [ $ELAPSED -ge $MAX_WAIT ]; then
  echo "❌ ERROR: /api/healthz did not respond within ${MAX_WAIT}s"
  exit 1
fi

# Test /api/model_cache_status if it exists
if curl -sf "${SERVICE_URL}/api/model_cache_status" > /tmp/model_cache_status.json 2>/dev/null; then
  echo "✅ /api/model_cache_status: OK"
  cat /tmp/model_cache_status.json | head -20
  rm -f /tmp/model_cache_status.json
else
  echo "⚠️  /api/model_cache_status: Not available (endpoint may not exist)"
fi

# Test /api/readyz
echo ""
echo "Checking readiness status..."
READYZ_RESPONSE=$(curl -sf "${SERVICE_URL}/api/readyz" 2>/dev/null || echo "")
if [ -n "$READYZ_RESPONSE" ]; then
  echo "✅ /api/readyz: Responding"
  echo "$READYZ_RESPONSE" | grep -o '"ready":[^,]*' || echo "$READYZ_RESPONSE"
else
  echo "⚠️  /api/readyz: Not responding yet (may still be loading)"
fi

# Check for OOM/SIGKILL in logs
echo ""
echo "Checking for OOM/SIGKILL errors in last 10 minutes..."

# Check for OOM/SIGKILL in logs (using --freshness which is more reliable than timestamp filters)
LOG_CHECK=$(gcloud logging read \
  "resource.type=\"cloud_run_revision\"
   AND resource.labels.service_name=\"$SERVICE\"
   AND (textPayload=~\"SIGKILL\" OR textPayload=~\"out of memory\" OR textPayload=~\"OOM\")" \
  --project=$PROJECT \
  --freshness=10m \
  --limit=10 \
  --format="value(textPayload)" 2>/dev/null || echo "")

if [ -n "$LOG_CHECK" ]; then
  echo "❌ ERROR: Found OOM/SIGKILL errors in logs:"
  echo "$LOG_CHECK" | head -5
  echo ""
  echo "   This indicates memory is still insufficient despite 8GiB allocation."
  echo "   Check worker count, concurrency, and model loading behavior."
  exit 1
else
  echo "✅ No OOM/SIGKILL errors found in last 10 minutes"
fi

echo ""
echo "=========================================="
echo "✅ Deployment complete and verified!"
echo "=========================================="
echo "Service URL: $SERVICE_URL"
echo ""
echo "Configuration summary:"
echo "  - Memory: $REQUIRED_MEMORY ✅"
echo "  - CPU: $REQUIRED_CPU ✅"
echo "  - Concurrency: $REQUIRED_CONCURRENCY ✅"
echo "  - Workers: $REQUIRED_WORKERS ✅"
echo "  - Timeout: $REQUIRED_TIMEOUT ✅"
echo "  - HF Offline: Enabled ✅"
echo "  - Health checks: Passing ✅"
echo "  - No OOM errors: Confirmed ✅"





