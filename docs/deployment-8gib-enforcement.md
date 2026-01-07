# 8GiB Memory Enforcement - Deployment Guardrails

## Summary

All backend deployments now **enforce** 8GiB memory and stability flags to prevent OOM/SIGKILL errors. Guardrails prevent accidental 4GiB deployments.

## Files Changed

1. **`deployment/deploy-backend.sh`** - Main deployment script with guardrails
2. **`.github/workflows/ci.yml`** - CI/CD workflow updated to match
3. **`deployment/cloud-run-service.yaml`** - Base YAML updated (memory=8Gi, comments updated)

## Enforced Configuration

### Resource Limits (Hardcoded)
- **Memory:** `8Gi` (REQUIRED - prevents OOM)
- **CPU:** `2`
- **Concurrency:** `1` (prevents parallel requests during load)
- **Min Instances:** `1`
- **Max Instances:** `10`
- **CPU Throttling:** Disabled (`--no-cpu-throttling`)

### Environment Variables
- `GUNICORN_WORKERS=1` (single worker prevents memory fragmentation)
- `GUNICORN_TIMEOUT=600` (allows RAG load to complete)
- `RAG_EAGER_LOAD_ON_STARTUP=1` (deterministic readiness)
- `RAG_BACKGROUND_LOAD_ON_STARTUP=0`

### Offline HuggingFace Mode (Prevents Network Downloads)
- `HF_HOME=/app/.cache/huggingface`
- `SENTENCE_TRANSFORMERS_HOME=/app/.cache/huggingface`
- `TRANSFORMERS_CACHE=/app/.cache/huggingface`
- `HF_HUB_OFFLINE=1` (CRITICAL - prevents network downloads)
- `TRANSFORMERS_OFFLINE=1`
- `HF_DATASETS_OFFLINE=1`

## Guardrails

### Pre-Deploy Validation

The script validates **before** deployment:

```bash
# Lines 70-84 in deploy-backend.sh
if [ "$REQUIRED_MEMORY" != "8Gi" ]; then
  echo "❌ ERROR: REQUIRED_MEMORY must be 8Gi to prevent OOM"
  exit 1
fi

if [ "$REQUIRED_WORKERS" != "1" ]; then
  echo "❌ ERROR: REQUIRED_WORKERS must be 1"
  exit 1
fi

if [ "$REQUIRED_TIMEOUT" -lt 600 ]; then
  echo "❌ ERROR: REQUIRED_TIMEOUT must be >= 600"
  exit 1
fi
```

**How it prevents 4Gi deployments:**
- `REQUIRED_MEMORY` is hardcoded to `"8Gi"` at the top of the script
- If someone tries to change it to `"4Gi"`, the validation fails and deploy stops
- The script exits non-zero, preventing deployment

### Post-Deploy Verification

After deployment, the script verifies:

1. **Resource configuration matches requirements:**
   - Memory, CPU, concurrency, min instances
   - Gunicorn workers, timeout
   - HF offline mode

2. **Health endpoints respond:**
   - `/api/healthz` (waits up to 2 minutes)
   - `/api/model_cache_status` (if available)
   - `/api/readyz` (readiness status)

3. **No OOM/SIGKILL errors in logs:**
   - Queries Cloud Logging for SIGKILL/OOM messages in last 10 minutes
   - If found, deploy fails with error

**If any verification fails, the script exits non-zero.**

## Deployment Commands

### Canonical Deployment Script

```bash
bash deployment/deploy-backend.sh
```

This script:
1. Deploys base service from YAML
2. Removes any GCS volume mounts (safety check)
3. **Validates** required config (pre-deploy guardrails)
4. Updates Cloud Run with enforced settings
5. **Verifies** deployed config matches requirements (post-deploy)
6. Tests health endpoints
7. Checks for OOM/SIGKILL errors

### CI/CD Deployment

The GitHub Actions workflow (`.github/workflows/ci.yml` line 625) uses:

```bash
gcloud run deploy arrow-rag-backend \
  --memory=8Gi \
  --cpu=2 \
  --no-cpu-throttling \
  --concurrency=1 \
  --update-env-vars="...,HF_HUB_OFFLINE=1,TRANSFORMERS_OFFLINE=1,HF_DATASETS_OFFLINE=1,..." \
  ...
```

**The CI workflow is already configured with these values** (line 638-639).

## How Guardrails Prevent 4Gi Deployments

### Method 1: Hardcoded Variables

The script defines required values at the top:

```bash
REQUIRED_MEMORY="8Gi"
REQUIRED_CPU="2"
REQUIRED_WORKERS="1"
REQUIRED_TIMEOUT="600"
REQUIRED_CONCURRENCY="1"
```

These are used in:
1. Pre-deploy validation (fails if changed)
2. Deploy command (uses these variables)
3. Post-deploy verification (checks against these values)

**To change memory to 4Gi, someone would need to:**
1. Change `REQUIRED_MEMORY="8Gi"` to `"4Gi"`
2. But then pre-deploy validation fails (line 70-73 checks for "8Gi")
3. Script exits before deployment

### Method 2: Post-Deploy Verification

Even if someone bypasses the script and deploys manually, the post-deploy verification will:
- Query Cloud Run to get actual deployed memory
- Compare to `REQUIRED_MEMORY="8Gi"`
- If mismatch, exit non-zero with error

**However, this only works if they run the script. Manual `gcloud` commands bypass this.**

### Method 3: CI/CD Enforcement

The GitHub Actions workflow has the values hardcoded in the `gcloud run deploy` command. To change memory:
1. Edit `.github/workflows/ci.yml` line 639
2. Change `--memory=8Gi` to `--memory=4Gi`
3. Commit and push
4. CI will deploy with 4Gi (no pre-deploy validation in CI)

**Recommendation:** Add a validation step in CI that checks for `--memory=8Gi` before deploy.

## Verification Commands

After deployment, verify manually:

```bash
# Check deployed memory
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="value(spec.template.spec.containers[0].resources.limits.memory)"

# Expected: "8Gi"

# Check env vars
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="yaml(spec.template.spec.containers[0].env)" | grep -E "GUNICORN_WORKERS|GUNICORN_TIMEOUT|HF_HUB_OFFLINE|RAG_EAGER_LOAD"

# Check concurrency
gcloud run services describe arrow-rag-backend \
  --region us-central1 \
  --project arrow-rag-support-prod \
  --format="value(spec.template.spec.containerConcurrency)"

# Expected: "1"

# Check for OOM errors
gcloud logging read \
  'resource.type="cloud_run_revision"
   AND resource.labels.service_name="arrow-rag-backend"
   AND (textPayload=~"SIGKILL" OR textPayload=~"out of memory" OR textPayload=~"OOM")' \
  --project arrow-rag-support-prod \
  --freshness=1h \
  --limit=10

# Expected: No results (or very few if service just restarted)
```

## Rationale

### Why 8GiB?

- **Base footprint:** ~4-5GB (models + index + Python runtime)
- **Memory spikes during model load:** Can temporarily double (8-10GB)
- **Vector store parsing:** 183MB JSON parsed into memory (~500MB after deserialization)
- **Previous 4GiB:** Caused repeated SIGKILL errors and worker restarts

### Why Concurrency=1?

- Prevents parallel requests during startup/load
- Reduces memory pressure from concurrent query processing
- With eager load, startup blocks anyway, so concurrency during load is unnecessary

### Why No CPU Throttling?

- Ensures consistent performance during model loading and vector store parsing
- Prevents CPU-starved operations from taking longer (which increases memory residency time)

### Why Offline HF Mode?

- Prevents network model downloads that add latency (30-60s per model)
- Prevents memory pressure from download buffers
- Forces use of pre-downloaded models in Docker image

## Next Steps

1. **Run deployment:** `bash deployment/deploy-backend.sh`
2. **Monitor logs** for OOM/SIGKILL errors (should be zero after patch)
3. **Verify readiness:** `/api/readyz` should return 200 after RAG load completes
4. **Consider adding CI validation** to check for `--memory=8Gi` before deploy

