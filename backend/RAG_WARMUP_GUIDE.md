# RAG Warm-Up Guide

This guide explains how to use the RAG warm-up endpoints to keep the backend service warm and ready for queries.

## Overview

The RAG pipeline uses **lazy initialization** to enable fast backend startup. The pipeline only initializes when first needed (e.g., on the first `/query` call). This means:

- **Fast startup**: Backend becomes ready in seconds instead of minutes
- **On-demand initialization**: RAG loads when first query arrives
- **Warm-up support**: Cloud Scheduler can keep RAG initialized during active hours

## Endpoints

### `/rag/status`

Get current RAG pipeline status. Returns:
- `rag_enabled`: Whether RAG is ready
- `initializing`: Whether RAG is currently initializing
- `status`: "ready" | "initializing" | "disabled"
- `storage_dir`: Path to RAG index storage
- `last_error`: Last error message (if initialization failed)

**Usage**: Frontend calls this to check RAG availability before showing query UI.

### `/rag/self-test`

Performs a simple test query to verify RAG is working. Triggers lazy initialization if needed.

**Usage**: 
- Manual testing/debugging
- Cloud Scheduler warm-up (alternative to `/rag/warmup`)
- Health checks

**Returns**:
- `status`: "ok" | "RAG_NOT_INITIALIZED" | "ERROR"
- `rag_enabled`: Whether RAG is ready
- `num_results`: Number of results from test query

### `/rag/warmup` (Recommended for Cloud Scheduler)

Lightweight warm-up endpoint specifically designed for Cloud Scheduler.

**Security**: Requires `X-RAG-Warmup-Token` header matching `RAG_WARMUP_TOKEN` environment variable.

**Behavior**:
- Triggers lazy initialization if RAG is not initialized
- Returns current RAG status
- Does NOT perform test queries (faster than `/rag/self-test`)

**Returns**:
- `status`: "ok" | "error"
- `rag_enabled`: Whether RAG is ready
- `debug_status`: Full pipeline debug information

## Cloud Scheduler Configuration

### Step 1: Set Warm-Up Token

Set the `RAG_WARMUP_TOKEN` environment variable in Cloud Run:

```bash
gcloud run services update arrow-rag-backend \
  --set-env-vars=RAG_WARMUP_TOKEN=your-secret-token-here \
  --region=us-central1
```

Generate a secure token:
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

### Step 2: Create Cloud Scheduler Job

Create a job that calls `/rag/warmup` every 5 minutes during active hours:

```bash
gcloud scheduler jobs create http rag-warmup \
  --schedule="*/5 8-18 * * *" \
  --uri="https://arrow-rag-backend-70705019874.us-central1.run.app/rag/warmup" \
  --http-method=GET \
  --headers="X-RAG-Warmup-Token=your-secret-token-here" \
  --time-zone="America/New_York" \
  --location=us-central1
```

**Schedule explanation**:
- `*/5 8-18 * * *`: Every 5 minutes between 8 AM and 6 PM
- Adjust time zone and hours as needed

### Step 3: Verify Warm-Up Works

Check Cloud Scheduler logs:
```bash
gcloud scheduler jobs describe rag-warmup --location=us-central1
```

Check backend logs for warm-up events:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-backend AND jsonPayload.event=rag_warmup" --limit=10
```

## Alternative: Using `/rag/self-test`

If you prefer to use `/rag/self-test` instead (performs actual test query):

```bash
gcloud scheduler jobs create http rag-warmup-self-test \
  --schedule="*/5 8-18 * * *" \
  --uri="https://arrow-rag-backend-70705019874.us-central1.run.app/rag/self-test" \
  --http-method=GET \
  --time-zone="America/New_York" \
  --location=us-central1
```

**Note**: `/rag/self-test` does not require authentication, but performs a test query which is slightly slower.

## What Gets Initialized

When RAG initializes (lazy or via warm-up), it:

1. **Loads embedding models** from cache (`/app/.cache/huggingface`)
2. **Loads vector index** from storage (`/app/latest_model` or configured path)
3. **Initializes query pipeline** with all components

**Important**: No ingestion is performed. The index must already be built and uploaded to the storage location.

## Monitoring

### Check RAG Status

```bash
curl https://arrow-rag-backend-70705019874.us-central1.run.app/rag/status
```

### Check Warm-Up Endpoint

```bash
curl -H "X-RAG-Warmup-Token: your-secret-token" \
  https://arrow-rag-backend-70705019874.us-central1.run.app/rag/warmup
```

### View Backend Logs

```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-backend" --limit=50 --format=json
```

Look for events:
- `rag_warmup_triggering_init`: Warm-up started initialization
- `rag_warmup_init_success`: Warm-up completed successfully
- `rag_pipeline_lazy_init_success`: Lazy initialization completed

## Troubleshooting

### Warm-Up Returns 503

**Cause**: `RAG_WARMUP_TOKEN` not set or invalid token provided.

**Fix**: Set the environment variable and ensure Cloud Scheduler uses the correct header.

### RAG Never Initializes

**Cause**: Storage path not configured or index files missing.

**Fix**: 
1. Check `/rag/status` for `storage_dir` and `last_error`
2. Verify index files exist at storage path
3. Check backend logs for initialization errors

### Warm-Up Works But Queries Still Slow

**Cause**: Instance scaled to zero between warm-up calls.

**Fix**: 
1. Set `--min-instances=1` in Cloud Run to keep instance warm
2. Reduce warm-up interval (e.g., every 3 minutes instead of 5)

## Best Practices

1. **Use `/rag/warmup` for Cloud Scheduler**: Faster and more efficient than `/rag/self-test`
2. **Set appropriate schedule**: Warm up during active hours only (e.g., 8 AM - 6 PM)
3. **Keep token secure**: Use strong random token, never commit to git
4. **Monitor warm-up success**: Check logs regularly to ensure warm-ups are working
5. **Combine with `--min-instances=1`**: Prevents cold starts even if warm-up fails

## Cost Considerations

- **Warm-up calls**: ~1 request every 5 minutes = 288 requests/day
- **Cloud Scheduler**: Free tier includes 3 jobs, then $0.10/job/month
- **Cloud Run**: Warm-up requests use minimal CPU/memory (just status check)
- **Total cost**: Essentially free for typical usage

## Summary

Lazy RAG initialization enables fast backend startup while maintaining full functionality. Use Cloud Scheduler with `/rag/warmup` to keep RAG initialized during active hours, ensuring users always experience fast query responses.

