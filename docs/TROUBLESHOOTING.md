# Cloud Run Deployment Troubleshooting

## Common Issues and Solutions

### 1. JWT_SECRET_KEY Missing Error

**Error:**
```
RuntimeError: JWT_SECRET_KEY environment variable is required in production
```

**Solution:**
```bash
# Generate a secure JWT secret
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Update Cloud Run service
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --update-env-vars "ENV=prod" \
  --update-env-vars "JWT_SECRET_KEY=<your-generated-secret>" \
  --update-env-vars "CORS_ALLOWED_ORIGINS=https://support.arrowsystems.com"
```

**Prevention:**
Make sure these environment variables are set before deploying:
- `JWT_SECRET_KEY` (32+ characters, random)
- `CORS_ALLOWED_ORIGINS` (comma-separated frontend URLs)
- `ENV=prod` (not APP_ENV)

### 2. CORS Errors

**Error:**
```
CORS policy: No 'Access-Control-Allow-Origin' header
```

**Solution:**
Ensure `CORS_ALLOWED_ORIGINS` includes your frontend URL:
```bash
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --update-env-vars "CORS_ALLOWED_ORIGINS=https://support.arrowsystems.com,https://app.arrowsystems.com"
```

**Note:** Never use `*` wildcard in production.

### 3. HuggingFace Cache Warnings

**Warning:**
```
There was a problem when trying to write in your cache folder (/tmp/hf)
```

**Status:** This is NORMAL and HARMLESS in Cloud Run's read-only filesystem.

**To eliminate (optional):**
1. Pre-bake models into Docker image
2. Use Cloud Storage for model caching
3. Simply ignore the warnings - app works fine

### 4. Worker Process Exceptions

**Error:**
```
[ERROR] Exception in worker process
```

**Common Causes:**
1. Missing environment variables (JWT_SECRET_KEY, CORS_ALLOWED_ORIGINS)
2. Database connection issues (DATABASE_URL)
3. Wrong environment variable names

**Debug:**
```bash
# Check environment variables
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --format=json | jq '.spec.template.spec.containers[0].env'

# Check logs
gcloud run services logs tail arrow-rag-backend --region=us-central1
```

### 5. Database Connection Issues

**Error:**
```
could not connect to server
```

**Check:**
1. DATABASE_URL is correctly formatted
2. Cloud SQL instance is running
3. Cloud Run service account has access
4. Cloud SQL connector is properly configured

```bash
# Verify Cloud SQL connection
gcloud sql instances describe rag-postgres --project=arrow-rag-support-prod
```

### 6. Memory/Timeout Issues

**Symptoms:**
- Service crashes under load
- 503 Service Unavailable errors
- Request timeouts

**Solution:**
```bash
# Increase memory and CPU
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --timeout 300
```

## Environment Variables Reference

### Required for Production

| Variable | Example | Description |
|----------|---------|-------------|
| `ENV` | `prod` | Environment mode (must be 'prod' for production) |
| `JWT_SECRET_KEY` | `<32+ char secret>` | JWT signing key |
| `CORS_ALLOWED_ORIGINS` | `https://app.com` | Allowed CORS origins (comma-separated) |
| `DATABASE_URL` | `postgresql://...` | Database connection string |
| `ANTHROPIC_API_KEY` | `sk-ant-...` | Anthropic API key |

### Optional

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level |
| `HF_HOME` | `/tmp/hf` | HuggingFace cache directory |
| `GUNICORN_WORKERS` | `2` | Number of worker processes |
| `GUNICORN_TIMEOUT` | `300` | Worker timeout in seconds |

## Security Best Practices

### 1. Use Google Secret Manager

Instead of environment variables, store secrets in Secret Manager:

```bash
# Create secrets
echo "your-jwt-secret" | gcloud secrets create jwt-secret-key --data-file=-
echo "postgresql://..." | gcloud secrets create database-url --data-file=-

# Grant access to Cloud Run service account
PROJECT_NUMBER=$(gcloud projects describe arrow-rag-support-prod --format="value(projectNumber)")
SERVICE_ACCOUNT="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

gcloud secrets add-iam-policy-binding jwt-secret-key \
  --member="serviceAccount:${SERVICE_ACCOUNT}" \
  --role="roles/secretmanager.secretAccessor"

# Update Cloud Run to use secrets
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --update-secrets "JWT_SECRET_KEY=jwt-secret-key:latest" \
  --update-secrets "DATABASE_URL=database-url:latest"
```

### 2. JWT Secret Requirements

- Minimum 32 characters
- Random and unpredictable
- Never use common defaults

**Generate securely:**
```bash
python -c 'import secrets; print(secrets.token_urlsafe(32))'
# or
openssl rand -base64 32
```

### 3. CORS Configuration

- Never use `*` wildcard in production
- List all legitimate frontend origins
- Keep the list minimal

## Verification Commands

### Check Service Health
```bash
# Get service URL
SERVICE_URL=$(gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --format="value(status.url)")

# Test health endpoint
curl $SERVICE_URL/health

# Test with authentication
curl -X POST $SERVICE_URL/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"test123"}'
```

### Monitor Logs
```bash
# Tail logs
gcloud run services logs tail arrow-rag-backend --region=us-central1

# Filter for errors
gcloud run services logs read arrow-rag-backend \
  --region=us-central1 \
  --filter="severity>=ERROR" \
  --limit=50
```

### Check Configuration
```bash
# View all environment variables
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --format="get(spec.template.spec.containers[0].env)"

# View resource limits
gcloud run services describe arrow-rag-backend \
  --region=us-central1 \
  --format="get(spec.template.spec.containers[0].resources)"
```

## Quick Fix Commands

### Update Environment Variables
```bash
# Set correct variables
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --update-env-vars "ENV=prod,JWT_SECRET_KEY=<secret>,CORS_ALLOWED_ORIGINS=<origins>"

# Remove incorrect variables
gcloud run services update arrow-rag-backend \
  --region us-central1 \
  --remove-env-vars "APP_ENV,ALLOWED_ORIGINS,TRANSFORMERS_CACHE"
```

### Rollback Deployment
```bash
# List revisions
gcloud run revisions list --service=arrow-rag-backend --region=us-central1

# Rollback to previous revision
gcloud run services update-traffic arrow-rag-backend \
  --region=us-central1 \
  --to-revisions=<revision-name>=100
```

### Force Redeploy
```bash
# Redeploy current image
gcloud run services update arrow-rag-backend \
  --region=us-central1 \
  --platform=managed
```

## Getting Help

1. **Check logs first:**
   ```bash
   gcloud run services logs tail arrow-rag-backend --region=us-central1
   ```

2. **Review environment configuration:**
   ```bash
   gcloud run services describe arrow-rag-backend --region=us-central1
   ```

3. **Test locally with docker-compose:**
   ```bash
   docker-compose -f docker-compose.prod.yml up
   ```

4. **Consult main deployment guide:**
   See `docs/DEPLOYMENT_GUIDE.md` for full deployment instructions

## Additional Resources

- [Google Cloud Run Documentation](https://cloud.google.com/run/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Gunicorn Documentation](https://docs.gunicorn.org/)

