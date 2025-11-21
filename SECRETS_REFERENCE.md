# Secrets and Environment Variables Reference

## Required GitHub Secrets

Go to: `https://github.com/YOUR_USERNAME/ArrowSystems/settings/secrets/actions`

### Backend Secrets

| Secret Name | Required | Description | How to Generate |
|------------|----------|-------------|-----------------|
| `JWT_SECRET_KEY` | ✅ YES | JWT signing secret for user sessions | `python -c 'import secrets; print(secrets.token_urlsafe(64))'` |
| `DATABASE_URL` | ✅ YES | PostgreSQL connection string | From Cloud SQL instance |

### Frontend Secrets

| Secret Name | Required | Description | How to Generate |
|------------|----------|-------------|-----------------|
| `FRONTEND_SESSION_SECRET` | ⚠️ Optional | Next.js session secret (currently not used) | `python -c 'import secrets; print(secrets.token_urlsafe(32))'` |

## Environment Variables Set in Workflows

### Backend (.github/workflows/deploy-backend.yml)

| Variable | Value | Description |
|----------|-------|-------------|
| `JWT_SECRET_KEY` | `${{ secrets.JWT_SECRET_KEY }}` | From GitHub secret |
| `DATABASE_URL` | `${{ secrets.DATABASE_URL }}` | From GitHub secret |
| `HF_HOME` | `/tmp/hf` | HuggingFace cache directory |
| `TRANSFORMERS_CACHE` | `/tmp/hf` | Transformers cache directory |
| `SENTENCE_TRANSFORMERS_HOME` | `/tmp/hf` | Sentence transformers cache |
| `ENABLE_INGESTION_ON_STARTUP` | `false` | Disable ingestion on Cloud Run startup |
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python output |
| `ENV` | `prod` | Environment mode (enables production settings) |
| `CORS_ALLOWED_ORIGINS` | `https://arrow-rag-frontend-...` | Frontend URL for CORS |
| `AUTH_COOKIE_SAMESITE` | `none` | Cookie SameSite for cross-origin |
| `AUTH_COOKIE_SECURE` | `true` | HTTPS-only cookies |

### Frontend (.github/workflows/deploy-frontend.yml)

| Variable | Value | Description |
|----------|-------|-------------|
| `NEXT_PUBLIC_API_URL` | `https://arrow-rag-backend-...` | Backend API URL |
| `NODE_ENV` | `production` | Node environment |
| `PORT` | `3000` | Server port |
| `SESSION_SECRET` | `${{ secrets.FRONTEND_SESSION_SECRET }}` | From GitHub secret (optional) |

## Setup Instructions

### First-Time Setup

1. **Generate JWT Secret**
   ```bash
   python -c 'import secrets; print(secrets.token_urlsafe(64))'
   ```
   Copy the output.

2. **Add to GitHub**
   - Go to: https://github.com/YOUR_USERNAME/ArrowSystems/settings/secrets/actions
   - Click "New repository secret"
   - Name: `JWT_SECRET_KEY`
   - Value: (paste the generated secret)
   - Click "Add secret"

3. **Get Database URL**
   - Go to Cloud SQL in GCP Console
   - Find your PostgreSQL instance: `rag-postgres`
   - Connection string format:
     ```
     postgresql://USER:PASSWORD@/DATABASE?host=/cloudsql/PROJECT_ID:REGION:INSTANCE_NAME
     ```

4. **Add Database URL to GitHub**
   - Same steps as JWT secret
   - Name: `DATABASE_URL`
   - Value: (your PostgreSQL connection string)

### Validation

Run the validation workflow to check all secrets are set:

```bash
# Manually trigger via GitHub UI
# Or push to main to run automatically
```

The workflow will check:
- ✅ All required secrets are set
- ✅ Secrets meet minimum length requirements
- ✅ Workflow files reference the secrets correctly
- ✅ No random secret generation in code
- ✅ URLs are configured consistently

## Troubleshooting

### "Invalid token" errors after login

**Cause:** JWT_SECRET_KEY is not set or changed between deployments

**Fix:**
1. Verify secret is set in GitHub
2. Check backend logs for "JWT_SECRET_KEY is REQUIRED" error
3. Redeploy backend after adding secret

### "Not authenticated" showing in UI

**Cause:** Frontend can't reach backend or JWT validation failing

**Fix:**
1. Check NEXT_PUBLIC_API_URL in frontend deployment
2. Check CORS_ALLOWED_ORIGINS in backend deployment
3. Verify both services are deployed and healthy

### Database connection errors

**Cause:** DATABASE_URL not set or Cloud SQL connection not configured

**Fix:**
1. Verify DATABASE_URL secret is set
2. Check `--set-cloudsql-instances` in backend workflow
3. Ensure Cloud SQL instance is running

## Security Best Practices

- ✅ Never commit secrets to code
- ✅ Use GitHub Secrets for all sensitive values
- ✅ Rotate JWT_SECRET_KEY if compromised (will invalidate all sessions)
- ✅ Use different secrets for dev and prod
- ✅ Keep secrets at least 32 characters long
- ✅ Never use common values like "secret" or "password"

## Updating Secrets

**To rotate JWT_SECRET_KEY:**
1. Generate new secret (same command as above)
2. Update in GitHub Secrets
3. Redeploy backend
4. **All users will need to log in again**

**To update DATABASE_URL:**
1. Update in GitHub Secrets
2. Redeploy backend
3. No user impact if database content unchanged

