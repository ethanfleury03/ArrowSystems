# GitHub Actions Workflows

## 🔍 Validate Secrets and Configuration (STANDALONE)

**File:** `validate-secrets.yml`

**Purpose:** Check that all required secrets and environment variables are properly configured BEFORE deploying.

**How to Run:**

1. Go to: https://github.com/YOUR_USERNAME/ArrowSystems/actions
2. Click on "🔍 Validate Secrets and Configuration" in the left sidebar
3. Click "Run workflow" button (top right)
4. Click the green "Run workflow" button in the dropdown
5. Wait ~30 seconds for results

**What it checks:**

- ✅ JWT_SECRET_KEY is set and meets length requirements (32+ chars)
- ✅ DATABASE_URL is set
- ✅ Frontend/Backend workflows reference secrets correctly
- ✅ No random secret generation in code
- ✅ CORS and URL configurations are present
- ✅ Cookie settings configured for cross-origin

**When to run:**

- ✅ After adding/updating secrets
- ✅ Before deploying to production
- ✅ When debugging authentication issues
- ✅ Weekly automated health check (runs automatically on Sundays)

**This workflow does NOT:**

- ❌ Deploy anything
- ❌ Modify any code
- ❌ Change any secrets
- ❌ Make any API calls

It's purely a validation tool - safe to run anytime!

---

## 🚀 Deploy Backend to Cloud Run

**File:** `deploy-backend.yml`

**Triggers:** Automatically when changes are pushed to `backend/` directory or workflow file

**What it does:**
- Builds backend Docker image
- Pushes to Artifact Registry
- Deploys to Cloud Run
- Sets all environment variables and secrets

---

## 🌐 Deploy Frontend to Cloud Run

**File:** `deploy-frontend.yml`

**Triggers:** Automatically when changes are pushed to `frontend/` directory or workflow file

**What it does:**
- Builds frontend Docker image (Next.js standalone)
- Pushes to Artifact Registry
- Deploys to Cloud Run
- Sets environment variables

---

## Troubleshooting

### Validation fails with "JWT_SECRET_KEY is not set"

**Fix:**
1. Generate a secret: `python -c 'import secrets; print(secrets.token_urlsafe(64))'`
2. Go to: Settings → Secrets → Actions → New secret
3. Name: `JWT_SECRET_KEY`
4. Value: (paste generated secret)
5. Re-run validation workflow

### Deployment fails after validation passes

**Possible causes:**
- Secrets were changed after validation
- Cloud Run service configuration issue
- Docker build failure

**Fix:**
1. Check deployment logs in Actions tab
2. Re-run validation to confirm secrets still valid
3. Check Cloud Run console for service errors

### Need to add a new secret?

1. Add the secret check to `validate-secrets.yml`
2. Add the secret to the appropriate deploy workflow
3. Document it in `SECRETS_REFERENCE.md`
4. Run validation to confirm

---

## Quick Reference

| Workflow | Type | Runs On | Purpose |
|----------|------|---------|---------|
| validate-secrets.yml | **Standalone** | Manual / Weekly | Validate configuration |
| deploy-backend.yml | Auto | Push to backend/ | Deploy backend |
| deploy-frontend.yml | Auto | Push to frontend/ | Deploy frontend |

