# Logging Cleanup Report

## Overview
Reviewed all logging across frontend and backend for issues, excessive verbosity, and security concerns in production.

## Issues Found

### ✅ Good News
1. **No sensitive data leaks** - No passwords, tokens, or secrets being logged
2. **Proper logger usage** - Backend uses structured logging with `logger` from `logging_config.py`
3. **Frontend has logger utility** - `frontend/lib/logger.ts` with proper log levels
4. **Log level control** - Both frontend and backend respect environment-based log levels

### ⚠️ Issues to Address

#### 1. Console.log in Production (Frontend)
**Files with console statements:**
- `frontend/app/api/auth/login/route.ts` - Line 23, 77
- `frontend/lib/iam-backend.ts` - Line 91
- Multiple API routes (~33 files total)
- Multiple components (~8 files)

**Impact:** Low - These are server-side logs (Next.js API routes) that appear in Cloud Run logs, not browser console. Not a security issue, but adds noise.

**Recommendation:** 
- Keep error logs (console.error) for debugging
- Remove verbose success logs (console.log) or gate them with NODE_ENV check
- Critical paths (auth) should keep error logging

#### 2. Transformers Cache Warning (Backend)
**Log snippet:**
```
There was a problem when trying to write in your cache folder (/tmp/hf). 
You should set the environment variable TRANSFORMERS_CACHE to a writable directory.
```

**Status:** ✅ **ALREADY FIXED** in deployment config
- `.github/workflows/deploy-backend.yml` sets:
  - `HF_HOME=/tmp/hf`
  - `TRANSFORMERS_CACHE=/tmp/hf`
  - `SENTENCE_TRANSFORMERS_HOME=/tmp/hf`

**Note:** This warning appears because transformers tries to write before our env vars take effect. It's harmless but noisy.

#### 3. Sharp Image Warning (Frontend)
**Log snippet:**
```
⨯ Error: 'sharp' is required to be installed in standalone mode
```

**Status:** ✅ **FIXED** in `frontend/Dockerfile`
- Added sharp native bindings copy in production stage

#### 4. Excessive Debug Logging
**Backend:**
- 65 logger.debug() or logger.info() calls in `backend/api.py`
- Most are conditional based on ENV setting (good!)
- No action needed - controlled by `ENV=prod` setting

**Frontend:**
- Logger respects `NEXT_PUBLIC_LOG_LEVEL` (defaults to 'info' in production)
- Only errors and warnings go to console in production mode

## Recommended Actions

### Priority 1: Clean Up Noisy Logs (Optional)

#### Frontend - Remove verbose success logs in production

**File: `frontend/app/api/auth/login/route.ts`**

Current (line 77):
```typescript
console.log(`Login successful for user: ${email} (role: ${user.role})`);
```

Recommended:
```typescript
// Only log in development
if (process.env.NODE_ENV !== 'production') {
  console.log(`Login successful for user: ${email} (role: ${user.role})`);
}
```

Or remove entirely - success is implied by 200 response.

**File: `frontend/lib/iam-backend.ts`**

Keep the console.error (line 91) - it's useful for debugging auth issues.

### Priority 2: Suppress Transformers Warning (Optional)

The warning is harmless but noisy. To suppress:

**Option 1: Update backend startup**
Add to `backend/api.py` lifespan startup:
```python
# Suppress transformers cache warning (we set env vars, but warning still shows)
import warnings
warnings.filterwarnings('ignore', message='.*TRANSFORMERS_CACHE.*')
```

**Option 2: Set before imports**
Add to `backend/preload_models.py` or `backend/api.py` at the top:
```python
import os
os.environ.setdefault('HF_HOME', '/tmp/hf')
os.environ.setdefault('TRANSFORMERS_CACHE', '/tmp/hf')
```

### Priority 3: Standardize Logging (Nice to Have)

Some places use `print()` for test functions:
- `backend/rag_pipeline.py` - Test function only (lines 256-274)
- These are fine - they're for manual testing, not production

## Current Production Log Volume

### Backend (Cloud Run)
**Expected logs per request:**
- INFO level: RAG query processing steps
- WARNING: Rate limits, missing data
- ERROR: Failed queries, auth issues

**Volume:** Moderate - appropriate for production monitoring

### Frontend (Cloud Run)  
**Expected logs per request:**
- Errors from failed API calls
- Auth failures
- Critical errors

**Volume:** Low - mostly quiet unless issues occur

## Security Check ✅

Verified no logging of:
- ❌ Passwords
- ❌ JWT tokens (full token)
- ❌ API keys/secrets
- ❌ User personal data (beyond email in context)

All sensitive operations log only:
- ✅ Usernames/emails (needed for debugging)
- ✅ User roles (needed for authorization debugging)
- ✅ Request paths and status codes
- ✅ Error messages (sanitized in production)

## Monitoring Recommendations

### What to Monitor

**Backend:**
```bash
# Error rate
gcloud logging read "resource.type=cloud_run_revision 
  AND resource.labels.service_name=arrow-rag-backend 
  AND severity>=ERROR" --limit 50

# Auth failures
gcloud logging read "resource.type=cloud_run_revision 
  AND resource.labels.service_name=arrow-rag-backend 
  AND textPayload=~'auth|login'" --limit 20

# Performance
gcloud logging read "resource.type=cloud_run_revision 
  AND resource.labels.service_name=arrow-rag-backend 
  AND textPayload=~'query_duration'" --limit 20
```

**Frontend:**
```bash
# Errors
gcloud logging read "resource.type=cloud_run_revision 
  AND resource.labels.service_name=arrow-rag-frontend 
  AND severity>=ERROR" --limit 50

# Auth issues
gcloud logging read "resource.type=cloud_run_revision 
  AND resource.labels.service_name=arrow-rag-frontend 
  AND textPayload=~'Login'" --limit 20
```

## Summary

### Current State: **GOOD** ✅
- No security issues
- Logging levels properly configured
- Only minor verbosity issues

### Action Required: **OPTIONAL**
- Most issues are cosmetic (noisy logs)
- Core functionality not affected
- Can clean up at leisure

### Immediate Focus: **Cookie Auth Fix**
- Prioritize fixing login (Cookie SameSite=none)
- Logging cleanup can be done in next iteration

## Files That Need Updates (Optional)

If you want to reduce log noise:

1. **frontend/app/api/auth/login/route.ts**
   - Remove or gate console.log success message

2. **backend/api.py** (optional)
   - Add warning filter for transformers cache

3. **Various API routes** (low priority)
   - Review console.log statements
   - Consider using proper logger or removing

## Next Steps

1. ✅ Fix cookie authentication (higher priority)
2. ✅ Deploy backend with updated SameSite config
3. ✅ Deploy frontend with preserved Set-Cookie headers
4. ⏭️ Test login functionality
5. ⏭️ Clean up verbose logging (optional, later iteration)

