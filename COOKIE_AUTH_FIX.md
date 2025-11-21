# Cookie Authentication Fix for Cross-Origin Cloud Run Services

## Problem
Login was failing because the frontend and backend are deployed on different Cloud Run domains:
- Frontend: `arrow-rag-frontend-70705019874.us-central1.run.app`
- Backend: `arrow-rag-backend-70705019874.us-central1.run.app`

The JWT authentication cookies from the backend were not working due to cross-origin cookie restrictions.

## Root Causes

### 1. **SameSite Cookie Policy**
- Backend was using `SameSite=lax` which doesn't allow cookies in cross-site requests
- For cookies to work across different domains, we need `SameSite=none`
- `SameSite=none` requires `Secure=true` (HTTPS only)

### 2. **Response Headers Not Preserved**
- The frontend's IAM backend proxy (`frontend/lib/iam-backend.ts`) was not preserving the `Set-Cookie` header from backend responses
- This caused authentication cookies to be lost when proxying requests

### 3. **Minor Issues**
- `sharp` library warning in production (needed for image optimization)
- Port configuration inconsistency (3000 vs 8080)

## Changes Made

### Backend Changes

#### 1. `backend/config/auth.py`
- Updated `AUTH_COOKIE_SAMESITE` to use `"none"` in production (instead of `"lax"`)
- Added comment explaining why `SameSite=none` is required for cross-origin cookies
- Keeps `"lax"` for development (local testing)

```python
# SameSite=None is required for cross-origin cookies (frontend/backend on different domains)
# This requires Secure=true (HTTPS only)
self.AUTH_COOKIE_SAMESITE = os.getenv("AUTH_COOKIE_SAMESITE", "none" if settings.is_prod else "lax")
```

#### 2. `.github/workflows/deploy-backend.yml`
- Added explicit environment variables for cookie configuration:
  ```bash
  --set-env-vars AUTH_COOKIE_SAMESITE=none \
  --set-env-vars AUTH_COOKIE_SECURE=true \
  ```

### Frontend Changes

#### 1. `frontend/lib/iam-backend.ts`
- **Fixed**: Now preserves `Set-Cookie` header from backend responses
- Updated `iamBackendRequest` to copy important headers including `set-cookie`
- Applied to both success and error responses

```typescript
// Preserve all headers from the backend response, especially Set-Cookie for auth
const responseHeaders: Record<string, string> = {
  'Content-Type': 'application/json',
};

// Copy important headers from backend response
if (response.headers) {
  // Preserve Set-Cookie header for authentication
  if (response.headers['set-cookie']) {
    responseHeaders['set-cookie'] = response.headers['set-cookie'];
  }
  // ... other headers
}
```

#### 2. `frontend/Dockerfile`
- Added sharp native bindings to production image:
  ```dockerfile
  # Copy sharp for image optimization in standalone mode
  COPY --from=builder --chown=nextjs:nodejs /app/node_modules/sharp ./node_modules/sharp
  ```

#### 3. `.github/workflows/deploy-frontend.yml`
- Added explicit port configuration:
  ```bash
  --port 3000 \
  --set-env-vars PORT=3000 \
  ```

## How It Works Now

### Authentication Flow:
1. **User logs in** → Frontend API route (`/api/auth/login`) receives credentials
2. **Frontend proxies** → Calls backend `/auth/login` using IAM authentication
3. **Backend responds** → Sets JWT in HTTP-only cookie with `SameSite=none; Secure`
4. **Frontend preserves** → The `Set-Cookie` header is forwarded to the browser
5. **Browser stores** → Cookie is saved and sent with subsequent requests

### Cookie Configuration in Production:
```
Set-Cookie: access_token=<jwt_token>; 
  HttpOnly; 
  Secure; 
  SameSite=None; 
  Path=/; 
  Max-Age=3600
```

- **HttpOnly**: Prevents JavaScript access (XSS protection)
- **Secure**: HTTPS only (required for SameSite=None)
- **SameSite=None**: Allows cross-origin requests (frontend ↔ backend)
- **Path=/**: Available for all routes
- **Max-Age=3600**: 1 hour expiration (configurable via `JWT_ACCESS_TOKEN_EXPIRE_MINUTES`)

## Testing the Fix

### 1. Deploy Backend First
```bash
git add backend/config/auth.py .github/workflows/deploy-backend.yml
git commit -m "fix: Update cookie config for cross-origin authentication"
git push origin main
```

### 2. Deploy Frontend
```bash
git add frontend/lib/iam-backend.ts frontend/Dockerfile .github/workflows/deploy-frontend.yml
git commit -m "fix: Preserve Set-Cookie headers and add sharp for image optimization"
git push origin main
```

### 3. Test Login
1. Navigate to `https://arrow-rag-frontend-70705019874.us-central1.run.app/login`
2. Enter credentials
3. Check browser DevTools → Application → Cookies
4. Verify `access_token` cookie is set with:
   - ✅ HttpOnly
   - ✅ Secure
   - ✅ SameSite: None

### 4. Test Authenticated Requests
1. After login, navigate to protected pages (e.g., `/admin`)
2. Verify requests include the cookie
3. Check that API calls succeed with authentication

## Security Considerations

### Why SameSite=None is Safe Here:
1. **HTTPS Only**: `Secure` flag ensures cookies only sent over HTTPS
2. **HttpOnly**: JavaScript cannot access the token
3. **CORS Configured**: Backend only accepts requests from the frontend origin
4. **JWT Signed**: Tokens are cryptographically signed and verified
5. **Short-lived**: Tokens expire after 1 hour (configurable)

### Alternative Approaches (Not Used):
1. **Same Domain** - Use a single domain with subdomain routing (e.g., `app.arrow.com` and `api.arrow.com`)
2. **Custom Domain** - Configure custom domain for both services
3. **Bearer Token** - Store JWT in memory and send via Authorization header (requires client-side storage)

## Monitoring

### Backend Logs to Watch:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-backend" \
  --limit 50 --format json | grep -i "auth\|cookie\|login"
```

### Frontend Logs to Watch:
```bash
gcloud logging read "resource.type=cloud_run_revision AND resource.labels.service_name=arrow-rag-frontend" \
  --limit 50 --format json | grep -i "auth\|cookie\|login"
```

### Common Issues:
- **Cookie not set**: Check Set-Cookie header in response
- **Cookie not sent**: Check SameSite and Secure attributes
- **CORS errors**: Verify backend CORS_ALLOWED_ORIGINS includes frontend URL
- **401 Unauthorized**: Check JWT_SECRET_KEY matches between services

## Environment Variables Reference

### Backend (Required):
```bash
JWT_SECRET_KEY=<your-secret-key>           # Must match across deployments
AUTH_COOKIE_SAMESITE=none                  # For cross-origin (production)
AUTH_COOKIE_SECURE=true                    # HTTPS only (production)
CORS_ALLOWED_ORIGINS=https://frontend-url  # Frontend origin
ENV=prod                                   # Enable production mode
```

### Frontend (Required):
```bash
NEXT_PUBLIC_API_URL=https://backend-url    # Backend URL
NODE_ENV=production                         # Production mode
PORT=3000                                  # Server port
```

## References
- [MDN: SameSite cookies](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite)
- [Chrome: SameSite Updates](https://www.chromium.org/updates/same-site/)
- [OWASP: Session Management](https://cheatsheetseries.owasp.org/cheatsheets/Session_Management_Cheat_Sheet.html)

