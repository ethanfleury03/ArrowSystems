# Authentication Architecture

## Overview

This application uses a **JWT cookie-based authentication system** where:
- Backend (FastAPI) creates and validates JWT tokens
- Backend sets tokens in HTTP-only cookies
- Frontend (Next.js) forwards JWT tokens to backend for authorization
- All auth state is server-side; no tokens in localStorage

## Authentication Flow

### Login Flow

```
1. User submits credentials on login page (frontend/app/login/page.tsx)
   ↓
2. Frontend calls POST /api/auth/login (Next.js API route)
   ↓
3. Next.js API route calls backend POST /auth/login via IAM authentication
   ↓
4. Backend validates credentials, creates JWT token
   ↓
5. Backend sets JWT in HTTP-only cookie via Set-Cookie header
   ↓
6. Next.js forwards Set-Cookie header to browser
   ↓
7. Browser stores cookie automatically
   ↓
8. Frontend redirects user to home page (/ or /admin based on role)
```

### Protected Route Access

```
1. User navigates to protected route
   ↓
2. Middleware checks for JWT cookie (frontend/middleware.ts)
   ↓
3. If no cookie → redirect to /login
   ↓
4. If cookie exists → allow access
   ↓
5. Page/component calls API endpoint (e.g., /api/admin/users)
   ↓
6. API route extracts JWT from cookie
   ↓
7. API route forwards JWT to backend in Authorization: Bearer header
   ↓
8. Backend validates JWT and returns data
```

### Logout Flow

```
1. User clicks logout button
   ↓
2. Frontend calls POST /api/auth/logout
   ↓
3. Next.js API route calls backend POST /auth/logout
   ↓
4. Backend clears JWT cookie (max_age=0)
   ↓
5. Next.js forwards cookie-clearing headers to browser
   ↓
6. Browser removes cookie
   ↓
7. Frontend redirects to /login
```

## Cookie Configuration

### Cookie Attributes

The JWT authentication cookie has the following attributes:

| Attribute | Value | Purpose |
|-----------|-------|---------|
| `name` | `access_token` (configurable) | Cookie name |
| `httpOnly` | `true` | Prevents JavaScript access (XSS protection) |
| `secure` | `true` in prod, `false` in dev | Requires HTTPS in production |
| `sameSite` | `lax` | CSRF protection |
| `maxAge` | 3600 seconds (1 hour) | Cookie expiration time |
| `path` | `/` | Cookie available to entire site |
| `domain` | Optional | Set for cross-subdomain sharing |

### Environment Variables

#### Backend (FastAPI)

Required:
```bash
ENV=prod
DATABASE_URL=postgresql://...
CORS_ALLOWED_ORIGINS=https://your-frontend.com
ANTHROPIC_API_KEY=your-api-key
DOCS_BUCKET_NAME=your-bucket
```

Optional (auth-related):
```bash
# JWT Configuration
JWT_SECRET_KEY=  # Leave empty to use baked-in default
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Cookie Configuration
AUTH_COOKIE_NAME=access_token
AUTH_COOKIE_DOMAIN=  # Optional: for cross-service cookies
AUTH_COOKIE_SECURE=true  # Auto-detected if not set
AUTH_COOKIE_SAMESITE=lax
```

#### Frontend (Next.js)

Required:
```bash
NEXT_PUBLIC_API_URL=https://your-backend.run.app
```

Optional (auth-related):
```bash
# JWT validation (must match backend secret for validation)
NEXT_PUBLIC_JWT_SECRET_KEY=same-as-backend-secret

# Cookie name (must match backend)
AUTH_COOKIE_NAME=access_token

# Auth bypass (dev only)
DISABLE_AUTH=false
```

## JWT Token Structure

### Payload

```json
{
  "email": "user@example.com",
  "role": "ADMIN",
  "exp": 1732155600
}
```

### Fields

- `email`: User's email address (unique identifier)
- `role`: User role (`ADMIN` or `CUSTOMER`)
- `exp`: Expiration timestamp (Unix time)

## Security Considerations

### HTTP-Only Cookies

✅ **Secure:**
- Cookies are `httpOnly: true` - not accessible via JavaScript
- Protects against XSS attacks
- Browser automatically sends cookie with requests

❌ **Not Used:**
- ~~localStorage~~ - vulnerable to XSS
- ~~sessionStorage~~ - vulnerable to XSS
- ~~Token in response body to client~~ - could be exposed

### Cookie Domain for Cloud Run

Since frontend and backend are separate Cloud Run services:

**Option 1: No domain set (Recommended)**
- Cookie is bound to frontend domain only
- Frontend always proxies requests to backend via API routes
- Simpler, more secure

**Option 2: Explicit domain**
- Set `AUTH_COOKIE_DOMAIN` if services share parent domain
- Allows direct backend access (not recommended with IAM architecture)

**Current Implementation:** Option 1 (no domain set)

### CORS Configuration

Backend CORS must be configured correctly:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ALLOWED_ORIGINS,  # Explicit frontend URLs
    allow_credentials=True,  # REQUIRED for cookies
    allow_methods=["*"],
    allow_headers=["*"],
)
```

⚠️ **Important:** 
- `allow_credentials=True` is required for cookies to work
- `allow_origins` must list explicit frontend URLs (no wildcards in production)

## Code Organization

### Backend Files

- `backend/config/auth.py` - Auth configuration (cookie options, JWT settings)
- `backend/security.py` - JWT creation, validation, extraction
- `backend/api.py` - Auth endpoints (`/auth/login`, `/auth/logout`, `/auth/me`)
- `backend/routes/admin_routes.py` - Admin endpoints with JWT dependency

### Frontend Files

- `frontend/lib/authClient.ts` - JWT validation utilities
- `frontend/lib/adminAuthHelpers.ts` - Admin API route helpers
- `frontend/middleware.ts` - Route protection (checks cookie presence)
- `frontend/app/api/auth/login/route.ts` - Login proxy
- `frontend/app/api/auth/logout/route.ts` - Logout proxy
- `frontend/app/api/auth/me/route.ts` - Current user proxy
- `frontend/app/api/admin/**/route.ts` - Admin API proxies (forward JWT)

## Troubleshooting

### Cookie Not Being Set

**Symptoms:** Login succeeds but cookie not visible in DevTools

**Causes:**
1. CORS not configured with `allow_credentials: true`
2. Frontend and backend on different domains without proper CORS
3. `sameSite` attribute incompatible with setup
4. HTTPS required but not available (when `secure: true`)

**Solutions:**
1. Check backend CORS configuration
2. Verify `CORS_ALLOWED_ORIGINS` includes frontend URL
3. Check DevTools → Network → Login request → Response Headers for `Set-Cookie`
4. Try `AUTH_COOKIE_SECURE=false` in development

### 401 Unauthorized on Admin Routes

**Symptoms:** Admin pages redirect to login or show 401 errors

**Causes:**
1. JWT cookie not being forwarded to backend
2. JWT expired
3. User doesn't have ADMIN role

**Solutions:**
1. Check admin API routes use `getJwtAuthHeaders()` helper
2. Check cookie expiration in DevTools
3. Check user role in JWT payload (decode at jwt.io)

### Token Expired Errors

**Symptoms:** User logged out unexpectedly

**Causes:**
1. `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` too short
2. Server time skew between frontend and backend
3. Cookie `maxAge` doesn't match token expiration

**Solutions:**
1. Increase token lifetime (e.g., `JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60`)
2. Implement refresh token flow (future enhancement)
3. Ensure cookie `maxAge` in `auth_config.get_cookie_max_age()` matches token lifetime

### Frontend Middleware Redirect Loop

**Symptoms:** Constant redirects between `/login` and `/`

**Causes:**
1. Cookie name mismatch between backend and frontend
2. Middleware not detecting cookie correctly
3. Cookie domain issues

**Solutions:**
1. Verify `AUTH_COOKIE_NAME` matches in both backend and frontend
2. Check middleware uses `getAuthCookieName()` helper
3. Clear all cookies and try again

## Migration from Old System

### Breaking Changes

This implementation replaces the previous `iron-session` based system:

**Old System:**
- Frontend used `iron-session` for encrypted session cookies
- Session stored `userId`, fetched full user from backend
- JWT tokens stored in `localStorage`
- Separate session cookie (`app_session_v2`) and JWT token

**New System:**
- Backend-owned JWT cookies
- No `iron-session` dependency
- No `localStorage` usage
- Single source of truth (JWT cookie)

### User Impact

- **All existing sessions invalidated** - users must log in again
- Old localStorage tokens ignored
- Old `app_session_v2` cookies ignored

### Code Changes Required

✅ **Completed:**
- Removed `iron-session` from package.json
- Deleted `frontend/lib/auth.ts` (iron-session code)
- Removed all `localStorage.setItem/getItem('auth_token')`
- Updated middleware to check JWT cookie
- Updated all admin API routes to forward JWT

❌ **Manual Steps:**
- Run `npm install` in frontend directory to install `jsonwebtoken`
- Set environment variables in deployment
- Deploy backend first, then frontend

## Future Enhancements

### Refresh Tokens

Currently, access tokens expire after 60 minutes. To implement refresh tokens:

1. Add `REFRESH_COOKIE_NAME` configuration
2. Create longer-lived refresh token (7 days)
3. Add `POST /auth/refresh` endpoint
4. Frontend calls refresh endpoint when access token expires
5. Set both cookies on login

### Remember Me

To implement "Remember Me" functionality:

1. Add checkbox to login form
2. Pass `remember_me: boolean` to backend
3. Backend sets different `maxAge` based on flag:
   - Normal: 1 hour
   - Remember Me: 30 days
4. Adjust token expiration accordingly

### Role-Based Access Control (RBAC)

Currently supports two roles: `ADMIN` and `CUSTOMER`

To add more granular permissions:

1. Add `permissions` field to JWT payload
2. Create permission check helper in `backend/security.py`
3. Use as FastAPI dependency on specific endpoints
4. Frontend checks permissions for UI visibility

## API Reference

### POST /auth/login

Request:
```json
{
  "email": "user@example.com",
  "password": "password123"
}
```

Response (200):
```json
{
  "user": {
    "id": "123",
    "email": "user@example.com",
    "role": "ADMIN"
  },
  "message": "Login successful"
}
```

Sets cookie:
```
Set-Cookie: access_token=eyJ...; HttpOnly; Secure; SameSite=Lax; Max-Age=3600; Path=/
```

### POST /auth/logout

Request: (empty body)

Response (200):
```json
{
  "message": "Logged out successfully"
}
```

Clears cookie:
```
Set-Cookie: access_token=; HttpOnly; Secure; SameSite=Lax; Max-Age=0; Path=/
```

### GET /auth/me

Requires: `Authorization: Bearer <token>` header or JWT cookie

Response (200):
```json
{
  "id": "123",
  "email": "user@example.com",
  "name": "John Doe",
  "role": "ADMIN"
}
```

Response (401):
```json
{
  "detail": "Not authenticated"
}
```

## Testing

### Local Testing

1. Start backend: `cd backend && python -m uvicorn backend.api:app --reload`
2. Start frontend: `cd frontend && npm run dev`
3. Navigate to `http://localhost:3000/login`
4. Login with test credentials
5. Check DevTools → Application → Cookies for `access_token`
6. Navigate to protected route (e.g., `/admin`)
7. Check DevTools → Network for requests with `Authorization: Bearer` header

### Production Testing

1. Deploy backend to Cloud Run
2. Update `CORS_ALLOWED_ORIGINS` with frontend URL
3. Deploy frontend to Cloud Run
4. Test login flow from production URL
5. Verify cookies work across requests
6. Test logout clears cookie
7. Verify middleware protects routes

### cURL Testing

Login:
```bash
curl -X POST https://your-backend.run.app/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}' \
  -c cookies.txt \
  -v
```

Check auth:
```bash
curl -X GET https://your-backend.run.app/auth/me \
  -b cookies.txt \
  -v
```

Logout:
```bash
curl -X POST https://your-backend.run.app/auth/logout \
  -b cookies.txt \
  -c cookies.txt \
  -v
```

