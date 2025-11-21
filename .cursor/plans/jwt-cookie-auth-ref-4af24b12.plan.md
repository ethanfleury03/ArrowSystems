<!-- 4af24b12-31ab-49f4-9fe2-38fc46ccbd8e e1e5ac31-a502-4b7e-8327-498186fd8951 -->
# JWT Cookie-Based Authentication Refactor

## Overview

Migrate from localStorage JWT + iron-session hybrid to a clean backend-owned JWT cookie architecture where the backend sets HTTP-only cookies containing JWTs, and the frontend validates and forwards these tokens.

## Architecture Changes

### Backend Changes (FastAPI)

#### 1. Create `backend/config/auth.py` - Centralized Auth Configuration

- Extract all auth-related config from `backend/config/env.py` and `backend/security.py`
- Add new environment variables:
  - `JWT_ACCESS_TOKEN_EXPIRE_MINUTES` (default: 60)
  - `JWT_REFRESH_TOKEN_EXPIRE_DAYS` (default: 7)
  - `AUTH_COOKIE_NAME` (default: "access_token")
  - `AUTH_COOKIE_DOMAIN` (optional, for Cloud Run cross-service cookies)
  - `AUTH_COOKIE_SECURE` (default: true in prod, false in dev)
  - `AUTH_COOKIE_SAMESITE` (default: "lax")
- Centralize cookie setting logic with proper attributes based on env

#### 2. Refactor `backend/api.py` - `/auth/login` Endpoint

- After successful authentication, create JWT as currently done
- **Instead of returning JWT in response body**, set it as HTTP-only cookie via `response.set_cookie()`
- Return only safe user data: `{"user": {...}, "message": "Login successful"}`
- Cookie options: `httpOnly=True`, `secure` based on env, `samesite` from config, `max_age` from token expiry, `path=/`, optional `domain`

#### 3. Add `backend/api.py` - `/auth/logout` Endpoint

- Create `POST /auth/logout` endpoint
- Clear auth cookie: `response.delete_cookie()` or set with `max_age=0`
- Return `{"message": "Logged out successfully"}`

#### 4. Refactor `backend/api.py` - `/auth/me` Endpoint

- Currently has no authentication dependency (signature shows no params)
- Add dependency to extract JWT from cookie or Authorization header
- Priority: check `Authorization: Bearer <token>` first (for API routes), then check cookie (for direct access)
- Decode JWT, validate, fetch user from DB, return user data

#### 5. Update `backend/security.py`

- Keep `create_access_token()` and `decode_access_token()` 
- Add `get_jwt_from_request(request: Request) -> Optional[str]` that checks both cookie and Authorization header
- Add dependency function `get_current_user(request: Request, db: DatabaseManager) -> dict` for protected endpoints

#### 6. Update CORS in `backend/api.py`

- Ensure `allow_credentials=True` is set
- Verify `allow_origins` uses `settings.CORS_ALLOWED_ORIGINS` (already done)
- Add support for frontend domain to receive cookies

### Frontend Changes (Next.js)

#### 7. Remove iron-session Dependencies

- Remove `iron-session` from `package.json`
- Delete `frontend/lib/auth.ts` entirely (session management code no longer needed)
- Remove `SESSION_SECRET` from frontend env vars

#### 8. Create `frontend/lib/authClient.ts` - JWT Cookie Client Utilities

- `extractJwtFromCookie(cookieName: string): string | null` - for server-side use in API routes and middleware
- `validateJwt(token: string): { email: string; role: string; exp: number } | null` - decode JWT client-side (requires adding `jsonwebtoken` to frontend)
- Note: Frontend only validates JWT structure/signature for routing decisions, backend is source of truth

#### 9. Refactor `frontend/app/api/auth/login/route.ts`

- Remove `setLoginSession()` call (no more iron-session)
- Call backend `/auth/login` via `iamBackendPost()`
- Backend response will include `Set-Cookie` header with JWT
- **Forward the Set-Cookie header from backend to browser**: Extract `Set-Cookie` from backend response, add to frontend response headers
- Return safe response to client: `{"user": {...}, "message": "Login successful"}` (no token in body)
- Remove all `localStorage` operations

#### 10. Create `frontend/app/api/auth/logout/route.ts`

- POST endpoint that calls backend `/auth/logout` via `iamBackendPost()`
- Forward cookie-clearing response from backend to browser
- Return success message

#### 11. Refactor `frontend/app/api/auth/me/route.ts`

- Read JWT from cookie (server-side): `cookies().get('access_token')`
- Call backend `/auth/me` with `Authorization: Bearer <jwt>` header via `iamBackendGet()`
- Return user data to client

#### 12. Update `frontend/middleware.ts`

- Remove iron-session usage (`app_session_v2` cookie check)
- Check for JWT cookie (`access_token` or configured name)
- Optionally: decode JWT to check expiration client-side (avoid redirect if expired anyway)
- Redirect to `/login` if no cookie or expired
- Allow logged-in users with valid cookies to access protected routes

#### 13. Refactor `frontend/app/login/page.tsx`

- Remove all `localStorage.setItem('auth_token', ...)` and `localStorage.setItem('user_profile', ...)` code
- After successful login (200 response), simply redirect - cookie is already set by API route
- Remove token handling entirely from client component

#### 14. Update Admin API Routes to Forward JWT

Fix all routes in `frontend/app/api/admin/**/route.ts`:

- Extract JWT from cookies: `cookies().get('access_token')?.value`
- Add to backend requests: `Authorization: Bearer <jwt>`
- Example for `frontend/app/api/admin/users/route.ts`: update all `iamBackend*` calls to include `{ 'Authorization': `Bearer ${jwt}` }` in headers

#### 15. Update Logout Buttons

- `frontend/app/account/logout-button.tsx`: Remove `localStorage.removeItem()` calls, call `/api/auth/logout`
- `frontend/components/sidebar.tsx`: Remove `localStorage.removeItem()` calls, call `/api/auth/logout`

### Environment Configuration

#### 16. Update Backend Environment Variables

Create/update `.env.example`:

```
# Auth Configuration
JWT_SECRET_KEY=<leave-empty-to-use-baked-in-default>
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=60
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7
AUTH_COOKIE_NAME=access_token
AUTH_COOKIE_DOMAIN=  # Optional: set for Cloud Run cross-service cookies
AUTH_COOKIE_SECURE=true  # auto-detected in prod
AUTH_COOKIE_SAMESITE=lax
```

#### 17. Update Frontend Environment Variables

Remove from frontend `.env.local` and `.env.example`:

```
SESSION_SECRET  # No longer needed
```

Add to frontend:

```
# JWT validation (must match backend)
NEXT_PUBLIC_JWT_SECRET_KEY=<same-as-backend-secret>
AUTH_COOKIE_NAME=access_token
```

### Testing & Verification

#### 18. End-to-End Flow Verification

- Start backend and frontend locally
- Test login: verify JWT cookie is set in browser (check DevTools → Application → Cookies)
- Test protected route: verify middleware allows access with valid cookie
- Test admin route: verify token is forwarded and admin content loads
- Test logout: verify cookie is cleared and redirect to login
- Test token expiration: manually edit cookie expiration, verify redirect to login

### Documentation

#### 19. Create `docs/auth-architecture.md`

Document:

- Auth flow diagram (login → JWT cookie → validation → protected routes)
- Cookie configuration and options
- Environment variables and their purposes
- How to rotate JWT secrets
- Troubleshooting common issues (CORS, cookie domain, secure flag)

#### 20. Update README or deployment docs

- List all new environment variables
- Note that frontend must share `JWT_SECRET_KEY` with backend for validation
- Explain cookie domain configuration for Cloud Run deployment

## Breaking Changes & Migration

- **localStorage auth_token**: Will be ignored. Users will be logged out on first visit after deployment.
- **iron-session cookies** (`app_session_v2`): Will be ignored. All existing sessions invalidated.
- **Environment variables**: Backend deployment must set `AUTH_COOKIE_NAME`, frontend must add `NEXT_PUBLIC_JWT_SECRET_KEY`.

## Key Implementation Details

### Cookie Domain for Cloud Run

Since frontend and backend are separate Cloud Run services with different domains, cookies need special handling:

- Option 1: Don't set domain (restricts cookie to frontend domain only) - simpler, frontend always proxies to backend
- Option 2: Set explicit domain if services share parent domain - more complex, allows direct backend access

For your architecture (frontend always proxies via IAM), **Option 1** is recommended.

### JWT in Response Headers

When backend sets cookies, the `Set-Cookie` header must be forwarded through the frontend API route response:

```typescript
// In frontend/app/api/auth/login/route.ts
const backendResponse = await iamBackendPost('/auth/login', body);
const setCookieHeader = backendResponse.headers.get('set-cookie');
const frontendResponse = NextResponse.json(data);
if (setCookieHeader) {
  frontendResponse.headers.set('set-cookie', setCookieHeader);
}
return frontendResponse;
```

### JWT Validation on Frontend

Frontend needs to decode JWT for middleware routing decisions. This requires:

- Installing `jsonwebtoken` package
- Sharing JWT_SECRET_KEY between backend and frontend (via env var)
- Validation is for UX only; backend always re-validates

### To-dos

- [ ] Create backend/config/auth.py with centralized auth configuration
- [ ] Refactor backend /auth/login to set JWT in HTTP-only cookie
- [ ] Add backend POST /auth/logout endpoint to clear cookies
- [ ] Update backend /auth/me to validate JWT from cookie or header
- [ ] Add JWT extraction and validation helpers in backend/security.py
- [ ] Remove iron-session dependencies and frontend/lib/auth.ts
- [ ] Create frontend/lib/authClient.ts with JWT cookie utilities
- [ ] Refactor frontend /api/auth/login to forward Set-Cookie headers
- [ ] Create frontend /api/auth/logout endpoint
- [ ] Update frontend /api/auth/me to forward JWT in Authorization header
- [ ] Update frontend middleware.ts to check JWT cookie instead of session
- [ ] Remove localStorage operations from login page
- [ ] Update all frontend admin API routes to forward JWT tokens
- [ ] Update logout buttons to call /api/auth/logout
- [ ] Add new auth-related environment variables to backend
- [ ] Update frontend environment variables (remove SESSION_SECRET)
- [ ] Test complete login/logout flow with JWT cookies
- [ ] Create docs/auth-architecture.md documenting the new flow