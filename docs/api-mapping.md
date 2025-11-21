## API Mapping: React → Next.js API → FastAPI Backend

This document captures the canonical paths and auth behavior for the main features.

### 1. Auth

- **Login**
  - **React**: `POST /api/auth/login`
  - **Next.js**: `app/api/auth/login/route.ts` → proxies to backend `/auth/login`
  - **Backend**: `POST /auth/login`
  - **Auth behavior**:
    - Backend sets an HTTP-only JWT cookie (`access_token`) on success.
    - No `X-User-Token` header is used for this endpoint.

- **Me (current user)**
  - **React**: `GET /api/auth/me`
  - **Next.js**: `app/api/auth/me/route.ts`
  - **Backend**: `GET /auth/me`
  - **Auth behavior**:
    - Next.js reads JWT from the auth cookie via `extractJwtFromCookie()`.
    - Sends it to backend as `X-User-Token`.
    - Backend decodes `X-User-Token` and returns user info.

- **Logout**
  - **React**: `POST /api/auth/logout`
  - **Next.js**: `app/api/auth/logout/route.ts`
  - **Backend**: `POST /auth/logout`
  - **Auth behavior**:
    - Backend clears the auth cookie.
    - React uses this to log the user out and redirect.

### 2. Query & RAG

- **Query**
  - **React**:
    - `sendQuery()` in `frontend/lib/api.ts` → `POST /api/query`
  - **Next.js**:
    - `app/api/query/route.ts`
      - Validates `body.query` as a non-empty string.
      - Optionally calls `/summarize-query`.
      - Maps body to backend `QueryRequest`:
        - `query`, `session_id?`, `top_k?`, `alpha?`, `metadata_filters?`,
          `dynamic_windowing?`, `machine_confirmation?`, `selected_machine?`.
      - Reads JWT from cookie and sends it to backend as `X-User-Token`.
  - **Backend**:
    - `POST /query` (handler `query_knowledge_base`)
    - Request model: `QueryRequest` in `backend/api.py`.
  - **Auth behavior**:
    - Frontend: JWT from cookie → `X-User-Token`.
    - Backend: Uses logging middleware / context and DB to resolve user and role.

### 3. Admin Users

- **List users**
  - **React**:
    - Admin Users page (`frontend/app/admin/users/page.tsx`) calls `GET /api/admin/users`.
  - **Next.js**:
    - `app/api/admin/users/route.ts` (`GET`)
      - Reads JWT from cookie via `extractJwtFromCookie()`.
      - Sends `X-User-Token` to backend.
  - **Backend**:
    - `GET /admin/users` in `backend/routes/admin_routes.py` (via `create_admin_router`).
    - Depends on `get_current_admin` (reads `X-User-Token`, enforces `ADMIN`).

- **Create user**
  - **React**:
    - Admin Users page → `POST /api/admin/users`.
  - **Next.js**:
    - `app/api/admin/users/route.ts` (`POST`)
      - Reads JWT from cookie.
      - Sends `X-User-Token`.
  - **Backend**:
    - `POST /admin/users` (`create_user` in `admin_routes.py`).
    - Requires `ADMIN` via `get_current_admin`.

- **Update user**
  - **React**:
    - Admin Users page → `PUT /api/admin/users/{userId}`.
  - **Next.js**:
    - `app/api/admin/users/[userId]/route.ts` (`PUT`)
      - Uses `getJwtAuthHeaders()` → includes `X-User-Token`.
  - **Backend**:
    - `PUT /admin/edit_user/{user_id}` (`edit_user` in `admin_routes.py`).

- **Delete user**
  - **React**:
    - Admin Users page (`submitDeleteUser`) now calls:
      - `DELETE /api/admin/users/{id}`
  - **Next.js**:
    - `app/api/admin/users/[userId]/route.ts` (`DELETE`)
      - Uses `getJwtAuthHeaders()` → `X-User-Token` header.
      - Proxies to backend `DELETE /admin/users/{userId}` (currently `DELETE /admin/delete_user/{user_id}`).
  - **Backend**:
    - `DELETE /admin/delete_user/{user_id}` in `admin_routes.py`.
    - Requires `ADMIN` via `get_current_admin`.

### 4. Admin Documents

- **List documents (admin view)**
  - **React**:
    - Admin Settings / Documents tab → `GET /api/admin/documents`.
  - **Next.js**:
    - `app/api/admin/documents/route.ts` (`GET`)
      - Reads JWT from cookie.
      - Sends `X-User-Token`.
      - Handles non-JSON error bodies gracefully.
  - **Backend**:
    - `GET /admin/documents` in `backend/api.py`.
      - Auth: decodes `X-User-Token`, enforces `ADMIN`.
      - Does **not** require the RAG pipeline to be initialized; uses DB and filesystem.

### 5. Admin Machine Models

- **List machine models**
  - **React**:
    - Admin “Machines” tab → `GET /api/admin/machines`.
  - **Next.js**:
    - `app/api/admin/machines/route.ts` (`GET`)
      - Uses `getJwtAuthHeaders()` / `X-User-Token`.
  - **Backend**:
    - `GET /admin/machines` in `admin_routes.py`.
    - Requires `ADMIN` via `get_current_admin`.

- **Create / delete machine models**
  - **React**:
    - Admin “Machines” UI (if enabled) → `POST /api/admin/machines`, `DELETE /api/admin/machines/{id}`.
  - **Next.js**:
    - `app/api/admin/machines/route.ts` (`POST`)
    - `app/api/admin/machines/[machineId]/route.ts` (`DELETE`) – if present.
  - **Backend**:
    - `POST /admin/machines`, `DELETE /admin/machines/{machine_id}` in `admin_routes.py`.
    - Both require `ADMIN`.

### 6. Admin Logs (if used)

- **Audit logs**
  - **React**:
    - Admin Logs page → `GET /api/admin/logs`.
  - **Next.js**:
    - `app/api/admin/logs/route.ts` (`GET`)
      - Uses JWT cookie → `X-User-Token`.
  - **Backend**:
    - `GET /admin/logs` in `admin_routes.py`.
    - Requires `ADMIN`.

---

### Auth Standard

- **Client → Next.js**:
  - JWT is stored in an HTTP-only cookie set by `/auth/login`.
  - React fetches to `/api/...` with `credentials: 'include'`.

- **Next.js → Backend**:
  - Admin and query endpoints:
    - Read JWT from cookie using `extractJwtFromCookie()` or `getJwtAuthHeaders()`.
    - Send it as `X-User-Token` to the backend.
  - IAM / Google identity is handled separately via `iamBackend*` helpers and `Authorization` header.


