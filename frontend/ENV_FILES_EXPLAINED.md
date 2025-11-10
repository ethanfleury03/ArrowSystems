# Environment Files Explained

## Which .env File is Used?

For **Docker Compose** (what you're using), the important file is:

### **Root `.env` file** (in the project root: `C:\Users\ethan\ArrowSystems\.env`)

This is the **ONLY** file that Docker Compose reads. The `docker-compose.yml` file uses `${VARIABLE_NAME}` syntax which reads from the root `.env` file.

### Why You Have Multiple .env Files

1. **Root `.env`** - Used by Docker Compose ✅ **THIS IS THE ONE YOU NEED**
2. **`frontend/.env`** - Not used by Docker (Docker sets env vars directly)
3. **`frontend/.env.local`** - Not used by Docker (only for local Next.js dev)
4. **`frontend/.env.production`** - Not used by Docker (only for local Next.js dev)

## What You Need to Do

Create or edit the **root `.env` file** (`C:\Users\ethan\ArrowSystems\.env`) with:

```env
# Database connection (for frontend)
DATABASE_URL=postgresql://postgres:postgres@postgres:5432/ragdb?schema=public

# Session secret (generate a random 32+ character string)
SESSION_SECRET=your-super-secret-random-string-at-least-32-characters-long-change-this

# Admin account credentials (used by seed script)
ADMIN_EMAIL=admin@example.com
ADMIN_PASSWORD=admin123

# Backend API key (if needed)
ANTHROPIC_API_KEY=your-api-key-here

# PostgreSQL settings (optional, defaults are fine)
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=ragdb
```

## How Docker Compose Uses It

In `docker-compose.yml`, you'll see:
```yaml
environment:
  - DATABASE_URL=${DATABASE_URL:-default-value}
```

This means:
- Read `DATABASE_URL` from the root `.env` file
- If not found, use the default value after `:-`

## For Local Development (Without Docker)

If you run `npm run dev` directly in the `frontend` folder, then Next.js reads:
1. `.env.local` (highest priority)
2. `.env.production` or `.env.development` (depending on NODE_ENV)
3. `.env` (lowest priority)

But since you're using Docker, **only the root `.env` matters**.

## Quick Fix

1. Create/edit `C:\Users\ethan\ArrowSystems\.env`
2. Add the variables above
3. Restart Docker: `docker compose down && docker compose up --build`



