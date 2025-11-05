# Docker Fixes Applied

## Issues Fixed

1. **OpenSSL Missing**: Added `openssl1.1-compat` or `openssl` to Alpine image (required for Prisma with PostgreSQL)
2. **ts-node Not Found**: Added `ts-node`, `@swc/core`, `@swc/helpers`, and `typescript` as runtime dependencies
3. **Seed Script**: Changed from `npm run prisma:seed` to `npx ts-node prisma/seed.ts` directly

## Next Steps

### Option 1: Create Migration First (Recommended)

Before running Docker, create the initial migration locally:

```powershell
cd frontend
# Set DATABASE_URL to PostgreSQL (or use your local .env)
$env:DATABASE_URL="postgresql://postgres:postgres@localhost:5432/ragdb?schema=public"
npm run prisma:migrate:dev -- --name init
```

This will create the migration file that Docker will use.

### Option 2: Let Docker Create It

The migration will be created automatically when you run:
```powershell
docker compose up --build
```

However, you may need to manually create the admin account after first startup:

```powershell
docker exec -it rag-frontend npx ts-node prisma/seed.ts
```

## After Containers Start

1. Wait for all containers to be healthy
2. Check logs: `docker logs rag-frontend`
3. If admin wasn't created, run seed manually:
   ```powershell
   docker exec -it rag-frontend sh -c "ADMIN_EMAIL=your-email@example.com ADMIN_PASSWORD=your-password npx ts-node prisma/seed.ts"
   ```
4. Go to http://localhost:3000 and login

## Troubleshooting

**If migrations fail:**
- Check PostgreSQL is running: `docker ps | grep postgres`
- Check connection: `docker exec -it rag-postgres psql -U postgres -d ragdb -c "\dt"`

**If seed fails:**
- Check environment variables are set in docker-compose.yml
- Run seed manually with the command above

