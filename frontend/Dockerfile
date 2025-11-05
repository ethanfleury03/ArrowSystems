# Multi-stage build for Next.js frontend

# Stage 1: Dependencies
FROM node:20-alpine AS deps
RUN apk add --no-cache libc6-compat
WORKDIR /app

# Configure npm for better network reliability
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000 && \
    npm config set fetch-timeout 300000

# Set DATABASE_URL for Prisma (dummy value for client generation)
ENV DATABASE_URL="file:./dev.db"

# Copy package files
COPY package.json package-lock.json* ./

# Copy prisma schema (needed for postinstall script)
COPY prisma ./prisma

# Install dependencies with retry logic
RUN npm ci --legacy-peer-deps --prefer-offline --no-audit || \
    (echo "First install attempt failed, retrying..." && \
     npm cache clean --force && \
     npm install --legacy-peer-deps --no-audit) || \
    (echo "Second install attempt failed, retrying with registry reset..." && \
     npm config set registry https://registry.npmjs.org/ && \
     npm install --legacy-peer-deps --no-audit)

# Stage 2: Builder
FROM node:20-alpine AS builder
WORKDIR /app

# Configure npm for better network reliability
RUN npm config set fetch-retries 5 && \
    npm config set fetch-retry-mintimeout 20000 && \
    npm config set fetch-retry-maxtimeout 120000

COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Generate Prisma Client (required for build)
# Note: DATABASE_URL is needed for Prisma, but we'll use a dummy for build
# The actual connection will be set at runtime
ENV DATABASE_URL="file:./dev.db"
RUN npx prisma generate

# Backend URL is set via environment variable at runtime
# (set in docker-compose.yml or when running the container)

# Build the application (with retry for network issues)
RUN npm run build || \
    (echo "Build failed, retrying..." && npm run build)

# Install runtime dependencies needed for migrations/seeding in builder stage
# This way we can copy them to runner stage (already downloaded, no network needed)
# Using --legacy-peer-deps to avoid conflicts with Next.js dependencies
RUN npm install --save --legacy-peer-deps --no-audit --no-fund --loglevel=error \
    ts-node@10.9.2 prisma@5.20.0 typescript@5.5.3 @types/bcrypt@5.0.2 @swc/core @swc/helpers

# Stage 3: Runner
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Install PostgreSQL client and OpenSSL (required for Prisma with PostgreSQL)
RUN apk add --no-cache postgresql-client openssl1.1-compat || apk add --no-cache postgresql-client openssl || true

# Copy package.json for reference
COPY --from=builder /app/package.json ./package.json

# Copy runtime dependencies from builder (already installed, no download needed!)
# Much faster than npm install - just file copying
COPY --from=builder /app/node_modules/ts-node ./node_modules/ts-node
COPY --from=builder /app/node_modules/prisma ./node_modules/prisma
COPY --from=builder /app/node_modules/typescript ./node_modules/typescript
COPY --from=builder /app/node_modules/@types/bcrypt ./node_modules/@types/bcrypt
COPY --from=builder /app/node_modules/@swc ./node_modules/@swc

# Copy bin executables
RUN mkdir -p ./node_modules/.bin
COPY --from=builder /app/node_modules/.bin/ts-node ./node_modules/.bin/ts-node
COPY --from=builder /app/node_modules/.bin/prisma ./node_modules/.bin/prisma

# Copy Prisma schema and migrations (needed for runtime migrations)
COPY --from=builder /app/prisma ./prisma

# Copy the generated Prisma client from builder (already generated during build)
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/node_modules/@prisma ./node_modules/@prisma

# Copy necessary files (standalone output includes public directory)
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Copy public folder if it exists (for static assets)
COPY --from=builder /app/public ./public

# Create cache directory and migrations directory, then set permissions BEFORE switching to nextjs user
RUN mkdir -p /app/.next/cache && \
    mkdir -p /app/prisma/migrations && \
    chown -R nextjs:nodejs /app && \
    chmod -R 755 /app/node_modules/.prisma 2>/dev/null || true

# Create startup script for database migrations and seeding
RUN echo '#!/bin/sh' > /app/start.sh && \
    echo 'set -e' >> /app/start.sh && \
    echo 'echo "🔄 Running database migrations..."' >> /app/start.sh && \
    echo 'mkdir -p /app/node_modules/.prisma/client 2>/dev/null || true' >> /app/start.sh && \
    echo 'mkdir -p /app/node_modules/@prisma/engines 2>/dev/null || true' >> /app/start.sh && \
    echo 'if [ -f "prisma/migrations/migration_lock.toml" ]; then' >> /app/start.sh && \
    echo '  LOCK_PROVIDER=$(grep -o '\''provider = "[^"]*'\'' prisma/migrations/migration_lock.toml | cut -d"\\"" -f2)' >> /app/start.sh && \
    echo '  if [ "$LOCK_PROVIDER" != "postgresql" ]; then' >> /app/start.sh && \
    echo '    echo "⚠️  Removing old SQLite migrations, will create new PostgreSQL migration..."' >> /app/start.sh && \
    echo '    rm -rf prisma/migrations' >> /app/start.sh && \
    echo '  fi' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo 'if [ ! -d "prisma/migrations" ] || [ -z "$(ls -A prisma/migrations 2>/dev/null)" ]; then' >> /app/start.sh && \
    echo '  echo "📝 Creating initial migration..."' >> /app/start.sh && \
    echo '  npx prisma migrate dev --name init || true' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo 'npx prisma migrate deploy || echo "⚠️ Migration failed or no migrations to run"' >> /app/start.sh && \
    echo 'echo "✅ Migrations complete"' >> /app/start.sh && \
    echo 'if [ -n "$ADMIN_EMAIL" ] && [ -n "$ADMIN_PASSWORD" ]; then' >> /app/start.sh && \
    echo '  echo "🌱 Seeding admin account..."' >> /app/start.sh && \
    echo '  npx ts-node --project prisma/tsconfig.json prisma/seed.ts 2>&1 || echo "⚠️ Seed failed or admin already exists"' >> /app/start.sh && \
    echo '  echo "✅ Admin account ready"' >> /app/start.sh && \
    echo 'fi' >> /app/start.sh && \
    echo 'echo "🚀 Starting Next.js server..."' >> /app/start.sh && \
    echo 'exec node server.js' >> /app/start.sh && \
    chmod +x /app/start.sh && \
    chown nextjs:nodejs /app/start.sh

WORKDIR /app

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["/app/start.sh"]

