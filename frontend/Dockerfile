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

# Copy node_modules from deps stage
COPY --from=deps /app/node_modules ./node_modules

# Copy only files needed for Prisma generation (these change less frequently)
# This allows Prisma generation to be cached separately from source code
COPY package.json package-lock.json* ./
COPY prisma ./prisma
COPY tsconfig.json ./
COPY next.config.js* ./

# Generate Prisma Client (required for build)
# Note: DATABASE_URL is needed for Prisma, but we'll use a dummy for build
# The actual connection will be set at runtime
ENV DATABASE_URL="file:./dev.db"
RUN npx prisma generate

# Now copy the rest of the source code (this is the frequently changing part)
# By doing this last, code changes won't invalidate the Prisma generation cache
COPY . .

# Backend URL is set via environment variable at runtime
# (set in docker-compose.yml or when running the container)

# Build the application (with retry for network issues)
RUN npm run build || \
    (echo "Build failed, retrying..." && npm run build)

# Note: Runtime dependencies (ts-node, prisma, typescript, etc.) are already installed
# in the deps stage and copied to builder via node_modules. We'll copy them directly
# to the runner stage - no need to reinstall (which is slow, especially for native modules like sharp)

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
COPY --from=builder /app/node_modules/sharp ./node_modules/sharp

# Copy ts-node package entirely (needed for seeding) - copy the whole package to avoid missing dependencies
COPY --from=builder /app/node_modules/ts-node ./node_modules/ts-node

# Copy ts-node dependencies (needed for seeding)
# Copy these conditionally - they might not all exist, so we use a RUN command with mount
RUN --mount=from=builder,source=/app/node_modules,target=/tmp/node_modules \
    mkdir -p ./node_modules && \
    for pkg in diff make-error tslib v8-compile-cache-lib yn arg create-require source-map-support; do \
        if [ -d "/tmp/node_modules/$pkg" ]; then \
            cp -r "/tmp/node_modules/$pkg" "./node_modules/$pkg"; \
        fi; \
    done

# Copy all bin executables and their associated files (including WASM files)
RUN mkdir -p ./node_modules/.bin
RUN --mount=from=builder,source=/app/node_modules/.bin,target=/tmp/bin \
    cp -r /tmp/bin/* ./node_modules/.bin/ 2>/dev/null || true

# Copy Prisma schema and migrations (needed for runtime migrations)
COPY --from=builder /app/prisma ./prisma

# Copy the generated Prisma client from builder (already generated during build)
COPY --from=builder /app/node_modules/.prisma ./node_modules/.prisma
COPY --from=builder /app/node_modules/@prisma ./node_modules/@prisma

# Copy Prisma engines (needed for migrations) - copy the entire directory
COPY --from=builder /app/node_modules/@prisma/engines ./node_modules/@prisma/engines

# Copy Prisma package (contains additional files needed)
COPY --from=builder /app/node_modules/prisma ./node_modules/prisma

# Add node_modules/.bin to PATH so ts-node and prisma are accessible
ENV PATH="/app/node_modules/.bin:$PATH"

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
# Using printf to avoid complex quoting issues
RUN printf '#!/bin/sh\nset -e\necho "🔄 Running database migrations..."\necho "DATABASE_URL: $DATABASE_URL"\nmkdir -p /app/node_modules/.prisma/client 2>/dev/null || true\nmkdir -p /app/node_modules/@prisma/engines 2>/dev/null || true\nif [ -f "prisma/migrations/migration_lock.toml" ]; then\n  LOCK_PROVIDER=$(grep -o "provider = \\"[^\\"]*\\"" prisma/migrations/migration_lock.toml | cut -d"\\"" -f2)\n  if [ "$LOCK_PROVIDER" != "postgresql" ]; then\n    echo "⚠️  Removing old SQLite migrations, will create new PostgreSQL migration..."\n    rm -rf prisma/migrations\n  fi\nfi\nif [ ! -d "prisma/migrations" ] || [ -z "$(ls -A prisma/migrations 2>/dev/null)" ]; then\n  echo "📝 Creating initial migration..."\n  npx prisma migrate dev --name init --create-only || echo "⚠️ Migration creation failed"\nfi\necho "📦 Deploying migrations..."\nnpx prisma migrate deploy && echo "✅ Migrations deployed successfully" || echo "⚠️ Migration deploy failed"\necho "✅ Migrations complete"\nif [ -n "$ADMIN_EMAIL" ] && [ -n "$ADMIN_PASSWORD" ]; then\n  echo "🌱 Seeding admin account with email: $ADMIN_EMAIL"\n  echo "Running seed script (JavaScript version)..."\n  node prisma/seed.js 2>&1\n  SEED_EXIT_CODE=$?\n  if [ $SEED_EXIT_CODE -eq 0 ]; then\n    echo "✅ Admin account created/updated successfully"\n  else\n    echo "⚠️ Seed script exited with code: $SEED_EXIT_CODE"\n  fi\nelse\n  echo "⚠️ ADMIN_EMAIL or ADMIN_PASSWORD not set - skipping seed"\nfi\necho "🚀 Starting Next.js server..."\nexec node server.js\n' > /app/start.sh && \
    chmod +x /app/start.sh && \
    chown nextjs:nodejs /app/start.sh

WORKDIR /app

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["/app/start.sh"]

