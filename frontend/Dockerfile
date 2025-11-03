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

# Copy package files
COPY package.json package-lock.json* ./

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

# Backend URL is set via environment variable at runtime
# (set in docker-compose.yml or when running the container)

# Build the application (with retry for network issues)
RUN npm run build || \
    (echo "Build failed, retrying..." && npm run build)

# Stage 3: Runner
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

# Copy necessary files (standalone output includes public directory)
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static

# Copy public folder if it exists (for static assets)
COPY --from=builder /app/public ./public

# Make sure we can access the standalone server.js
WORKDIR /app

USER nextjs

EXPOSE 3000

ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

CMD ["node", "server.js"]

