# =============================================================================
# Dependencies Stage - Install dependencies
# =============================================================================
FROM node:20-alpine AS deps
WORKDIR /app
RUN apk add --no-cache libc6-compat
COPY package.json package-lock.json* ./
RUN npm ci --legacy-peer-deps --prefer-offline --no-audit || \
    (npm cache clean --force && npm ci --legacy-peer-deps --no-audit)

# =============================================================================
# Builder Stage - Build the app (creates .next/standalone)
# =============================================================================
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NODE_ENV=production
ARG NEXT_PUBLIC_API_URL
ARG SESSION_SECRET
ENV NODE_ENV=${NODE_ENV}
ENV NEXT_PUBLIC_API_URL=${NEXT_PUBLIC_API_URL}
ENV SESSION_SECRET=${SESSION_SECRET}
# Build the app - this creates .next/standalone directory
# Note: google-auth-library and form-data are already installed via npm ci from package.json
RUN npm run build

# =============================================================================
# Development Stage
# =============================================================================
FROM node:20-alpine AS development
WORKDIR /app
ARG NODE_ENV=development
ENV NODE_ENV=${NODE_ENV}
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Copy dependencies and source code for hot reload
COPY --from=deps /app/node_modules ./node_modules
COPY . .

EXPOSE 3000
CMD ["npm", "run", "dev"]

# =============================================================================
# Production Runner Stage - Cloud Run Optimized (Standalone Mode)
# =============================================================================
FROM node:20-alpine AS production
WORKDIR /app

ENV NODE_ENV=production
# PORT is set by Cloud Run automatically - don't override it
ENV HOSTNAME="0.0.0.0"

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy only the standalone output from builder
# This contains only the minimal runtime files needed
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
# Copy public folder for static assets
COPY --from=builder --chown=nextjs:nodejs /app/public ./public
# Copy .next/static for static assets
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

# Copy sharp for image optimization in standalone mode
# Sharp needs its native bindings which aren't included in standalone by default
COPY --from=builder --chown=nextjs:nodejs /app/node_modules/sharp ./node_modules/sharp
# Copy @img packages (optional dependencies for sharp native binaries)
COPY --from=builder --chown=nextjs:nodejs /app/node_modules/@img ./node_modules/@img

USER nextjs

# Cloud Run sets PORT automatically (typically 3000)
# EXPOSE is informational - actual port comes from $PORT env var
EXPOSE 3000

# Use node directly to run the standalone server (faster than npm start)
# The standalone server.js respects PORT env var automatically
CMD ["node", "server.js"]

