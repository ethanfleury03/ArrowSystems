# Build argument to determine environment
ARG NODE_ENV=production

# =============================================================================
# Dependencies Stage
# =============================================================================
FROM node:20-alpine AS deps
WORKDIR /app
RUN apk add --no-cache libc6-compat
COPY package.json package-lock.json* ./
RUN npm install --legacy-peer-deps --prefer-offline --no-audit || \
    (npm cache clean --force && npm install --legacy-peer-deps --no-audit)

# =============================================================================
# Builder Stage (Production only)
# =============================================================================
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}
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
# Production Runner Stage
# =============================================================================
FROM node:20-alpine AS production
WORKDIR /app
ARG NODE_ENV=production
ENV NODE_ENV=${NODE_ENV}
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Copy built artifacts from builder
COPY --from=builder /app/public ./public
COPY --from=builder /app/.next ./.next
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/package.json ./package.json

RUN mkdir -p /app/.next/cache/images && \
    chown -R nextjs:nodejs /app/.next

USER nextjs

EXPOSE 3000

CMD ["npm", "run", "start"]

