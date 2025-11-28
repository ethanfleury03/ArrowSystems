/**
 * DEPRECATED: This file has been replaced by authClient.ts
 * 
 * The old iron-session based authentication has been migrated to
 * backend-owned JWT cookies. See:
 * - frontend/lib/authClient.ts for new auth utilities
 * - docs/auth-architecture.md for documentation
 * - docs/AUTH_MIGRATION_SUMMARY.md for migration details
 * 
 * This file is kept as a stub to prevent build errors during transition.
 * It can be safely deleted once all references are removed.
 */

// Re-export auth client for any lingering imports (temporary compatibility)
export { extractJwtFromCookie, validateJwt, getAuthCookieName } from './authClient';
