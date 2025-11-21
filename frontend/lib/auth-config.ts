/**
 * Authentication Configuration
 * 
 * Simple configuration utilities that don't require Node.js APIs.
 * Safe to use in middleware and Edge Runtime.
 */

/**
 * Get the configured auth cookie name from environment or use default
 */
export function getAuthCookieName(): string {
  return process.env.AUTH_COOKIE_NAME || 'access_token';
}

