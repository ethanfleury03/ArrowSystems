/**
 * JWT Cookie-Based Authentication Client Utilities
 * 
 * This module provides utilities for JWT cookie management and validation
 * in the Next.js frontend. It works with backend-set HTTP-only cookies.
 * 
 * Note: This module uses Node.js-specific APIs (jsonwebtoken) and should only
 * be imported in API routes or server components, NOT in middleware.
 */

import { cookies } from 'next/headers';
import jwt from 'jsonwebtoken';
import { getAuthCookieName } from './auth-config';

// Re-export getAuthCookieName for compatibility
export { getAuthCookieName };

/**
 * JWT payload structure from backend
 */
export interface JwtPayload {
  email: string;
  role: string;
  exp: number;
  [key: string]: any;
}

/**
 * Extract JWT from cookies (server-side only)
 * 
 * This function can only be called from server components, API routes, or middleware.
 * 
 * @returns JWT token string or null if not found
 */
export async function extractJwtFromCookie(): Promise<string | null> {
  const cookieStore = await cookies();
  const cookieName = getAuthCookieName();
  const token = cookieStore.get(cookieName)?.value;
  return token || null;
}

/**
 * Extract JWT from cookies synchronously (for middleware)
 * 
 * @param cookieString - Cookie string from request.cookies
 * @returns JWT token string or null if not found
 */
export function extractJwtFromCookieSync(cookieString: string): string | null {
  const cookieName = getAuthCookieName();
  const cookies = cookieString.split(';').map(c => c.trim());
  
  for (const cookie of cookies) {
    const [name, value] = cookie.split('=');
    if (name === cookieName) {
      return value;
    }
  }
  
  return null;
}

/**
 * Validate and decode JWT token (client-side validation for routing)
 * 
 * WARNING: This is for UX/routing decisions only. The backend ALWAYS
 * re-validates tokens for actual authorization.
 * 
 * @param token - JWT token string
 * @returns Decoded JWT payload or null if invalid
 */
export function validateJwt(token: string): JwtPayload | null {
  try {
    // Get JWT secret from environment
    const secret = process.env.NEXT_PUBLIC_JWT_SECRET_KEY || process.env.JWT_SECRET_KEY;
    
    if (!secret) {
      console.warn('JWT secret not configured for frontend validation');
      // If no secret, just decode without verification (less secure but UX still works)
      const decoded = jwt.decode(token) as JwtPayload;
      return decoded;
    }
    
    // Verify and decode the token
    const decoded = jwt.verify(token, secret) as JwtPayload;
    
    // Check expiration manually (jwt.verify should handle this, but double-check)
    if (decoded.exp && decoded.exp < Date.now() / 1000) {
      return null;
    }
    
    return decoded;
  } catch (error) {
    // Token is invalid or expired
    console.debug('JWT validation failed:', error instanceof Error ? error.message : 'Unknown error');
    return null;
  }
}

/**
 * Check if a JWT token is expired
 * 
 * @param token - JWT token string or decoded payload
 * @returns true if expired, false if still valid
 */
export function isTokenExpired(token: string | JwtPayload): boolean {
  try {
    let payload: JwtPayload;
    
    if (typeof token === 'string') {
      // Decode without verification for expiration check
      payload = jwt.decode(token) as JwtPayload;
    } else {
      payload = token;
    }
    
    if (!payload || !payload.exp) {
      return true; // No expiration means invalid
    }
    
    // Check if expired (exp is in seconds, Date.now() is in milliseconds)
    return payload.exp < Date.now() / 1000;
  } catch {
    return true; // If we can't decode, treat as expired
  }
}

/**
 * Get user info from JWT token without backend call
 * 
 * Useful for quick client-side checks. For actual user data,
 * always call /api/auth/me which validates with backend.
 * 
 * @param token - JWT token string
 * @returns User info from token or null
 */
export function getUserFromToken(token: string): { email: string; role: string } | null {
  const payload = validateJwt(token);
  if (!payload) {
    return null;
  }
  
  return {
    email: payload.email,
    role: payload.role,
  };
}

