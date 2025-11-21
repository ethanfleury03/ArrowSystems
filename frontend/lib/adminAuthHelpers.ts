/**
 * Admin API Route Authentication Helpers
 * 
 * Utilities for extracting and forwarding JWT tokens in admin API routes
 */

import { NextResponse } from 'next/server';
import { extractJwtFromCookie } from './authClient';

/**
 * Get JWT authentication headers for backend requests
 * 
 * Extracts JWT from cookie and formats it as Authorization header.
 * Returns null if no token is found.
 * 
 * @returns Object with Authorization header or null
 */
export async function getJwtAuthHeaders(): Promise<{ Authorization: string } | null> {
  const token = await extractJwtFromCookie();
  
  if (!token) {
    return null;
  }
  
  return {
    'Authorization': `Bearer ${token}`,
  };
}

/**
 * Create a 401 Unauthorized response for missing/invalid auth
 * 
 * @returns NextResponse with 401 status
 */
export function createUnauthorizedResponse(): NextResponse {
  return NextResponse.json(
    { detail: 'Not authenticated' },
    { status: 401 }
  );
}

/**
 * Wrapper for admin API route handlers that automatically handles JWT extraction
 * 
 * Use this to wrap your route handler function. It will:
 * 1. Extract JWT from cookie
 * 2. Return 401 if not authenticated
 * 3. Call your handler with the auth headers
 * 
 * Example:
 * ```typescript
 * export const GET = withAdminAuth(async (request, authHeaders) => {
 *   const response = await iamBackendGet('/admin/users', authHeaders);
 *   const data = await response.json();
 *   return NextResponse.json(data);
 * });
 * ```
 */
export function withAdminAuth<T extends any[]>(
  handler: (request: any, authHeaders: { Authorization: string }, ...args: T) => Promise<NextResponse>
) {
  return async (request: any, ...args: T): Promise<NextResponse> => {
    const authHeaders = await getJwtAuthHeaders();
    
    if (!authHeaders) {
      return createUnauthorizedResponse();
    }
    
    return handler(request, authHeaders, ...args);
  };
}

