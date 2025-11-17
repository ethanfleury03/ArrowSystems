/**
 * Server-side helper to detect backend URL from request hostname.
 * This allows the Next.js API routes to work when accessed from network IP addresses.
 */

import { NextRequest } from 'next/server';
import { isProd } from './env';

export function getBackendUrl(request?: NextRequest | Request | { headers: Headers | { get: (name: string) => string | null } }): string {
  // Production: NEXT_PUBLIC_API_URL must be provided (validated at build time)
  if (isProd) {
    const apiUrl = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL;
    if (!apiUrl) {
      throw new Error('NEXT_PUBLIC_API_URL is required in production but was not set');
    }
    return apiUrl;
  }

  // Development: Check environment variables first
  if (process.env.BACKEND_URL) {
    return process.env.BACKEND_URL;
  }
  if (process.env.NEXT_PUBLIC_API_URL) {
    return process.env.NEXT_PUBLIC_API_URL;
  }
  
  // If request is provided, detect from hostname
  if (request) {
    try {
      // Try to get headers from request
      // NextRequest has a 'headers' property that is a Headers object
      let host = '';
      let forwardedHost = '';
      
      if (request && typeof request === 'object' && 'headers' in request) {
        // Handle NextRequest or Request objects
        const headers = request.headers as Headers;
        if (headers && typeof headers.get === 'function') {
          host = headers.get('host') || headers.get('Host') || '';
          forwardedHost = headers.get('x-forwarded-host') || headers.get('X-Forwarded-Host') || '';
        }
      } else if (request && typeof request === 'object' && 'get' in request) {
        // Handle headers-like object
        const headersObj = request as { get: (name: string) => string | null };
        host = headersObj.get('host') || headersObj.get('Host') || '';
        forwardedHost = headersObj.get('x-forwarded-host') || headersObj.get('X-Forwarded-Host') || '';
      }
      
      const hostname = (forwardedHost || host || '').trim();
      
      if (hostname) {
        // Extract hostname (remove port if present)
        const hostnameOnly = hostname.split(':')[0].trim();
        
        // If accessing from localhost or 127.0.0.1, use localhost:8000
        if (hostnameOnly === 'localhost' || hostnameOnly === '127.0.0.1' || hostnameOnly === '') {
          return 'http://localhost:8000';
        }
        
        // Otherwise, use the same hostname but port 8000 for backend
        // This allows network access: if frontend is at 192.168.1.100:3000,
        // backend will be at 192.168.1.100:8000
        return `http://${hostnameOnly}:8000`;
      }
    } catch (error) {
      console.warn('Failed to detect backend URL from request:', error);
    }
  }
  
  // Fallback to localhost for development
  return 'http://localhost:8000';
}

