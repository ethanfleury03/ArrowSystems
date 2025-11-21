import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';
import { unsealData, sealData } from 'iron-session';
import { isProd } from './env';
import { getBackendUrl } from './backend-url';

// Get backend URL (validated based on environment)
const BACKEND_URL = (() => {
  try {
    return getBackendUrl();
  } catch {
    // Fallback for server-side usage where request is not available
    return process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';
  }
})();

export interface SessionData {
  userId?: string;
}

// Get session secret with validation
const getSessionSecret = (): string => {
  const envSecret = process.env.SESSION_SECRET;
  
  if (isProd) {
    // Production: require SESSION_SECRET
    if (!envSecret) {
      throw new Error(
        'SESSION_SECRET environment variable is required in production. ' +
        'Set a secure random string of at least 32 characters.'
      );
    }
    
    // Check for unsafe defaults
    const unsafeDefaults = [
      'change-this-to-a-random-string-at-least-32-characters-long',
      'dev-session-secret-not-for-production',
      'secret',
      'password',
      'default-secret',
    ];
    
    if (unsafeDefaults.includes(envSecret) || envSecret.length < 32) {
      throw new Error(
        'SESSION_SECRET is set to an unsafe default or is too short. ' +
        'In production, SESSION_SECRET must be at least 32 characters ' +
        'and not be a common default value.'
      );
    }
    
    return envSecret;
  }
  
  // Development: allow fallback to dev secret
  return envSecret || 'dev-session-secret-not-for-production-use-only';
};

// Session configuration
// Secure flag: only true if explicitly set via env var AND we're on HTTPS
// For local network access (IP addresses), allow HTTP cookies
export const sessionOptions = {
  password: getSessionSecret(),
  cookieName: 'app_session',
  cookieOptions: {
    // Only use secure cookies if explicitly enabled AND on HTTPS
    // This allows local network access via IP address (HTTP)
    secure: process.env.FORCE_SECURE_COOKIES === 'true',
    httpOnly: true,
    sameSite: 'lax' as const,
    maxAge: 60 * 60 * 24 * 7, // 7 days
  },
};

// Session helpers for Next.js App Router
export async function getSession(req?: NextRequest, res?: NextResponse): Promise<SessionData> {
  let sessionCookie: string | undefined;
  
  if (req) {
    // For API routes - get cookie from request
    sessionCookie = req.cookies.get(sessionOptions.cookieName)?.value;
  } else {
    // For server components - get cookie from cookies()
    const cookieStore = await cookies();
    sessionCookie = cookieStore.get(sessionOptions.cookieName)?.value;
  }
  
  if (!sessionCookie) {
    return {};
  }
  
  try {
    const session = await unsealData(sessionCookie, {
      password: sessionOptions.password,
    });
    return session as SessionData;
  } catch (error) {
    console.error('Error parsing session:', error);
    return {};
  }
}

export async function setLoginSession(userId: string, req: NextRequest, res: NextResponse): Promise<NextResponse> {
  const session: SessionData = { userId };
  
  // Seal the session data with iron-session compatible options
  const sealed = await sealData(session, {
    password: sessionOptions.password,
    ttl: sessionOptions.cookieOptions.maxAge,
  });
  
  // Detect if request is HTTPS (for secure cookie flag)
  const isHttps = req.url.startsWith('https://') || 
                  req.headers.get('x-forwarded-proto') === 'https' ||
                  req.headers.get('x-forwarded-ssl') === 'on';
  
  // Use secure flag only if explicitly enabled AND on HTTPS
  // This allows local network access via IP address (HTTP)
  const useSecure = process.env.FORCE_SECURE_COOKIES === 'true' && isHttps;
  
  // Set the cookie with proper options
  res.cookies.set(sessionOptions.cookieName, sealed, {
    httpOnly: sessionOptions.cookieOptions.httpOnly,
    secure: useSecure,
    sameSite: sessionOptions.cookieOptions.sameSite,
    maxAge: sessionOptions.cookieOptions.maxAge,
    path: '/',
  });
  
  return res;
}

export async function logout(req: NextRequest, res: NextResponse): Promise<NextResponse> {
  // Delete the session cookie
  res.cookies.delete(sessionOptions.cookieName);
  return res;
}

export async function getUserFromSession() {
  const session = await getSession();
  if (!session.userId) {
    return null;
  }

  try {
    const response = await fetch(`/api/auth/users/${session.userId}`, {
      cache: 'no-store',
    });
    if (!response.ok) {
      return null;
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching user from session:', error);
    return null;
  }
}

