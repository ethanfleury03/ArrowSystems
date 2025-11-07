import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';
import bcrypt from 'bcrypt';
import { prisma } from './prisma';
import { unsealData, sealData } from 'iron-session';

export interface SessionData {
  userId?: string;
}

// Session configuration
// Secure flag: only true if explicitly set via env var AND we're on HTTPS
// For local network access (IP addresses), allow HTTP cookies
export const sessionOptions = {
  password: process.env.SESSION_SECRET || 'change-this-to-a-random-string-at-least-32-characters-long',
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

// Password hashing
export async function hashPassword(plainPassword: string): Promise<string> {
  return bcrypt.hash(plainPassword, 10);
}

export async function verifyPassword(plainPassword: string, hash: string): Promise<boolean> {
  return bcrypt.compare(plainPassword, hash);
}

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
    const user = await prisma.user.findUnique({
      where: { id: session.userId },
      select: {
        id: true,
        email: true,
        role: true,
        createdAt: true,
      },
    });
    return user;
  } catch (error) {
    console.error('Error fetching user from session:', error);
    return null;
  }
}

