import { getIronSession } from 'iron-session';
import { cookies } from 'next/headers';
import { NextRequest, NextResponse } from 'next/server';
import bcrypt from 'bcrypt';
import { prisma } from './prisma';
import { unsealData, sealData } from 'iron-session';

export interface SessionData {
  userId?: string;
}

// Session configuration
export const sessionOptions = {
  password: process.env.SESSION_SECRET || 'change-this-to-a-random-string-at-least-32-characters-long',
  cookieName: 'app_session',
  cookieOptions: {
    secure: process.env.NODE_ENV === 'production',
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
  if (req && res) {
    // For API routes
    return getIronSession<SessionData>(req, res, sessionOptions);
  } else {
    // For server components - parse cookie directly
    const cookieStore = await cookies();
    const sessionCookie = cookieStore.get(sessionOptions.cookieName);
    
    if (!sessionCookie) {
      return {};
    }
    
    try {
      const session = await unsealData(sessionCookie.value, {
        password: sessionOptions.password,
      });
      return session as SessionData;
    } catch (error) {
      console.error('Error parsing session:', error);
      return {};
    }
  }
}

export async function setLoginSession(userId: string, req: NextRequest, res: NextResponse): Promise<NextResponse> {
  const session = await getIronSession<SessionData>(req, res, sessionOptions);
  session.userId = userId;
  await session.save();
  return res;
}

export async function logout(req: NextRequest, res: NextResponse): Promise<NextResponse> {
  const session = await getIronSession<SessionData>(req, res, sessionOptions);
  session.destroy();
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

