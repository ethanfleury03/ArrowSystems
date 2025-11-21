import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

const ADMIN_EMAIL = process.env.ADMIN_EMAIL || process.env.SEED_ADMIN_EMAIL;
const ADMIN_PASSWORD =
  process.env.ADMIN_PASSWORD || process.env.SEED_ADMIN_PASSWORD;

// Quick bypass endpoint for development/testing
// Only works in development mode
export async function POST(request: NextRequest) {
  if (!ADMIN_EMAIL || !ADMIN_PASSWORD) {
    return NextResponse.json(
      { error: 'Bypass login not configured. Set ADMIN_EMAIL and ADMIN_PASSWORD.' },
      { status: 503 }
    );
  }

  try {
    // Call backend login - backend will set JWT cookie
    const loginResponse = await iamBackendPost('/auth/login', {
      email: ADMIN_EMAIL,
      password: ADMIN_PASSWORD,
    });

    if (!loginResponse.ok) {
      const detail = await loginResponse.text();
      console.error('Bypass login backend error:', detail);
      return NextResponse.json(
        { error: 'Failed to authenticate bypass user' },
        { status: loginResponse.status }
      );
    }

    const { user, message } = await loginResponse.json();

    if (!user?.id) {
      return NextResponse.json(
        { error: 'Bypass login failed: user payload missing' },
        { status: 500 }
      );
    }

    // Forward Set-Cookie header from backend to browser
    const setCookieHeader = loginResponse.headers.get('set-cookie');
    
    const response = NextResponse.json(
      {
        message: message || 'Bypass login successful',
        userId: user.id,
        email: user.email,
        user,
      },
      { status: 200 }
    );

    // Forward the cookie from backend
    if (setCookieHeader) {
      response.headers.set('set-cookie', setCookieHeader);
    }

    return response;
  } catch (error) {
    console.error('Bypass login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}