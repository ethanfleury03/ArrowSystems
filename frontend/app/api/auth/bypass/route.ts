import { NextRequest, NextResponse } from 'next/server';
import { setLoginSession } from '@/lib/auth';

const BACKEND_URL =
  process.env.BACKEND_URL ||
  process.env.NEXT_PUBLIC_API_URL ||
  'http://localhost:8000';
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
    const loginResponse = await fetch(`${BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        email: ADMIN_EMAIL,
        password: ADMIN_PASSWORD,
      }),
      cache: 'no-store',
    });

    if (!loginResponse.ok) {
      const detail = await loginResponse.text();
      console.error('Bypass login backend error:', detail);
      return NextResponse.json(
        { error: 'Failed to authenticate bypass user' },
        { status: loginResponse.status }
      );
    }

    const { user } = await loginResponse.json();

    if (!user?.id) {
      return NextResponse.json(
        { error: 'Bypass login failed: user payload missing' },
        { status: 500 }
      );
    }

    // Create response and set session
    const response = NextResponse.json(
      {
        message: 'Bypass login successful',
        userId: user.id,
        email: user.email,
      },
      { status: 200 }
    );

    return await setLoginSession(user.id, request, response);
  } catch (error) {
    console.error('Bypass login error:', error);
    return NextResponse.json(
      { error: 'Internal server error' },
      { status: 500 }
    );
  }
}