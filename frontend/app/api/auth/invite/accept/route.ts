import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { token, password } = body;

    if (!token || !password) {
      return NextResponse.json(
        { detail: 'Missing token or password' },
        { status: 400 }
      );
    }

    const backendResponse = await iamBackendPost('/auth/invite/accept', {
      token,
      password,
    });

    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => null);
      return NextResponse.json(
        errorBody || { detail: 'Failed to accept invite' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();

    // Extract JWT token from backend's Set-Cookie header (mirror /api/auth/login logic)
    let jwtToken: string | null = null;
    const setCookieHeader = backendResponse.headers.get('set-cookie');
    if (setCookieHeader) {
      const match = setCookieHeader.match(/access_token=([^;]+)/);
      if (match) {
        jwtToken = match[1];
      }
    }

    const response = NextResponse.json(data, { status: 200 });

    // Set JWT cookie on the frontend domain (mirror /api/auth/login behavior)
    if (jwtToken) {
      response.cookies.set('access_token', jwtToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
        path: '/',
        // No maxAge = session cookie (expires when browser closes)
      });
    }

    return response;
  } catch (error) {
    console.error('Invite accept error:', error);
    return NextResponse.json(
      {
        detail: 'Internal server error',
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

