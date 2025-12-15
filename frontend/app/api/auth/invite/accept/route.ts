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

    // Backend no longer returns JWT - user must sign in via /auth/login
    // Simply forward the success response
    return NextResponse.json(data, { status: 200 });
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

