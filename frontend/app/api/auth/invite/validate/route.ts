import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export const dynamic = 'force-dynamic';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const token = searchParams.get('token');

    if (!token) {
      return NextResponse.json(
        { detail: 'Missing token' },
        { status: 400 }
      );
    }

    const backendResponse = await iamBackendGet(
      `/auth/invite/validate?token=${encodeURIComponent(token)}`
    );

    if (!backendResponse.ok) {
      const errorBody = await backendResponse.json().catch(() => null);
      return NextResponse.json(
        errorBody || { detail: 'Failed to validate invite token' },
        { status: backendResponse.status }
      );
    }

    const data = await backendResponse.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Invite validate error:', error);
    return NextResponse.json(
      {
        detail: 'Internal server error',
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

