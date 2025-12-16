import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPut } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function PUT(
  request: NextRequest,
  { params }: { params: { machineId: string } }
) {
  try {
    const body = await request.json();

    // Extract JWT from cookie and forward as X-User-Token
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    const response = await iamBackendPut(`/admin/machines/${params.machineId}`, body, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to update machine' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin machine update API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

