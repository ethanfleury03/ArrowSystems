import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPut, iamBackendDelete } from '@/lib/iam-backend';
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

export async function DELETE(
  request: NextRequest,
  { params }: { params: { machineId: string } }
) {
  try {
    // Extract JWT from cookie and forward as X-User-Token
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    const response = await iamBackendDelete(`/admin/machines/${params.machineId}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to delete machine model' }));
      return NextResponse.json(
        { detail: error.detail || 'Failed to delete machine model' },
        { status: response.status }
      );
    }

    // 204 No Content - return empty response
    return new NextResponse(null, { status: 204 });
  } catch (error) {
    console.error('Admin machine delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

