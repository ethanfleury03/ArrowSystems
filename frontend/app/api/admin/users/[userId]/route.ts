import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPut, iamBackendDelete } from '@/lib/iam-backend';
import { getJwtAuthHeaders, createUnauthorizedResponse } from '@/lib/adminAuthHelpers';

export async function PUT(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    const body = await request.json();
    
    // Get JWT auth headers
    const authHeaders = await getJwtAuthHeaders();
    if (!authHeaders) {
      return createUnauthorizedResponse();
    }

    const response = await iamBackendPut(`/admin/users/${params.userId}`, body, authHeaders);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to update user' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin user update API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function DELETE(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    // Get JWT auth headers
    const authHeaders = await getJwtAuthHeaders();
    if (!authHeaders) {
      return createUnauthorizedResponse();
    }

    const response = await iamBackendDelete(`/admin/users/${params.userId}`, authHeaders);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to delete user' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin user delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

