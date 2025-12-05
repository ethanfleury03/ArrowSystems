import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPut, iamBackendDelete } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function PUT(
  request: NextRequest,
  { params }: { params: { userId: string } }
) {
  try {
    const body = await request.json();

    // Extract JWT from cookie and forward as X-User-Token (same pattern as list users)
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    const response = await iamBackendPut(`/admin/users/${params.userId}`, body, {
      'X-User-Token': token,
    });

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
    // Extract JWT from cookie and forward as X-User-Token (same pattern as list users)
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    const response = await iamBackendDelete(`/admin/users/${params.userId}`, {
      'X-User-Token': token,
    });

    // Handle 204 No Content: backend successfully deleted the user
    // Return 200 with success JSON (204 cannot have a body in NextResponse)
    if (response.status === 204) {
      return NextResponse.json({ success: true }, { status: 200 });
    }

    // For error responses, try to parse error details
    if (!response.ok) {
      let error: any = {};
      try {
        // Only try to parse JSON if there's actually content
        const text = await response.text();
        if (text) {
          error = JSON.parse(text);
        }
      } catch {
        // If parsing fails, use default error
        error = { detail: 'Failed to delete user' };
      }
      return NextResponse.json(
        { detail: error.detail || 'Failed to delete user' },
        { status: response.status }
      );
    }

    // For other success statuses, parse and return data
    let data: any = {};
    try {
      const text = await response.text();
      if (text) {
        data = JSON.parse(text);
      }
    } catch {
      // If parsing fails, return success indicator
      data = { success: true };
    }
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin user delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

