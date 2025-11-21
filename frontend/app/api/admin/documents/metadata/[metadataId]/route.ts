import { NextRequest, NextResponse } from 'next/server';
import { iamBackendDelete } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function DELETE(
  request: NextRequest,
  { params }: { params: { metadataId: string } }
) {
  try {
    // Extract JWT from cookie
    const token = await extractJwtFromCookie();
    
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    // Forward JWT in Authorization header to backend
    const response = await iamBackendDelete(
      `/admin/documents/metadata/${encodeURIComponent(params.metadataId)}`,
      {
        'Authorization': `Bearer ${token}`,
      }
    );

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to delete document' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin document delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

