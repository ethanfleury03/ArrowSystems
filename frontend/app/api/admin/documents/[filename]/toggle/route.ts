import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function POST(
  request: NextRequest,
  { params }: { params: { filename: string } }
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

    const body = await request.json();

    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendPost(
      `/admin/documents/${encodeURIComponent(params.filename)}/toggle`, 
      body,
      {
        'X-User-Token': token,
      }
    );

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to toggle document status' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin document toggle API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

