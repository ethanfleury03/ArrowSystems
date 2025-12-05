import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    // Extract JWT from cookie and forward as X-User-Token (same pattern as admin routes)
    const cookieStore = await cookies();
    const token = cookieStore.get('access_token')?.value || cookieStore.get('auth_token')?.value;
    
    const headers: Record<string, string> = {};
    if (token) {
      headers['X-User-Token'] = token;
    }
    
    // Proxy to backend /documents endpoint
    // The backend will filter documents based on the authenticated user's machine models
    const response = await iamBackendGet('/documents', headers);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Backend request failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Documents API route error:', error);
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

