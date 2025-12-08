import { NextRequest, NextResponse } from 'next/server';
import { cookies } from 'next/headers';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '50';
    
    // Extract JWT from cookie and forward as X-User-Token (same pattern as admin routes)
    const cookieStore = await cookies();
    const token = cookieStore.get('access_token')?.value || cookieStore.get('auth_token')?.value;
    
    const headers: Record<string, string> = {};
    if (token) {
      headers['X-User-Token'] = token;
    }
    
    // Backend will extract user email from JWT token, so we don't need to pass user param
    const response = await iamBackendGet(`/history?limit=${limit}`, headers);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Backend request failed' },
        { status: response.status }
      );
    }

    const backendJson = await response.json();
    console.log("/api/history backendJson", backendJson);
    return NextResponse.json(backendJson);
  } catch (error) {
    console.error('API route error:', error);
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

