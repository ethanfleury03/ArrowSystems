import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(request: NextRequest) {
  try {
    // Extract JWT from cookie
    const token = await extractJwtFromCookie();
    
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }
    
    // Get limit query param if present
    const searchParams = request.nextUrl.searchParams;
    const limit = searchParams.get('limit') || '50';
    
    // Build backend path with limit param
    const backendPath = `/admin/query-insights/recent-queries?limit=${encodeURIComponent(limit)}`;
    
    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendGet(backendPath, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to fetch recent queries' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin query insights recent queries API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

