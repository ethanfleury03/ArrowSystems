import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

interface RouteParams {
  params: {
    customerId: string;
  };
}

export async function GET(
  request: NextRequest,
  { params }: RouteParams
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
    
    // Get search query param if present
    const searchParams = request.nextUrl.searchParams;
    const search = searchParams.get('search');
    
    // Build backend path with search param if present
    let backendPath = `/admin/query-insights/customers/${params.customerId}/queries`;
    if (search) {
      backendPath += `?search=${encodeURIComponent(search)}`;
    }
    
    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendGet(backendPath, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to fetch customer queries' },
        { status: response.status }
      );
    }

    const backendJson = await response.json();
    console.log("API /api/admin/query-insights/customers/[customerId]/queries backendJson", backendJson);
    return NextResponse.json(backendJson);
  } catch (error) {
    console.error('Admin query insights customer queries API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

