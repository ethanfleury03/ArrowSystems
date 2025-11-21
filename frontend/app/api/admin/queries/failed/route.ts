import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams;
    let path = '/admin/queries/failed';
    
    // Build query string if params exist
    const params: string[] = [];
    searchParams.forEach((value, key) => {
      params.push(`${key}=${encodeURIComponent(value)}`);
    });
    
    if (params.length > 0) {
      path += '?' + params.join('&');
    }

    const response = await iamBackendGet(path);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to fetch failed queries' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin failed queries API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

