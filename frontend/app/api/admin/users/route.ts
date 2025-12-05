import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet, iamBackendPost, iamBackendPut, iamBackendDelete } from '@/lib/iam-backend';
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
    
    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendGet('/admin/users', {
      'X-User-Token': token,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to fetch users' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin users API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    
    // Extract JWT from cookie
    const token = await extractJwtFromCookie();
    
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }
    
    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendPost('/admin/users', body, {
      'X-User-Token': token,
    });

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
        error = { detail: 'Failed to create user' };
      }
      return NextResponse.json(
        { detail: error.detail || 'Failed to create user' },
        { status: response.status }
      );
    }

    // Parse response data
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
    return NextResponse.json(data, { status: response.status || 201 });
  } catch (error) {
    console.error('Admin users create API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

