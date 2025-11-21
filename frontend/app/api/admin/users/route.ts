import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet, iamBackendPost, iamBackendPut, iamBackendDelete } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('Authorization');
    const headers = authHeader ? { 'Authorization': authHeader } : undefined;
    
    const response = await iamBackendGet('/admin/users', headers);

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
    const authHeader = request.headers.get('Authorization');
    const headers = authHeader ? { 'Authorization': authHeader } : undefined;

    const response = await iamBackendPost('/admin/users', body, headers);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to create user' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin users create API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

