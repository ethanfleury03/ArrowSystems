import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet, iamBackendPost } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('Authorization');
    const headers = authHeader ? { 'Authorization': authHeader } : undefined;
    
    const response = await iamBackendGet('/admin/machines', headers);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to fetch machines' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin machines API error:', error);
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

    const response = await iamBackendPost('/admin/machines', body, headers);

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to create machine' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin machines API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

