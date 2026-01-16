import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function POST(request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json(
        { detail: 'Authentication required' },
        { status: 401 }
      );
    }

    const response = await iamBackendPost(
      '/admin/ticket-index/reindex',
      {},
      { 'X-User-Token': token }
    );

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to trigger reindex' }));
      return NextResponse.json(
        errorData,
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Error triggering ticket index reindex:', error);
    return NextResponse.json(
      { detail: 'Internal server error' },
      { status: 500 }
    );
  }
}
