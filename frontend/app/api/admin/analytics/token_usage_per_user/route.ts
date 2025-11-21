import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json({ detail: 'Missing user token' }, { status: 401 });
    }

    const search = request.nextUrl.search;

    const response = await iamBackendGet(`/admin/analytics/token_usage_per_user${search}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let errorBody: any = null;
      try {
        errorBody = await response.json();
      } catch {}

      console.error('IAM Backend Request Error (token_usage_per_user):', {
        path: `/admin/analytics/token_usage_per_user${search}`,
        status: response.status,
        error: errorBody,
      });

      return NextResponse.json(
        errorBody ?? { detail: 'Failed to load token_usage_per_user' },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Admin analytics token_usage_per_user API error:', error);
    return NextResponse.json(
      { detail: 'Failed to load token_usage_per_user' },
      { status: 500 },
    );
  }
}


