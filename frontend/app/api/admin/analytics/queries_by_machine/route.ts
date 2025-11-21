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

    const response = await iamBackendGet(`/admin/analytics/queries_by_machine${search}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let errorBody: any = null;
      try {
        errorBody = await response.json();
      } catch {}

      console.error('IAM Backend Request Error (queries_by_machine):', {
        path: `/admin/analytics/queries_by_machine${search}`,
        status: response.status,
        error: errorBody,
      });

      return NextResponse.json(
        errorBody ?? { detail: 'Failed to load queries_by_machine' },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Admin analytics queries_by_machine API error:', error);
    return NextResponse.json(
      { detail: 'Failed to load queries_by_machine' },
      { status: 500 },
    );
  }
}


