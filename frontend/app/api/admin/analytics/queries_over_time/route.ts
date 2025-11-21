import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();
    if (!token) {
      return NextResponse.json({ detail: 'Missing user token' }, { status: 401 });
    }

    const search = request.nextUrl.search; // includes leading "?" if present

    const response = await iamBackendGet(`/admin/analytics/queries_over_time${search}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let errorBody: any = null;
      try {
        errorBody = await response.json();
      } catch {
        // ignore JSON parse errors
      }

      console.error('IAM Backend Request Error (queries_over_time):', {
        path: `/admin/analytics/queries_over_time${search}`,
        status: response.status,
        error: errorBody,
      });

      return NextResponse.json(
        errorBody ?? { detail: 'Failed to load queries_over_time' },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Admin analytics queries_over_time API error:', error);
    return NextResponse.json(
      { detail: 'Failed to load queries_over_time' },
      { status: 500 },
    );
  }
}


