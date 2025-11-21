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

    const response = await iamBackendGet(`/admin/analytics/document_usage${search}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let errorBody: any = null;
      try {
        errorBody = await response.json();
      } catch {}

      console.error('IAM Backend Request Error (document_usage):', {
        path: `/admin/analytics/document_usage${search}`,
        status: response.status,
        error: errorBody,
      });

      return NextResponse.json(
        errorBody ?? { detail: 'Failed to load document_usage' },
        { status: response.status },
      );
    }

    const data = await response.json();
    return NextResponse.json(data, { status: 200 });
  } catch (error) {
    console.error('Admin analytics document_usage API error:', error);
    return NextResponse.json(
      { detail: 'Failed to load document_usage' },
      { status: 500 },
    );
  }
}


