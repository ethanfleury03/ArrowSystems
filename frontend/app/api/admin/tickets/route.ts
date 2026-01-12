import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();

    if (!token) {
      return NextResponse.json(
        { detail: 'Missing user token' },
        { status: 401 },
      );
    }

    // Extract query parameters
    const { searchParams } = new URL(request.url);
    const page = searchParams.get('page') || '1';
    const page_size = searchParams.get('page_size') || '50';
    const q = searchParams.get('q') || undefined;
    const sort = searchParams.get('sort') || 'judged_at DESC';

    // Build query string
    const queryParams = new URLSearchParams({
      page,
      page_size,
      ...(q && { q }),
      sort,
    });

    const response = await iamBackendGet(`/admin/tickets?${queryParams.toString()}`, {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let detail = 'Failed to fetch tickets';
      try {
        const error = await response.json();
        if (error && typeof error === 'object' && 'detail' in error) {
          detail = (error as any).detail ?? detail;
        }
      } catch {
        // Ignore JSON parse errors and fall back to generic detail
      }

      return NextResponse.json({ detail }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin tickets API error:', error);
    const detail =
      error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ detail }, { status: 500 });
  }
}
