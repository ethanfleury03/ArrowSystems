import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function GET(_request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();

    if (!token) {
      return NextResponse.json(
        { detail: 'Missing user token' },
        { status: 401 },
      );
    }

    const response = await iamBackendGet('/admin/documents/diagnostics', {
      'X-User-Token': token,
    });

    const contentType = response.headers.get('content-type') || '';
    console.info('[api/admin/documents/diagnostics] proxy response', {
      status: response.status,
      ok: response.ok,
      contentType,
    });

    if (!response.ok) {
      let detail = 'Failed to fetch diagnostics';
      try {
        const error = contentType.includes('application/json')
          ? await response.json()
          : { detail: (await response.text()).slice(0, 300) };
        if (error && typeof error === 'object' && 'detail' in error) {
          detail = (error as any).detail ?? detail;
        }
      } catch {
        // Ignore JSON parse errors and fall back to generic detail
      }

      return NextResponse.json({ detail }, { status: response.status });
    }

    // Defensive: never throw HTML at the client; if backend returned non-JSON, return a JSON error.
    if (!contentType.includes('application/json')) {
      const text = (await response.text()).slice(0, 300);
      return NextResponse.json(
        { detail: `Diagnostics backend returned non-JSON (${response.status}): ${text}` },
        { status: 502 },
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin documents diagnostics API error:', error);
    const detail =
      error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ detail }, { status: 500 });
  }
}

