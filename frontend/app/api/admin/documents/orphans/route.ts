import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet, iamBackendDelete } from '@/lib/iam-backend';
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

    const response = await iamBackendGet('/admin/documents/orphans', {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let detail = 'Failed to fetch orphans';
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
    console.error('Admin documents orphans API error:', error);
    const detail =
      error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ detail }, { status: 500 });
  }
}

export async function DELETE(_request: NextRequest) {
  try {
    const token = await extractJwtFromCookie();

    if (!token) {
      return NextResponse.json(
        { detail: 'Missing user token' },
        { status: 401 },
      );
    }

    const response = await iamBackendDelete('/admin/documents/orphans', {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let detail = 'Failed to delete orphans';
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
    console.error('Admin documents orphans delete API error:', error);
    const detail =
      error instanceof Error ? error.message : 'Internal server error';
    return NextResponse.json({ detail }, { status: 500 });
  }
}

