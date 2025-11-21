import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

// List allowed machine models for admin UI
// Proxies to backend /admin/machine_models
export async function GET(_request: NextRequest) {
  try {
    // Extract JWT from cookie
    const token = await extractJwtFromCookie();

    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 },
      );
    }

    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendGet('/admin/machine_models', {
      'X-User-Token': token,
    });

    if (!response.ok) {
      let detail = 'Failed to fetch machine models';
      try {
        const error = await response.json();
        detail = (error as any)?.detail || detail;
      } catch {
        // ignore JSON parse errors
      }

      return NextResponse.json({ detail }, { status: response.status });
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin machine_models API error:', error);
    return NextResponse.json(
      {
        detail:
          error instanceof Error ? error.message : 'Internal server error',
      },
      { status: 500 },
    );
  }
}


