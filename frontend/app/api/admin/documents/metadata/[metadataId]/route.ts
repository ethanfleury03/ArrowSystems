import { NextRequest, NextResponse } from 'next/server';
import { iamBackendDelete } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ metadataId: string }> | { metadataId: string } }
) {
  try {
    // Extract JWT from cookie
    const token = await extractJwtFromCookie();
    
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

    // Handle params (Next.js 15+ uses Promise, older versions use object)
    const resolvedParams = params instanceof Promise ? await params : params;

    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendDelete(
      `/admin/documents/metadata/${encodeURIComponent(resolvedParams.metadataId)}`,
      {
        'X-User-Token': token,
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { detail: (error as any)?.detail || 'Failed to delete document' },
        { status: response.status }
      );
    }

    // Handle 204 No Content (no body)
    if (response.status === 204) {
      // Pass through warning header if present
      const headers: Record<string, string> = {};
      const indexWarning = response.headers.get("X-Index-Warning");
      if (indexWarning) {
        headers["X-Index-Warning"] = indexWarning;
      }
      return new NextResponse(null, { status: 204, headers });
    }

    // For other success statuses, return JSON
    const data = await response.json().catch(() => ({}));
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin document delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

