import { NextRequest, NextResponse } from 'next/server';
import { iamBackendDelete } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function DELETE(
  request: NextRequest,
  { params }: { params: Promise<{ documentId: string }> | { documentId: string } }
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
    const documentId = resolvedParams.documentId;

    // Validate documentId is a number
    if (!documentId || isNaN(parseInt(documentId))) {
      return NextResponse.json(
        { detail: 'Invalid document ID' },
        { status: 400 }
      );
    }

    // Forward JWT in custom header to backend (X-User-Token)
    const response = await iamBackendDelete(
      `/admin/documents/orphan/${encodeURIComponent(documentId)}`,
      {
        'X-User-Token': token,
      }
    );

    if (!response.ok) {
      const error = await response.json().catch(() => ({}));
      return NextResponse.json(
        { detail: (error as any)?.detail || 'Failed to delete orphan document' },
        { status: response.status }
      );
    }

    // Return JSON response
    const data = await response.json().catch(() => ({}));
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin orphan document delete API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}
