import { NextRequest, NextResponse } from 'next/server';

// Use BACKEND_URL from env (set in Docker) or default to localhost for local dev
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export async function GET(
  request: NextRequest,
  { params }: { params: { path: string[] } }
) {
  try {
    // Reconstruct the filename from path segments
    const filename = params.path.join('/');
    
    // Security: prevent directory traversal
    if (filename.includes('..') || filename.includes('//')) {
      return NextResponse.json(
        { detail: 'Invalid filename' },
        { status: 400 }
      );
    }
    
    // URL encode the filename for the backend request
    const encodedFilename = encodeURIComponent(filename);
    
    const response = await fetch(`${BACKEND_URL}/documents/${encodedFilename}`, {
      method: 'GET',
      headers: {
        'Accept': 'application/pdf',
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to fetch document' }));
      return NextResponse.json(
        { detail: error.detail || 'Document not found' },
        { status: response.status }
      );
    }

    // Return PDF as blob
    const blob = await response.blob();
    return new NextResponse(blob, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${filename}"`,
      },
    });
  } catch (error) {
    console.error('Document API route error:', error);
    
    // Check if it's a network error (backend not reachable)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: `Cannot connect to backend at ${BACKEND_URL}. Make sure the backend is running on port 8000.` },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

