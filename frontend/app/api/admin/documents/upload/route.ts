import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;

    if (!file) {
      return NextResponse.json(
        { detail: 'No file provided' },
        { status: 400 }
      );
    }

    // Create FormData for backend
    const backendFormData = new FormData();
    backendFormData.append('file', file);

    const response = await fetch(`${BACKEND_URL}/admin/documents/upload`, {
      method: 'POST',
      body: backendFormData,
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Failed to upload document' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Admin document upload API error:', error);
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

