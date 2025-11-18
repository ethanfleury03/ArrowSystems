import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const machine_model = formData.get('machine_model') as string;
    const description = formData.get('description') as string | null;

    if (!file) {
      return NextResponse.json(
        { detail: 'No file provided' },
        { status: 400 }
      );
    }

    if (!machine_model) {
      return NextResponse.json(
        { detail: 'Machine model is required' },
        { status: 400 }
      );
    }

    // Create FormData for backend
    const backendFormData = new FormData();
    backendFormData.append('file', file);
    backendFormData.append('machine_model', machine_model);
    if (description) {
      backendFormData.append('description', description);
    }

    const response = await fetch(`${BACKEND_URL}/admin/documents/upload`, {
      method: 'POST',
      headers: {
        'Authorization': request.headers.get('Authorization') || '',
      },
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

