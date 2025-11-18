import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function POST(request: NextRequest) {
  try {
    // Validate content type
    const contentType = request.headers.get('content-type');
    
    if (!contentType || !contentType.includes('multipart/form-data')) {
      return NextResponse.json(
        { detail: 'Invalid content type. Expected multipart/form-data' },
        { status: 400 }
      );
    }

    // Parse form data to validate required fields
    const formData = await request.formData();
    const file = formData.get('file') as File;
    const machine_model = formData.get('machine_model');
    
    if (!file) {
      return NextResponse.json(
        { detail: 'No file provided' },
        { status: 400 }
      );
    }

    if (!machine_model || (typeof machine_model === 'string' && machine_model.trim() === '')) {
      return NextResponse.json(
        { detail: 'Machine model is required' },
        { status: 400 }
      );
    }

    // Recreate FormData for backend forwarding
    // CRITICAL: In Node.js, we must explicitly convert all text fields to strings
    // and include the filename for File objects
    const backendFormData = new FormData();
    
    // Append file with explicit filename
    backendFormData.append('file', file, file.name);
    
    // Append machine_model - MUST be a string
    backendFormData.append('machine_model', String(machine_model).trim());
    
    // Append description if provided
    const description = formData.get('description');
    if (description) {
      const descStr = String(description).trim();
      if (descStr) {
        backendFormData.append('description', descStr);
      }
    }

    // Forward to backend
    // CRITICAL: Do NOT set Content-Type header manually
    // fetch() will automatically set it with the correct multipart boundary
    const response = await fetch(`${BACKEND_URL}/admin/documents/upload`, {
      method: 'POST',
      headers: {
        'Authorization': request.headers.get('Authorization') || '',
        // DO NOT set Content-Type - fetch handles multipart/form-data automatically
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
