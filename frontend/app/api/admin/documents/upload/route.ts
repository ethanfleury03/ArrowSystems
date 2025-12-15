import { NextRequest, NextResponse } from 'next/server';
import { GoogleAuth } from 'google-auth-library';
import FormData from 'form-data';
import { extractJwtFromCookie } from '@/lib/authClient';

const BACKEND_URL = process.env.NEXT_PUBLIC_API_URL || process.env.BACKEND_URL || 'http://localhost:8080';

if (!BACKEND_URL) {
  throw new Error('NEXT_PUBLIC_API_URL or BACKEND_URL environment variable must be set');
}

export async function POST(request: NextRequest) {
  try {
    // Extract JWT from cookie for authentication
    const token = await extractJwtFromCookie();
    
    if (!token) {
      return NextResponse.json(
        { detail: 'Not authenticated' },
        { status: 401 }
      );
    }

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

    // Use IAM authentication for multipart upload
    const auth = new GoogleAuth();
    const client = await auth.getIdTokenClient(BACKEND_URL);

    // Convert File to Buffer for form-data library
    const fileBuffer = Buffer.from(await file.arrayBuffer());
    
    // Create form-data with form-data library (compatible with gaxios)
    const backendFormData = new FormData();
    backendFormData.append('file', fileBuffer, {
      filename: file.name,
      contentType: file.type,
    });
    backendFormData.append('machine_model', String(machine_model).trim());
    
    const description = formData.get('description');
    if (description) {
      const descStr = String(description).trim();
      if (descStr) {
        backendFormData.append('description', descStr);
      }
    }

    // Get headers from FormData (includes Content-Type with boundary)
    const formDataHeaders = backendFormData.getHeaders();
    
    // Forward JWT in custom header to backend (X-User-Token)
    // IMPORTANT: Do NOT override Content-Type - let FormData set it with the correct boundary
    const headers: any = {
      ...formDataHeaders,
      'X-User-Token': token,
    };

    // Log request config in dev (without the actual file data)
    if (process.env.NODE_ENV === 'development') {
      console.log('Upload request config:', {
        url: `${BACKEND_URL}/admin/documents/upload`,
        method: 'POST',
        hasData: !!backendFormData,
        dataType: backendFormData.constructor.name,
        contentType: headers['content-type'],
        hasBody: false, // Confirm we're NOT setting body
      });
    }

    // Make request with FormData as data (NOT body)
    // gaxios will handle the multipart encoding automatically
    const response = await client.request({
      url: `${BACKEND_URL}/admin/documents/upload`,
      method: 'POST',
      headers,
      data: backendFormData,
      // DO NOT set 'body' property - only use 'data'
    });

    return NextResponse.json(response.data);
  } catch (error: any) {
    console.error('Admin document upload API error:', {
      message: error.message,
      status: error.response?.status || error.status,
      statusText: error.response?.statusText || error.statusText,
      detail: error.response?.data?.detail || error.detail,
      // Don't log the full response data as it might contain file content
    });
    
    const errorDetail = error.response?.data?.detail || error.detail || error.message || 'Internal server error';
    const errorStatus = error.response?.status || error.status || 500;
    
    return NextResponse.json(
      { detail: errorDetail },
      { status: errorStatus }
    );
  }
}
