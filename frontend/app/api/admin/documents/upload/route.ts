import { NextRequest, NextResponse } from 'next/server';
import { GoogleAuth } from 'google-auth-library';
import FormData from 'form-data';
import { extractJwtFromCookie } from '@/lib/authClient';

const BACKEND_URL = 'https://arrow-rag-backend-70705019874.us-central1.run.app';

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

    // Forward JWT in custom header to backend (X-User-Token)
    const headers: any = {
      ...backendFormData.getHeaders(),
      'X-User-Token': token,
    };

    const response = await client.request({
      url: `${BACKEND_URL}/admin/documents/upload`,
      method: 'POST',
      headers,
      data: backendFormData,
    });

    return NextResponse.json(response.data);
  } catch (error: any) {
    console.error('Admin document upload API error:', error);
    return NextResponse.json(
      { detail: error.message || 'Internal server error' },
      { status: error.response?.status || 500 }
    );
  }
}
