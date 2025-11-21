import { NextRequest, NextResponse } from 'next/server';
import { GoogleAuth } from 'google-auth-library';

const BACKEND_URL = 'https://arrow-rag-backend-70705019874.us-central1.run.app';

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
    
    // Use IAM authentication for binary document download
    const auth = new GoogleAuth();
    const client = await auth.getIdTokenClient(BACKEND_URL);
    
    const response = await client.request({
      url: `${BACKEND_URL}/documents/${encodedFilename}`,
      method: 'GET',
      headers: {
        'Accept': 'application/pdf',
      },
      responseType: 'arraybuffer',
    });

    if (response.status !== 200) {
      return NextResponse.json(
        { detail: 'Document not found' },
        { status: response.status }
      );
    }

    // Return PDF as response
    return new NextResponse(response.data as ArrayBuffer, {
      headers: {
        'Content-Type': 'application/pdf',
        'Content-Disposition': `inline; filename="${filename}"`,
      },
    });
  } catch (error: any) {
    console.error('Document API route error:', error);
    
    return NextResponse.json(
      { detail: error.message || 'Internal server error' },
      { status: error.response?.status || 500 }
    );
  }
}

