import { NextRequest, NextResponse } from 'next/server';
import { getBackendBaseUrl, getBackendIdentityToken } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

export async function POST(request: NextRequest) {
  try {
    // Extract JWT from cookie for user authentication
    const userToken = await extractJwtFromCookie();
    
    if (!userToken) {
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

    // Get IAM identity token for backend authentication
    const iamToken = await getBackendIdentityToken();
    const backendUrl = `${getBackendBaseUrl()}/admin/documents/upload`;

    // Prepare headers - forward Content-Type with boundary from client request
    const headers = new Headers();
    
    // Forward the Content-Type header with boundary from the client
    // This is critical - the boundary must match the actual body
    if (contentType) {
      headers.set('Content-Type', contentType);
    }
    
    // Add IAM authentication token
    headers.set('Authorization', `Bearer ${iamToken}`);
    
    // Add user JWT token for backend authorization
    headers.set('X-User-Token', userToken);
    
    // Remove headers that shouldn't be forwarded
    headers.delete('host');
    headers.delete('content-length'); // Let fetch calculate it from the stream

    // Get the request body stream
    // In Next.js App Router, request.body is a ReadableStream
    const requestBody = request.body;
    
    if (!requestBody) {
      return NextResponse.json(
        { detail: 'Request body is missing' },
        { status: 400 }
      );
    }

    // Forward the raw request body stream directly to backend
    // Do NOT parse it as JSON or FormData - just stream it through
    // This preserves the multipart/form-data encoding with the correct boundary
    const backendResponse = await fetch(backendUrl, {
      method: 'POST',
      headers,
      body: requestBody, // Raw ReadableStream - preserves multipart encoding
    });

    // Forward the response
    const responseBody = backendResponse.body;
    const responseHeaders = new Headers();
    
    // Copy response headers
    backendResponse.headers.forEach((value, key) => {
      responseHeaders.set(key, value);
    });

    return new NextResponse(responseBody, {
      status: backendResponse.status,
      statusText: backendResponse.statusText,
      headers: responseHeaders,
    });
  } catch (error: any) {
    console.error('Admin document upload API error:', {
      message: error.message,
      status: error.status,
      detail: error.detail,
    });
    
    const errorDetail = error.detail || error.message || 'Internal server error';
    const errorStatus = error.status || 500;
    
    return NextResponse.json(
      { detail: errorDetail },
      { status: errorStatus }
    );
  }
}
