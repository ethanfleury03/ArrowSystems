import { NextRequest, NextResponse } from 'next/server';
import { getBackendBaseUrl, getBackendIdentityToken } from '@/lib/iam-backend';
import { extractJwtFromCookie } from '@/lib/authClient';

// This route must be dynamic so we can stream the body
export const dynamic = 'force-dynamic';

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
    const backendBaseUrl = getBackendBaseUrl();
    const backendUrl = `${backendBaseUrl}/admin/documents/upload`;

    // Start from the incoming headers
    const headers = new Headers(request.headers);
    
    // Forward the Content-Type header with boundary from the client
    // This is critical - the boundary must match the actual body
    // (Already set from request.headers, but ensure it's preserved)
    
    // Add IAM authentication token
    headers.set('Authorization', `Bearer ${iamToken}`);
    
    // Add user JWT token for backend authorization
    headers.set('X-User-Token', userToken);
    
    // Update host to match backend URL
    const backendHost = new URL(backendBaseUrl).host;
    headers.set('host', backendHost);
    
    // Let fetch compute the content-length from the stream
    headers.delete('content-length');
    
    // Remove Next.js specific headers that shouldn't be forwarded
    headers.delete('x-forwarded-host');
    headers.delete('x-forwarded-port');
    headers.delete('x-forwarded-proto');

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
    // Node/undici requires duplex: 'half' when sending a stream body
    const backendResponse = await fetch(backendUrl, {
      method: 'POST',
      headers,
      body: requestBody, // Raw ReadableStream - preserves multipart encoding
      // @ts-ignore – duplex isn't in the TS types yet but is required by Node.js fetch
      duplex: 'half',
    });

    // Try to pass back JSON if possible, otherwise stream text
    const responseContentType = backendResponse.headers.get('content-type') || '';
    
    if (responseContentType.includes('application/json')) {
      const data = await backendResponse.json();
      return NextResponse.json(data, { status: backendResponse.status });
    } else {
      const text = await backendResponse.text();
      return new NextResponse(text, {
        status: backendResponse.status,
        headers: { 'content-type': responseContentType || 'text/plain' },
      });
    }
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
