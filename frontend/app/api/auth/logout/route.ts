import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

export async function POST(request: NextRequest) {
  try {
    // Call backend logout endpoint
    // Backend will clear the JWT cookie
    const backendResponse = await iamBackendPost('/auth/logout', {});

    if (!backendResponse.ok) {
      console.error('Logout failed on backend');
      // Even if backend fails, clear the cookie on frontend
    }

    // Forward Set-Cookie header from backend (cookie clearing)
    const setCookieHeader = backendResponse.headers.get('set-cookie');
    
    const response = NextResponse.json(
      { message: 'Logged out successfully' },
      { status: 200 }
    );
    
    // Forward cookie-clearing headers from backend
    if (setCookieHeader) {
      response.headers.set('set-cookie', setCookieHeader);
    }
    
    return response;
  } catch (error) {
    console.error('Logout error:', error);
    // Even on error, return success for UX
    return NextResponse.json(
      { message: 'Logged out successfully' },
      { status: 200 }
    );
  }
}
