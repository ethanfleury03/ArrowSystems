import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { email, password } = body;

    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      );
    }

    // Delegate authentication to backend using IAM-authenticated request.
    // Backend sets the auth cookie in its response; we mirror that cookie on
    // the frontend domain so the browser sends it to /api/* routes.
    const backendResponse = await iamBackendPost('/auth/login', { email, password });

    if (!backendResponse.ok) {
      let errorBody: any = null;
      try {
        errorBody = await backendResponse.json();
      } catch {
        // ignore JSON parse errors; fall back to generic error
      }

      console.error('IAM Backend Request Error (login):', {
        path: '/auth/login',
        status: backendResponse.status,
        error: errorBody,
      });

      // Improve error messages for 503 (cold start / initialization)
      let errorMessage = errorBody?.detail || errorBody?.error || 'Backend request failed';
      if (backendResponse.status === 503) {
        errorMessage = errorBody?.detail || 'Service temporarily unavailable. The server is starting up. Please try again in a few seconds.';
      }
      
      // Forward backend status code and body directly where possible so the
      // browser sees 401/403/503 instead of a generic 500 from this route.
      return NextResponse.json(
        errorBody ? { ...errorBody, detail: errorMessage } : { detail: errorMessage },
        { status: backendResponse.status },
      );
    }

    const data = await backendResponse.json();
    const { user, message } = data;
    
    if (!user) {
      console.error('Login backend response missing user object:', data);
      return NextResponse.json(
        { detail: 'Invalid response from authentication service' },
        { status: 502 }
      );
    }

    // Backend sets the access_token cookie in its response.
    // Since backend and frontend are on the same domain (or properly configured for cross-domain),
    // we forward the Set-Cookie header from backend to browser.
    // DO NOT set cookies here - backend is the single source of truth for cookie setting.
    const response = NextResponse.json(
      {
        message: message || 'Login successful',
        userId: user.id,
        role: user.role,
        user,
      },
      { status: 200 }
    );
    
    // Forward Set-Cookie header from backend response to browser
    // This ensures the cookie set by backend reaches the browser
    const setCookieHeader = backendResponse.headers.get('set-cookie');
    if (setCookieHeader) {
      response.headers.set('set-cookie', setCookieHeader);
    } else {
      console.warn('No Set-Cookie header in backend login response - cookie may not be set');
    }
    
    // Only log in development to reduce production log noise
    if (process.env.NODE_ENV !== 'production') {
      console.log(`Login successful for user: ${email} (role: ${user.role})`);
    }
    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      {
        detail: 'Internal server error',
        error: error instanceof Error ? error.message : 'Unknown error',
      },
      { status: 500 }
    );
  }
}

