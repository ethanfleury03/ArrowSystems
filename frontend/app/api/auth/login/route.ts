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

    // Delegate authentication to backend using IAM-authenticated request
    // Backend will set JWT cookie in response
    const backendResponse = await iamBackendPost('/auth/login', { email, password });

    if (!backendResponse.ok) {
      const errorData = await backendResponse.json().catch(() => null);
      console.error(`Login failed for ${email}:`, errorData);
      
      // Handle FastAPI validation errors (array of error objects)
      let errorMessage = 'Invalid email or password';
      if (errorData) {
        if (errorData.detail) {
          if (Array.isArray(errorData.detail)) {
            // Pydantic validation errors: array of {type, loc, msg, input}
            errorMessage = errorData.detail.map((err: any) => 
              `${err.loc?.join('.') || 'field'}: ${err.msg || 'Invalid value'}`
            ).join(', ');
          } else if (typeof errorData.detail === 'string') {
            errorMessage = errorData.detail;
          }
        } else if (errorData.error && typeof errorData.error === 'string') {
          errorMessage = errorData.error;
        }
      }
      
      return NextResponse.json(
        { error: errorMessage },
        { status: backendResponse.status === 401 ? 401 : 500 }
      );
    }

    const data = await backendResponse.json();
    const { user, message } = data;
    
    if (!user) {
      return NextResponse.json(
        { error: 'Invalid response from authentication service' },
        { status: 502 }
      );
    }

    // Extract JWT token from backend's Set-Cookie header
    // Backend sets the cookie on its domain, but we need it on the frontend domain
    let jwtToken: string | null = null;
    const setCookieHeader = backendResponse.headers.get('set-cookie');
    if (setCookieHeader) {
      const match = setCookieHeader.match(/access_token=([^;]+)/);
      if (match) {
        jwtToken = match[1];
      }
    }
    
    if (!jwtToken) {
      console.error('No JWT token found in backend response');
      return NextResponse.json(
        { error: 'Authentication token not received from backend' },
        { status: 502 }
      );
    }
    
    const response = NextResponse.json(
      {
        message: message || 'Login successful',
        userId: user.id,
        role: user.role,
        user,
      },
      { status: 200 }
    );
    
    // Set JWT cookie on the frontend domain
    if (jwtToken) {
      response.cookies.set('access_token', jwtToken, {
        httpOnly: true,
        secure: process.env.NODE_ENV === 'production',
        sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
        path: '/',
        maxAge: undefined, // Session cookie - expires when browser closes
      });
    }
    
    // Only log in development to reduce production log noise
    if (process.env.NODE_ENV !== 'production') {
      console.log(`Login successful for user: ${email} (role: ${user.role})`);
    }
    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

