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

    // Forward Set-Cookie header from backend to browser
    // The backend sets the JWT in an HTTP-only cookie
    const setCookieHeader = backendResponse.headers.get('set-cookie');
    
    const response = NextResponse.json(
      {
        message: message || 'Login successful',
        userId: user.id,
        role: user.role,
        user,
      },
      { status: 200 }
    );
    
    // Forward the cookie from backend
    if (setCookieHeader) {
      response.headers.set('set-cookie', setCookieHeader);
    }
    
    console.log(`Login successful for user: ${email} (role: ${user.role})`);
    return response;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

