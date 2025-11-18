import { NextRequest, NextResponse } from 'next/server';
import { setLoginSession } from '@/lib/auth';
import { getBackendUrl } from '@/lib/backend-url';

export async function POST(request: NextRequest) {
  try {
    // Detect backend URL from request hostname (for network access)
    const BACKEND_URL = getBackendUrl(request);
    
    const body = await request.json();
    const { email, password } = body;

    // Basic validation
    if (!email || !password) {
      return NextResponse.json(
        { error: 'Email and password are required' },
        { status: 400 }
      );
    }

    // Delegate authentication to backend
    const backendResponse = await fetch(`${BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

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

    const { user, token } = await backendResponse.json() as { user: { id: string; role: string }; token: string };
    if (!user || !token) {
      return NextResponse.json(
        { error: 'Invalid response from authentication service' },
        { status: 502 }
      );
    }

    // Create response and set session
    const response = NextResponse.json(
      {
        message: 'Login successful',
        userId: user.id,
        role: user.role,
        user,
        token,
      },
      { status: 200 }
    );
    
    const sessionResponse = await setLoginSession(user.id, request, response);
    console.log(`Login successful for user: ${email} (role: ${user.role})`);
    return sessionResponse;
  } catch (error) {
    console.error('Login error:', error);
    return NextResponse.json(
      { error: 'Internal server error', details: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

