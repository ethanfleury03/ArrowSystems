import { NextRequest, NextResponse } from 'next/server';
import { setLoginSession } from '@/lib/auth';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

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

    // Delegate authentication to backend
    const backendResponse = await fetch(`${BACKEND_URL}/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ email, password }),
    });

    if (!backendResponse.ok) {
      const detail = await backendResponse.json().catch(() => null);
      console.error(`Login failed for ${email}:`, detail);
      return NextResponse.json(
        { error: detail?.detail || 'Invalid email or password' },
        { status: backendResponse.status === 401 ? 401 : 500 }
      );
    }

    const { user } = await backendResponse.json() as { user: { id: string; role: string } };

    // Create response and set session
    const response = NextResponse.json(
      { 
        message: 'Login successful', 
        userId: user.id,
        role: user.role  // Include role for frontend redirect logic
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

