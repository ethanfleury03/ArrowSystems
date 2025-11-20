import { NextRequest, NextResponse } from 'next/server';

// Use BACKEND_URL from env (set in Docker) or default to localhost for local dev
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8080';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const user = searchParams.get('user') || 'api_user';
    const limit = searchParams.get('limit') || '50';
    
    const response = await fetch(`${BACKEND_URL}/history?user=${user}&limit=${limit}`, {
      method: 'GET',
      headers: {
        'Content-Type': 'application/json',
      },
    });

    if (!response.ok) {
      const error = await response.json();
      return NextResponse.json(
        { detail: error.detail || 'Backend request failed' },
        { status: response.status }
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('API route error:', error);
    
    // Check if it's a network error (backend not reachable)
    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: `Cannot connect to backend at ${BACKEND_URL}. Make sure the backend is running on the configured backend port.` },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

