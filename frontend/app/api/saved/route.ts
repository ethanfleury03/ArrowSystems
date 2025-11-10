import { NextRequest, NextResponse } from 'next/server';

// Use BACKEND_URL from env (set in Docker) or default to localhost for local dev
const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '50';
    const min_helpful_count = searchParams.get('min_helpful_count') || '2';
    
    const response = await fetch(`${BACKEND_URL}/saved?limit=${limit}&min_helpful_count=${min_helpful_count}`, {
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
        { detail: `Cannot connect to backend at ${BACKEND_URL}. Make sure the backend is running on port 8000.` },
        { status: 503 }
      );
    }
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${BACKEND_URL}/saved`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      let detail = 'Backend request failed'
      try {
        const error = await response.json()
        detail = error.detail || detail
      } catch {
        // ignore parse errors
      }

      return NextResponse.json({ detail }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('API route error:', error)

    if (error instanceof TypeError && error.message.includes('fetch')) {
      return NextResponse.json(
        { detail: `Cannot connect to backend at ${BACKEND_URL}. Make sure the backend is running on port 8000.` },
        { status: 503 }
      )
    }

    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

