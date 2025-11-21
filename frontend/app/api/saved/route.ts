import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet, iamBackendPost } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const limit = searchParams.get('limit') || '50';
    const min_helpful_count = searchParams.get('min_helpful_count') || '2';
    
    const response = await iamBackendGet(`/saved?limit=${limit}&min_helpful_count=${min_helpful_count}`);

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
    
    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    );
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await iamBackendPost('/saved', body)

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

    return NextResponse.json(
      { detail: error instanceof Error ? error.message : 'Internal server error' },
      { status: 500 }
    )
  }
}

