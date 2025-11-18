import { NextRequest, NextResponse } from 'next/server';

const BACKEND_URL = process.env.BACKEND_URL || process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export async function GET(request: NextRequest) {
  try {
    // Check if test mode is enabled by checking environment variable
    // Note: This is a client-side check, the backend also validates
    const testMode = process.env.TEST_MODE === 'true' || process.env.NEXT_PUBLIC_TEST_MODE === 'true';
    
    return NextResponse.json({ test_mode: testMode });
  } catch (error) {
    console.error('Test mode check error:', error);
    return NextResponse.json(
      { test_mode: false, error: error instanceof Error ? error.message : 'Unknown error' },
      { status: 500 }
    );
  }
}

