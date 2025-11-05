import { NextRequest, NextResponse } from 'next/server';

// Registration is disabled - only admin-created accounts can access the system
export async function POST(request: NextRequest) {
  return NextResponse.json(
    { error: 'Registration is disabled. Please contact your administrator for account access.' },
    { status: 403 }
  );
}

