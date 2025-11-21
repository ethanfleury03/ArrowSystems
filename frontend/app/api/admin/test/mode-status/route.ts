import { NextRequest, NextResponse } from 'next/server';
import { iamBackendGet } from '@/lib/iam-backend';

export async function GET(request: NextRequest) {
  try {
    const response = await iamBackendGet('/admin/test/mode-status');

    if (!response.ok) {
      return NextResponse.json(
        { test_mode: false },
        { status: 200 }  // Default to false if endpoint doesn't exist
      );
    }

    const data = await response.json();
    return NextResponse.json(data);
  } catch (error) {
    console.error('Test mode status API error:', error);
    return NextResponse.json(
      { test_mode: false },
      { status: 200 }  // Default to false on error
    );
  }
}

