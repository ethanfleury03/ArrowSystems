import { NextRequest, NextResponse } from 'next/server';
import { iamBackendPost } from '@/lib/iam-backend';

export async function POST(request: NextRequest) {
  try {
    // Call backend logout endpoint (optional - backend also clears its cookie)
    const backendResponse = await iamBackendPost('/auth/logout', {});

    if (!backendResponse.ok) {
      console.error('Logout failed on backend');
      // Even if backend fails, clear the cookie on frontend
    }

    const response = NextResponse.json(
      { message: 'Logged out successfully' },
      { status: 200 }
    );
    
    // Clear the frontend's JWT cookie
    // Use set with max-age=0 to ensure it's cleared across all browsers
    response.cookies.set('access_token', '', {
      httpOnly: true,
      secure: process.env.NODE_ENV === 'production',
      sameSite: process.env.NODE_ENV === 'production' ? 'none' : 'lax',
      path: '/',
      maxAge: 0,
    });
    
    return response;
  } catch (error) {
    console.error('Logout error:', error);
    // Even on error, return success for UX
    return NextResponse.json(
      { message: 'Logged out successfully' },
      { status: 200 }
    );
  }
}
