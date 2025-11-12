import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

const disableAuth =
  process.env.DISABLE_AUTH === 'true' ||
  process.env.NEXT_PUBLIC_DISABLE_AUTH === 'true';

export function middleware(request: NextRequest) {
  if (disableAuth) {
    return NextResponse.next();
  }

  // Get the session cookie
  const sessionCookie = request.cookies.get('app_session');
  const pathname = request.nextUrl.pathname;

  // Redirect /register to /login (registration is disabled)
  if (pathname === '/register') {
    return NextResponse.redirect(new URL('/login', request.url));
  }

  // Allow access to login page
  if (pathname === '/login') {
    if (sessionCookie) {
      // Redirect to account if already logged in
      return NextResponse.redirect(new URL('/account', request.url));
    }
    return NextResponse.next();
  }

  // Protect all other routes (including root)
  if (!sessionCookie) {
    // Redirect to login if no session
    return NextResponse.redirect(new URL('/login', request.url));
  }

  return NextResponse.next();
}

export const config = {
  matcher: [
    /*
     * Match all request paths except for the ones starting with:
     * - api (API routes)
     * - _next/static (static files)
     * - _next/image (image optimization files)
     * - favicon.ico (favicon file)
     * - public files (public folder)
     */
    '/((?!api|_next/static|_next/image|favicon.ico|.*\\.(?:svg|png|jpg|jpeg|gif|webp)$).*)',
  ],
};

